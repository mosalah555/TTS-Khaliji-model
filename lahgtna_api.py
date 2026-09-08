"""Flask API around the fine-tuned lahgtna Gulf-Arabic TTS model.

Nothing is written to disk: reference audio is decoded in memory and the
generated speech comes back inside the response.

Run:
    ./.venv312/bin/pip install flask
    ./.venv312/bin/python lahgtna_api.py

Env overrides:
    LAHGTNA_MODEL   path to the checkpoint  (default "./lahgtna finetuned/final_model")
    LAHGTNA_REF     default reference wav   (default "input.wav")
    PORT / HOST     bind address            (default 0.0.0.0:8000)
"""

import base64
import hashlib
import io
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file
from omnivoice import OmniVoice

MODEL_ID = os.environ.get("LAHGTNA_MODEL", "./lahgtna finetuned/final_model")
DEFAULT_REF_AUDIO = os.environ.get("LAHGTNA_REF", "input.wav")

# OmniVoice has no generic "Arabic" language id, and an unknown name silently
# falls back to language-agnostic mode. Gulf Arabic is "afb"; other options are
# "ars" (Najdi), "acw" (Hijazi), "arb" (Standard).
DEFAULT_LANGUAGE = "Gulf Arabic"
LANGUAGES = ["Gulf Arabic", "Najdi Arabic", "Hijazi Arabic", "Standard Arabic"]

# Voice-design items must come from OmniVoice's fixed vocabulary
# (gender: male/female; age: child/teenager/young adult/middle-aged/elderly).
DEFAULT_INSTRUCT = "male, young adult"
GENDERS = ["male", "female"]
AGES = ["child", "teenager", "young adult", "middle-aged", "elderly"]

MAX_TEXT_CHARS = 2000
MAX_BATCH_ITEMS = 16

app = Flask(__name__)

# The model is not thread-safe, so every generate() call is serialised.
_model = None
_model_lock = threading.Lock()
_gpu_lock = threading.Lock()
_voice_cache = {}
_device = _dtype = None


class ApiError(Exception):
    def __init__(self, code, message, status=400, field=None):
        super().__init__(message)
        self.code, self.message, self.status, self.field = code, message, status, field


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def get_model():
    """Load the checkpoint on first use so the server still starts without it."""
    global _model, _device, _dtype
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not Path(MODEL_ID).exists():
                raise ApiError(
                    "model_not_found",
                    f"Checkpoint {MODEL_ID!r} does not exist. Finish training or set "
                    "LAHGTNA_MODEL to an existing checkpoint.",
                    status=503,
                )
            _device, _dtype = pick_device()
            _model = OmniVoice.from_pretrained(
                MODEL_ID, device_map=_device, dtype=_dtype
            )
            print(f"loaded {MODEL_ID} device={_device} sr={_model.sampling_rate}")
    return _model


def get_voice_prompt(model, ref_audio, ref_text):
    """Encode a reference voice once and reuse it across requests.

    ref_audio is either a path or an in-memory (waveform, sample_rate) tuple;
    the cache key is the path or the audio's content hash.
    """
    key = (ref_audio[0] if isinstance(ref_audio, tuple) else ref_audio, ref_text)
    if key in _voice_cache:
        return _voice_cache[key]
    if isinstance(ref_audio, tuple):
        source = ref_audio[1]
    else:
        if not Path(ref_audio).exists():
            raise ApiError("ref_audio_not_found", f"{ref_audio!r} does not exist",
                           field="ref_audio")
        source = str(Path(ref_audio).resolve())
    prompt = model.create_voice_clone_prompt(ref_audio=source, ref_text=ref_text)
    _voice_cache[key] = prompt
    return prompt


def wav_bytes(audio, sampling_rate):
    buf = io.BytesIO()
    sf.write(buf, audio, sampling_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #

def require_json():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("invalid_json", "Body must be a JSON object")
    return body


def clean_text(value, field="text"):
    if not isinstance(value, str) or not value.strip():
        raise ApiError("invalid_request", f"{field!r} must be a non-empty string", field=field)
    text = value.strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ApiError(
            "text_too_long",
            f"{field!r} is {len(text)} chars, max is {MAX_TEXT_CHARS}",
            field=field,
        )
    return text


def clean_instruct(body):
    """Accept either a plain string or {"gender": ..., "age": ...}."""
    value = body.get("instruct", DEFAULT_INSTRUCT)
    if isinstance(value, dict):
        gender = value.get("gender", "male")
        age = value.get("age", "young adult")
        if gender not in GENDERS:
            raise ApiError("invalid_instruct", f"gender must be one of {GENDERS}", field="instruct")
        if age not in AGES:
            raise ApiError("invalid_instruct", f"age must be one of {AGES}", field="instruct")
        return f"{gender}, {age}"
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("invalid_instruct", "instruct must be a string or object", field="instruct")
    return value


def clean_number(body, field, low, high, default=None):
    value = body.get(field, default)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ApiError("invalid_request", f"{field!r} must be a number", field=field)
    if not low <= value <= high:
        raise ApiError("invalid_request", f"{field!r} must be within [{low}, {high}]", field=field)
    return float(value)


def decode_inline_ref(raw, field):
    """Turn raw audio-file bytes into a (key, (waveform, sample_rate)) pair.

    OmniVoice takes an in-memory waveform, so nothing touches the filesystem.
    """
    try:
        waveform, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except Exception:
        raise ApiError("invalid_audio", f"{field!r} is not a readable audio file", field=field)
    if waveform.size == 0:
        raise ApiError("invalid_audio", f"{field!r} contains no audio samples", field=field)
    key = hashlib.sha256(raw).hexdigest()
    return (key, (waveform.T, sr))  # (T, C) -> (C, T)


def resolve_reference(body):
    """ref_audio may be a file path, a data: URI, or a bare base64 payload."""
    ref_text = body.get("ref_text")
    if ref_text is not None and not isinstance(ref_text, str):
        raise ApiError("invalid_request", "ref_text must be a string or null", field="ref_text")

    field = "ref_audio"
    value = body.get("ref_audio")
    if not value:  # accepted alias, kept so older clients keep working
        field, value = "ref_audio_base64", body.get("ref_audio_base64")
    if not value:
        return DEFAULT_REF_AUDIO, ref_text
    if not isinstance(value, str):
        raise ApiError("invalid_request", f"{field!r} must be a string", field=field)

    if value.startswith("data:"):
        _, _, payload = value.partition(",")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception:
            raise ApiError("invalid_base64", f"{field!r} data: URI is not valid base64",
                           field=field)
        return decode_inline_ref(raw, field), ref_text

    # A short string that names a real file is a path; anything else that
    # decodes cleanly is treated as inline base64 audio.
    if len(value) < 4096 and Path(value).exists():
        return value, ref_text
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception:
        raise ApiError("ref_audio_not_found",
                       f"{value[:80]!r} is neither an existing file nor valid base64 audio",
                       field=field)
    return decode_inline_ref(raw, field), ref_text


def synthesize(texts, body):
    # Validate every field before touching the model, so a bad request is a 400
    # rather than a 503 from a checkpoint that was never needed.
    language = body.get("language", DEFAULT_LANGUAGE)
    if language is not None and not isinstance(language, str):
        raise ApiError("invalid_request", "language must be a string or null", field="language")

    kwargs = dict(
        language=language,
        instruct=clean_instruct(body),
        normalize_text=bool(body.get("normalize_text", False)),
    )
    speed = clean_number(body, "speed", 0.25, 4.0)
    duration = clean_number(body, "duration", 0.1, 120.0)
    if speed is not None:
        kwargs["speed"] = speed
    if duration is not None:
        kwargs["duration"] = duration

    model = get_model()
    ref_audio, ref_text = resolve_reference(body)
    voice_prompt = get_voice_prompt(model, ref_audio, ref_text)

    tick = time.time()
    with _gpu_lock:
        audios = model.generate(text=texts, voice_clone_prompt=voice_prompt, **kwargs)
    return audios, model.sampling_rate, time.time() - tick


def package(audio, sampling_rate):
    """The generated speech, base64 wav, inline in the response."""
    audio = np.asarray(audio)
    return {
        "encoding": "base64",
        "format": "wav",
        "sample_rate": sampling_rate,
        "duration_sec": round(len(audio) / sampling_rate, 3),
        "data": base64.b64encode(wav_bytes(audio, sampling_rate)).decode(),
    }


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #

@app.errorhandler(ApiError)
def handle_api_error(err):
    body = {"ok": False, "error": {"code": err.code, "message": err.message}}
    if err.field:
        body["error"]["field"] = err.field
    return jsonify(body), err.status


@app.errorhandler(Exception)
def handle_unexpected(err):
    app.logger.exception("unhandled error")
    return jsonify({
        "ok": False,
        "error": {"code": "internal_error", "message": str(err)},
    }), 500


@app.get("/health")
def health():
    """GET /health -> is the checkpoint present and is the model warm?"""
    return jsonify({
        "ok": True,
        "model_path": MODEL_ID,
        "model_present": Path(MODEL_ID).exists(),
        "model_loaded": _model is not None,
        "device": _device,
        "sample_rate": _model.sampling_rate if _model is not None else None,
    })


@app.get("/options")
def options():
    """GET /options -> the enumerations a client may send."""
    return jsonify({
        "ok": True,
        "languages": LANGUAGES,
        "default_language": DEFAULT_LANGUAGE,
        "genders": GENDERS,
        "ages": AGES,
        "default_instruct": DEFAULT_INSTRUCT,
        "default_ref_audio": DEFAULT_REF_AUDIO,
        "ref_audio_accepts": ["file path", "data: URI", "bare base64 audio"],
        "max_text_chars": MAX_TEXT_CHARS,
        "max_batch_items": MAX_BATCH_ITEMS,
        "output_formats": ["base64", "wav"],
    })


@app.post("/warmup")
def warmup():
    """POST /warmup -> load the checkpoint ahead of the first real request."""
    model = get_model()
    return jsonify({"ok": True, "device": _device, "sample_rate": model.sampling_rate})


@app.post("/tts")
def tts():
    """POST /tts -> {text, ref_audio, instruct} in, the audio back in the response."""
    body = require_json()
    text = clean_text(body.get("text"))
    want = body.get("format", "base64")
    if want not in ("base64", "wav"):
        raise ApiError("invalid_format", "format must be base64 or wav", field="format")

    audios, sampling_rate, elapsed = synthesize(text, body)
    audio = np.asarray(audios[0])

    if want == "wav":
        return send_file(
            io.BytesIO(wav_bytes(audio, sampling_rate)),
            mimetype="audio/wav",
            as_attachment=True,
            download_name="lahgtna.wav",
        )

    return jsonify({
        "ok": True,
        "request_id": uuid.uuid4().hex,
        "audio": package(audio, sampling_rate),
        "meta": {
            "text": text,
            "language": body.get("language", DEFAULT_LANGUAGE),
            "instruct": clean_instruct(body),
            "model": MODEL_ID,
            "device": _device,
        },
        "timings": {"generate_sec": round(elapsed, 3)},
    })


@app.post("/tts/batch")
def tts_batch():
    """POST /tts/batch -> synthesize several utterances in one model call."""
    body = require_json()
    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise ApiError("invalid_request", "'items' must be a non-empty array", field="items")
    if len(items) > MAX_BATCH_ITEMS:
        raise ApiError("batch_too_large",
                       f"at most {MAX_BATCH_ITEMS} items per request", field="items")

    texts, ids = [], []
    for i, item in enumerate(items):
        if isinstance(item, str):
            texts.append(clean_text(item, f"items[{i}]"))
            ids.append(str(i))
        elif isinstance(item, dict):
            texts.append(clean_text(item.get("text"), f"items[{i}].text"))
            ids.append(str(item.get("id", i)))
        else:
            raise ApiError("invalid_request",
                           f"items[{i}] must be a string or object", field=f"items[{i}]")

    audios, sampling_rate, elapsed = synthesize(texts, body)
    return jsonify({
        "ok": True,
        "request_id": uuid.uuid4().hex,
        "count": len(audios),
        "results": [
            {"id": item_id, "text": text, "audio": package(audio, sampling_rate)}
            for item_id, text, audio in zip(ids, texts, audios)
        ],
        "timings": {"generate_sec": round(elapsed, 3)},
    })


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        threaded=True,
    )
