from __future__ import annotations

import argparse
import json
import struct
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

DEFAULT_REPO = "kyutai/stt-1b-en_fr-mlx"
MODEL_RATE = 24000
STEP_MS = 80
STEP = MODEL_RATE * STEP_MS // 1000
IN_RATE = 16000
MAX_UTTERANCE_SECONDS = 60
SILENCE_AFTER_SECONDS = 1.0
SILENT_TEXT_TOKENS = (0, 3)
HEADER_BYTES = 5


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read_exactly(n: int) -> bytes | None:
    data = b""

    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))

        if not chunk:
            return None

        data += chunk

    return data


def load_language_model(repo: str, config: dict, lm_config) -> models.Lm:
    weights = hf_hub_download(repo, config.get("moshi_name", "model.safetensors"))
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    quantize_to_match(model, weights)
    model.load_weights(weights, strict=True)

    return model


def quantize_to_match(model: models.Lm, weights: str) -> None:
    if weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)


def load_audio_tokenizer(repo: str, config: dict, lm_config) -> rustymimi.Tokenizer:
    mimi = hf_hub_download(repo, config["mimi_name"])
    codebooks = max(lm_config.generated_codebooks, lm_config.other_codebooks)

    return rustymimi.Tokenizer(mimi, num_codebooks=codebooks)


def condition_tensor(model: models.Lm):
    if model.condition_provider is None:
        return None

    return model.condition_provider.condition_tensor("description", "very_good")


def resample_to_model_rate(pcm16: np.ndarray) -> np.ndarray:
    n_out = int(len(pcm16) * MODEL_RATE / IN_RATE)
    positions = np.arange(n_out) * IN_RATE / MODEL_RATE

    return np.interp(positions, np.arange(len(pcm16)), pcm16).astype(np.float32)


def silence(seconds: float, rate: int) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


class Engine:
    def __init__(self, repo: str) -> None:
        config = json.load(open(hf_hub_download(repo, "config.json")))
        stt = config.get("stt_config") or {}
        self.delay = float(stt.get("audio_delay_seconds", 0.0))
        self.prefix = float(stt.get("audio_silence_prefix_seconds", 0.0))
        self.lm_config = models.LmConfig.from_config_dict(config)
        self.other_codebooks = self.lm_config.other_codebooks
        self.model = load_language_model(repo, config, self.lm_config)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(hf_hub_download(repo, config["tokenizer_name"]))
        self.audio_tokenizer = load_audio_tokenizer(repo, config, self.lm_config)
        self.ct = condition_tensor(self.model)
        self.gen = None
        self.pending = silence(0, MODEL_RATE)
        self.text: list[str] = []

        self.model.warmup(self.ct)

    def start(self) -> None:
        mx.random.seed(299792458)
        self.audio_tokenizer.reset()
        self.gen = models.LmGen(
            model=self.model,
            max_steps=int(MAX_UTTERANCE_SECONDS * MODEL_RATE / STEP),
            text_sampler=utils.Sampler(top_k=25, temp=0.0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.0),
            cfg_coef=1.0,
            check=False,
        )
        self.pending = silence(self.prefix, MODEL_RATE)
        self.text = []

    def feed(self, pcm16: np.ndarray) -> None:
        self.pending = np.concatenate([self.pending, resample_to_model_rate(pcm16)])

        while len(self.pending) >= STEP:
            self._step(self.pending[:STEP])
            self.pending = self.pending[STEP:]

    def end(self) -> str:
        self.feed(silence(self.delay + SILENCE_AFTER_SECONDS, IN_RATE))
        text = "".join(self.text).replace("▁", " ").strip()
        self.gen = None

        return text

    def _step(self, samples: np.ndarray) -> None:
        tokens = self.audio_tokenizer.encode_step(samples[None, None, :])
        tokens = mx.array(tokens).transpose(0, 2, 1)[:, :, : self.other_codebooks]
        token = self.gen.step(tokens[0], self.ct)[0].item()

        if token not in SILENT_TEXT_TOKENS:
            self.text.append(self.text_tokenizer.id_to_piece(token))


def read_message() -> tuple[bytes, np.ndarray] | None:
    header = read_exactly(HEADER_BYTES)

    if header is None:
        return None

    command, length = header[:1], struct.unpack("<I", header[1:])[0]
    payload = read_exactly(length) if length else b""

    if payload is None:
        return None

    return command, np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767


def handle(engine: Engine, command: bytes, pcm: np.ndarray) -> None:
    if command == b"T":
        engine.start()
        engine.feed(pcm)
        emit({"text": engine.end()})
    elif command == b"S":
        engine.start()
    elif command == b"F":
        engine.feed(pcm)
    elif command == b"E":
        emit({"text": engine.end()})
    else:
        emit({"error": f"unknown command {command!r}"})


def serve(engine: Engine) -> None:
    while True:
        message = read_message()

        if message is None:
            return

        command, pcm = message

        try:
            handle(engine, command, pcm)
        except Exception as exc:
            emit({"error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default=DEFAULT_REPO)
    args = parser.parse_args()

    try:
        engine = Engine(args.hf_repo)
    except Exception as exc:
        emit({"error": f"could not load {args.hf_repo}: {exc}"})
        return

    emit({"ready": True})
    serve(engine)


if __name__ == "__main__":
    main()
