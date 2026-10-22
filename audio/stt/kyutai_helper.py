"""Kyutai STT as a helper process for audio/stt/kyutai.py. Runs in its own environment with moshi-mlx.

Started by kyutai.py as ``uv run --no-project --with moshi-mlx python kyutai_helper.py``,
because moshi-mlx pins dependencies this project cannot share. Speaks the pipe
protocol in helper.py: a one-byte command, a four-byte little-endian length and
that many bytes of 16-bit mono PCM at 16 kHz. ``T`` transcribes a whole
utterance; ``S``, ``F`` and ``E`` start an utterance, feed it a frame and end
it, so the model works while the user is still talking and the text is ready
about half a second after they stop. Replies are one JSON line each for ``T``
and ``E``.

Based on moshi_mlx.run_inference. The model listens at 24 kHz in 80 ms steps;
the configured audio delay plus a second of silence is appended at the end so
the last words come out.
"""

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

MODEL_RATE = 24000
STEP = 1920  # samples per model step at 24 kHz (80 ms)
IN_RATE = 16000


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


class Engine:
    def __init__(self, repo: str) -> None:
        config = json.load(open(hf_hub_download(repo, "config.json")))
        stt = config.get("stt_config") or {}
        self.delay = float(stt.get("audio_delay_seconds", 0.0))
        self.prefix = float(stt.get("audio_silence_prefix_seconds", 0.0))
        mimi = hf_hub_download(repo, config["mimi_name"])
        weights = hf_hub_download(repo, config.get("moshi_name", "model.safetensors"))
        tokenizer = hf_hub_download(repo, config["tokenizer_name"])
        self.lm_config = models.LmConfig.from_config_dict(config)
        self.model = models.Lm(self.lm_config)
        self.model.set_dtype(mx.bfloat16)
        if weights.endswith(".q4.safetensors"):
            nn.quantize(self.model, bits=4, group_size=32)
        elif weights.endswith(".q8.safetensors"):
            nn.quantize(self.model, bits=8, group_size=64)
        self.model.load_weights(weights, strict=True)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)
        self.other_codebooks = self.lm_config.other_codebooks
        codebooks = max(self.lm_config.generated_codebooks, self.other_codebooks)
        self.audio_tokenizer = rustymimi.Tokenizer(mimi, num_codebooks=codebooks)
        self.ct = None
        if self.model.condition_provider is not None:
            self.ct = self.model.condition_provider.condition_tensor("description", "very_good")
        self.model.warmup(self.ct)
        self.gen = None
        self.pending = np.zeros(0, dtype=np.float32)  # 24 kHz samples not yet a whole step
        self.text: list[str] = []

    # -- one utterance -----------------------------------------------------------
    def start(self) -> None:
        mx.random.seed(299792458)
        self.audio_tokenizer.reset()
        self.gen = models.LmGen(
            model=self.model,
            max_steps=int(60 * MODEL_RATE / STEP),  # up to a minute of speech
            text_sampler=utils.Sampler(top_k=25, temp=0.0),
            audio_sampler=utils.Sampler(top_k=250, temp=0.0),
            cfg_coef=1.0,
            check=False,
        )
        self.pending = np.zeros(int(self.prefix * MODEL_RATE), dtype=np.float32)
        self.text = []

    def feed(self, pcm16: np.ndarray) -> None:
        # 16 kHz -> 24 kHz by linear interpolation: ample for speech recognition.
        n_out = int(len(pcm16) * MODEL_RATE / IN_RATE)
        resampled = np.interp(np.arange(n_out) * IN_RATE / MODEL_RATE, np.arange(len(pcm16)), pcm16)
        self.pending = np.concatenate([self.pending, resampled.astype(np.float32)])
        while len(self.pending) >= STEP:
            self._step(self.pending[:STEP])
            self.pending = self.pending[STEP:]

    def end(self) -> str:
        self.feed(np.zeros(int((self.delay + 1.0) * IN_RATE), dtype=np.float32))
        text = "".join(self.text).replace("▁", " ").strip()
        self.gen = None
        return text

    def _step(self, samples: np.ndarray) -> None:
        tokens = self.audio_tokenizer.encode_step(samples[None, None, :])
        tokens = mx.array(tokens).transpose(0, 2, 1)[:, :, : self.other_codebooks]
        token = self.gen.step(tokens[0], self.ct)[0].item()
        if token not in (0, 3):
            self.text.append(self.text_tokenizer.id_to_piece(token))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="kyutai/stt-1b-en_fr-mlx")
    args = parser.parse_args()
    try:
        engine = Engine(args.hf_repo)
    except Exception as exc:  # noqa: BLE001 - report it to the parent, then exit
        emit({"error": f"could not load {args.hf_repo}: {exc}"})
        return
    emit({"ready": True})
    while True:
        header = read_exactly(5)
        if header is None:
            return
        command, length = header[:1], struct.unpack("<I", header[1:])[0]
        payload = read_exactly(length) if length else b""
        if payload is None:
            return
        pcm = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767
        try:
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
        except Exception as exc:  # noqa: BLE001
            emit({"error": str(exc)})


if __name__ == "__main__":
    main()
