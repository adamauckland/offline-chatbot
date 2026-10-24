from __future__ import annotations

import os
from collections.abc import Callable

import mlx.core as mx
import numpy as np


def silence_transformers_notices() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


silence_transformers_notices()

import phi_3_vision_mlx as pv
from phi import SuRoPE

import events
from chatbot.conversation import END, Conversation, clean_reply
from tools import parse_tool_call

REPO_ID = "microsoft/Phi-3.5-vision-instruct"
DEFAULT_IMAGE_CROPS = 4
CACHE_TOKENS = 4096
SIDE_CACHE_TOKENS = 1024
TRIM_TO = 0.6


class LayerCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.offset = 0
        self.kv: mx.array | None = None

    def __call__(self, keys: mx.array, values: mx.array, n_beam: int) -> tuple[mx.array, mx.array]:
        if self.kv is None:
            b, heads, _, dim = keys.shape
            self.kv = mx.zeros((2, b, heads, self.capacity, dim), dtype=keys.dtype)

        end = self.offset + keys.shape[2]
        self.kv[0, :, :, self.offset : end, :] = keys
        self.kv[1, :, :, self.offset : end, :] = values
        self.offset = end

        return self.kv[0, :, :, :end, :], self.kv[1, :, :, :end, :]


class Session:
    def __init__(self, model, capacity: int) -> None:
        self.model = model
        self.capacity = capacity
        self.layers = model.model.layers
        self.caches = [LayerCache(capacity) for _ in self.layers]
        self.ids: list[int] = []
        self.pictures: dict[int, object] = {}
        self.rope = SuRoPE(model.config, capacity, None)

    @property
    def length(self) -> int:
        return self.caches[0].offset

    def rewind(self, n: int) -> None:
        for cache in self.caches:
            cache.offset = n

        del self.ids[n:]
        kept = {-token for token in self.ids if token < 0}
        self.pictures = {number: picture for number, picture in self.pictures.items() if number in kept}

    def reset(self) -> None:
        self.rewind(0)

    def forward(self, ids: list[int], pixel_values=None, image_sizes=None, positions=None) -> mx.array:
        model = self.model.model
        past, new = self.length, len(ids)

        if past + new > self.capacity:
            raise OverflowError("session cache is full")

        x = model.embed_tokens(mx.array(ids)[None])

        if pixel_values is not None:
            x = model.vision_embed_tokens(x, pixel_values, image_sizes, positions)

        mask = mx.triu(mx.full((new, past + new), -mx.inf), k=past + 1)[None, None]
        cos, sin = self.rope(past, new)

        for layer, cache in zip(self.layers, self.caches):
            x = layer(x, cache, cos, sin, mask, 1)

        self.ids.extend(ids)

        return self.model.lm_head(model.norm(x))[:, -1, :]

    def prepare(self, inputs: dict, pictures: list, reserve: int) -> mx.array | None:
        ids = token_ids(inputs)

        if len(ids) + reserve > self.capacity:
            return None

        common = self._shared_prefix_length(ids, pictures)
        self.rewind(common)
        suffix = ids[common:]
        self._note_pictures(suffix, pictures)
        logits = self.forward(suffix, *images_in(suffix, inputs))
        mx.eval(logits)

        return logits

    def _shared_prefix_length(self, ids: list[int], pictures: list) -> int:
        common = 0

        for cached, wanted in zip(self.ids, ids):
            if cached != wanted or self._is_another_picture(wanted, pictures):
                break

            common += 1

        return back_out_of_picture_run(ids, min(common, len(ids) - 1))

    def _is_another_picture(self, token: int, pictures: list) -> bool:
        return token < 0 and self.pictures.get(-token) is not pictures[-token - 1]

    def _note_pictures(self, suffix: list[int], pictures: list) -> None:
        for token in suffix:
            if token < 0:
                self.pictures[-token] = pictures[-token - 1]


def token_ids(inputs: dict) -> list[int]:
    return [int(i) for i in np.asarray(inputs["input_ids"][0]).tolist()]


def back_out_of_picture_run(ids: list[int], common: int) -> int:
    while common > 0 and ids[common] < 0 and ids[common - 1] == ids[common]:
        common -= 1

    return common


def images_in(suffix: list[int], inputs: dict):
    wanted = picture_numbers_in_order(suffix)

    if not wanted:
        return None, None, None

    pick = mx.array([n - 1 for n in wanted])
    positions = mx.array([[0, i] for i, token in enumerate(suffix) if token < 0])

    return inputs["pixel_values"][pick], inputs["image_sizes"][pick], positions


def picture_numbers_in_order(suffix: list[int]) -> list[int]:
    wanted: list[int] = []

    for token in suffix:
        if token < 0 and -token not in wanted:
            wanted.append(-token)

    return wanted


class Engine:
    def __init__(self, preload, cache_tokens: int = CACHE_TOKENS) -> None:
        self.model, self.processor = preload
        cache_tokens = max(0, min(cache_tokens, CACHE_TOKENS))
        side = min(SIDE_CACHE_TOKENS, cache_tokens)
        self.chat = Session(self.model, cache_tokens) if cache_tokens else None
        self.router = Session(self.model, side) if cache_tokens else None
        self.extractor = Session(self.model, side) if cache_tokens else None

    def __iter__(self):
        return iter((self.model, self.processor))


def load_model(
    quantize: bool = True, image_crops: int = DEFAULT_IMAGE_CROPS, cache_tokens: int = CACHE_TOKENS
) -> Engine:
    preload = pv.load(repo_id=REPO_ID, quantize_model=quantize)
    preload[1].img_processor.num_crops = max(1, image_crops)

    return Engine(preload, cache_tokens)


def encode(processor, convo: Conversation, extra_system: str, prefill: str, note: str) -> dict:
    return processor(convo.prompt(extra_system, prefill, note), convo.images or None)


def prompt_length(inputs: dict) -> int:
    return inputs["input_ids"].shape[-1]


def trim_to_fit(
    engine, convo: Conversation, session: Session, reserve: int, extra_system: str, prefill: str, note: str
) -> int:
    _, processor = engine

    def prompt_tokens() -> int:
        return prompt_length(encode(processor, convo, extra_system, prefill, note))

    if prompt_tokens() + reserve <= session.capacity:
        return 0

    dropped = 0

    while prompt_tokens() + reserve > TRIM_TO * session.capacity and convo.drop_oldest_exchange():
        dropped += 1

    if dropped:
        print(
            f"(forgetting the oldest {dropped} exchange(s) to keep the conversation inside the model's cache)",
            flush=True,
        )

    return dropped


class SessionDecoder:
    def __init__(self, session: Session, logits: mx.array) -> None:
        self.session = session
        self.logits = logits

    def step(self, token: mx.array) -> None:
        self.logits = self.session.forward([token.item()])


class LibraryDecoder:
    def __init__(self, model, inputs: dict, max_tokens: int) -> None:
        self.model = model
        logits, self.cache = model(**inputs, max_tokens=max_tokens)
        self.logits = logits[:, -1, :]

    def step(self, token: mx.array) -> None:
        logits, self.cache = self.model(input_ids=token[:, None], cache=self.cache)
        self.logits = logits[:, -1, :]


def decoder_for(
    engine, convo: Conversation, max_tokens: int, extra_system: str, prefill: str, note: str, session: Session | None
) -> SessionDecoder | LibraryDecoder:
    model, processor = engine
    inputs = encode(processor, convo, extra_system, prefill, note)

    if session is not None and prompt_length(inputs) + max_tokens > session.capacity:
        if trim_to_fit(engine, convo, session, max_tokens, extra_system, prefill, note):
            inputs = encode(processor, convo, extra_system, prefill, note)

    logits = session.prepare(inputs, convo.images, max_tokens) if session is not None else None

    if logits is None:
        return LibraryDecoder(model, inputs, max_tokens)

    return SessionDecoder(session, logits)


def tool_call_start(text: str) -> int:
    brace = text.find("{")

    if brace == -1:
        return -1

    fence = text.find("```")

    return brace if fence == -1 else min(brace, fence)


def safe_to_show(text: str, use_tools: bool) -> int:
    cut = text.rfind(" ")
    hold = tool_call_start(text) if use_tools else -1

    if hold != -1:
        cut = min(cut, len(text[:hold].rstrip()))

    return cut


def is_complete_tool_call(text: str, use_tools: bool) -> bool:
    if not use_tools or "{" not in text or not text.rstrip().endswith("}"):
        return False

    return parse_tool_call(text) is not None


class ReplyPrinter:
    def __init__(self, quiet: bool, use_tools: bool, on_text: Callable[[str], None] | None) -> None:
        self.quiet = quiet
        self.use_tools = use_tools
        self.on_text = on_text
        self.printed = 0

    def show(self, text: str) -> None:
        if self.quiet:
            return

        cut = safe_to_show(text, self.use_tools)

        if cut > self.printed:
            self._emit(text[self.printed : cut])
            self.printed = cut

    def finish(self, text: str, prefill: str) -> None:
        if self.quiet:
            return

        if self.use_tools and (prefill or parse_tool_call(text)):
            self._end_line_if_started()
            return

        self._emit(text[self.printed :])
        print("\n", flush=True)

    def _emit(self, piece: str) -> None:
        print(piece, end="", flush=True)

        if self.on_text is not None:
            self.on_text(piece)

    def _end_line_if_started(self) -> None:
        if self.printed:
            print(flush=True)


def chat_once(
    engine,
    convo: Conversation,
    max_tokens: int,
    use_tools: bool = False,
    extra_system: str = "",
    quiet: bool = False,
    prefill: str = "",
    on_text: Callable[[str], None] | None = None,
    session: Session | None = None,
    note: str = "",
) -> str:
    _, processor = engine
    tokenizer = processor.tokenizer
    eos_id = tokenizer.encode(END)[-1]
    decoder = decoder_for(engine, convo, max_tokens, extra_system, prefill, note, session)
    printer = ReplyPrinter(quiet, use_tools, on_text)
    tokens: list[int] = []

    for _ in range(max_tokens):
        if events.CANCEL.is_set():
            raise KeyboardInterrupt

        token = mx.argmax(decoder.logits, axis=-1)
        mx.eval(token)

        if token.item() == eos_id:
            break

        tokens.append(token.item())
        text = prefill + tokenizer.decode(tokens)
        printer.show(text)

        if is_complete_tool_call(text, use_tools):
            break

        decoder.step(token)

    text = prefill + tokenizer.decode(tokens)
    printer.finish(text, prefill)

    return clean_reply(text)
