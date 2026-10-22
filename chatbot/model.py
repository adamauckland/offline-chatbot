"""Loading Phi-3.5-vision and generating one reply from it, reusing the model's cache across turns.

Every turn re-sends the whole conversation to the model. Without help that
means re-encoding all of it each time, about 1.3 s for a short chat and more
as it grows, with a picture costing hundreds of tokens every turn. A
``Session`` keeps the model's key/value cache alive between calls and
remembers which token ids are in it: the next prompt is tokenised, the
longest prefix it shares with the cache is kept, and only the rest is fed
to the model. The cache is rewound rather than cleared, so an interrupted
reply or a dropped picture just means encoding from the point of difference.

There are three sessions, since each prompt family has a constant prefix:
the chat itself, the "which tool?" question and the fact extraction. The
``Engine`` holds them with the model and processor.

The forward pass here mirrors the library's ``Phi3F.__call__`` (phi.py) with
our own cache, mask and rotary tables, because the library ties the mask and
tables to one cache created per call. Decoding is greedy, so a cached turn
must produce exactly the tokens an uncached one would; ``chat_once`` with
``session=None`` is that uncached path, kept as the fallback for prompts
that do not fit the cache.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import mlx.core as mx
import numpy as np

# Silence transformers' "PyTorch was not found" notice; MLX does the inference.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import phi_3_vision_mlx as pv  # noqa: E402
from phi import SuRoPE  # noqa: E402 - the library's rotary tables, built once per session

import events  # noqa: E402
from chatbot.conversation import END, Conversation, clean_reply  # noqa: E402
from tools import parse_tool_call  # noqa: E402

REPO_ID = "microsoft/Phi-3.5-vision-instruct"
# Tiles per picture. The library resizes every image to this grid whatever its size, so this,
# not the input resolution, sets the cost: 16 tiles are ~2350 prompt tokens per picture, 4 are
# ~750, 1 is ~310, and the whole conversation (pictures included) is re-encoded every turn.
DEFAULT_IMAGE_CROPS = 4
# Tokens the chat session's cache can hold (1.6 GB at 4096). Not more than 4096: beyond the
# model's original context the library switches rotary scaling, which would change every reply.
CACHE_TOKENS = 4096
SIDE_CACHE_TOKENS = 1024  # the router and extractor prompts are short


class LayerCache:
    """One layer's key/value store: the library's ``KVCache`` interface, but persistent and rewindable."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.offset = 0
        self.kv: mx.array | None = None

    def __call__(self, keys: mx.array, values: mx.array, n_beam: int) -> tuple[mx.array, mx.array]:  # noqa: ARG002
        if self.kv is None:
            b, heads, _, dim = keys.shape
            self.kv = mx.zeros((2, b, heads, self.capacity, dim), dtype=keys.dtype)
        end = self.offset + keys.shape[2]
        self.kv[0, :, :, self.offset : end, :] = keys
        self.kv[1, :, :, self.offset : end, :] = values
        self.offset = end
        return self.kv[0, :, :, :end, :], self.kv[1, :, :, :end, :]


class Session:
    """The model's cache for one prompt family, with the token ids it holds."""

    def __init__(self, model, capacity: int) -> None:
        self.model = model
        self.capacity = capacity
        self.layers = model.model.layers
        self.caches = [LayerCache(capacity) for _ in self.layers]
        self.ids: list[int] = []  # what the cache holds; negative ids stand for image tokens, as in the library
        # The picture objects encoded so far, by picture number. Two different pictures of the
        # same shape produce the same token run, so the ids alone cannot tell them apart.
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
        """Feed ``ids`` after what is cached; returns the logits for the token that follows them."""
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
        """Rewind to the prefix shared with ``inputs``, feed the rest, and return the next-token logits.

        ``pictures`` are the conversation's picture objects, in number order.
        None when the prompt plus ``reserve`` tokens of reply would not fit.
        The last prompt token is always fed again, so there are logits even
        when the whole prompt is already cached (after an interrupted reply).
        """
        ids = [int(i) for i in np.asarray(inputs["input_ids"][0]).tolist()]
        if len(ids) + reserve > self.capacity:
            return None
        common = 0
        for a, b in zip(self.ids, ids):
            if a != b or (b < 0 and self.pictures.get(-b) is not pictures[-b - 1]):
                break  # a different token, or the same picture number but another picture
            common += 1
        common = min(common, len(ids) - 1)
        while common > 0 and ids[common] < 0 and ids[common - 1] == ids[common]:
            common -= 1  # never cut inside a picture's run of tokens
        self.rewind(common)
        suffix = ids[common:]
        for token in suffix:
            if token < 0:
                self.pictures[-token] = pictures[-token - 1]
        logits = self.forward(suffix, *_images_in(suffix, inputs))
        mx.eval(logits)  # MLX is lazy: do the encoding now, not when the first token is sampled
        return logits


def _images_in(suffix: list[int], inputs: dict):
    """The pictures whose tokens lie in ``suffix``, with positions relative to it, for ``forward``.

    The library places pictures into their token runs in order, so the
    selection only has to keep the order of first appearance.
    """
    wanted: list[int] = []  # picture numbers, 1-based
    for token in suffix:
        if token < 0 and -token not in wanted:
            wanted.append(-token)
    if not wanted:
        return None, None, None
    pick = mx.array([n - 1 for n in wanted])
    positions = mx.array([[0, i] for i, token in enumerate(suffix) if token < 0])
    return inputs["pixel_values"][pick], inputs["image_sizes"][pick], positions


class Engine:
    """The loaded model and processor, plus a cache session for each kind of prompt."""

    def __init__(self, preload, cache_tokens: int = CACHE_TOKENS) -> None:
        self.model, self.processor = preload
        cache_tokens = max(0, min(cache_tokens, CACHE_TOKENS))
        side = min(SIDE_CACHE_TOKENS, cache_tokens)
        self.chat = Session(self.model, cache_tokens) if cache_tokens else None
        self.router = Session(self.model, side) if cache_tokens else None
        self.extractor = Session(self.model, side) if cache_tokens else None

    def __iter__(self):  # so ``model, processor = engine`` works like the old (model, processor) pair
        return iter((self.model, self.processor))


def load_model(
    quantize: bool = True, image_crops: int = DEFAULT_IMAGE_CROPS, cache_tokens: int = CACHE_TOKENS
) -> Engine:
    """Load the model and processor; ``cache_tokens=0`` re-encodes every prompt in full."""
    preload = pv.load(repo_id=REPO_ID, quantize_model=quantize)
    preload[1].img_processor.num_crops = max(1, image_crops)
    return Engine(preload, cache_tokens)


TRIM_TO = 0.6  # when a conversation outgrows the cache, drop old exchanges down to this share of it


def trim_to_fit(
    engine, convo: Conversation, session: Session, reserve: int, extra_system: str, prefill: str, note: str
) -> int:
    """Drop the oldest exchanges until the prompt plus ``reserve`` tokens fits the session's cache.

    Dropping invalidates the cache from the system prompt on, so when it has
    to happen it drops down to ``TRIM_TO`` of the capacity in one go: one
    re-encode of the remaining turns every so often, instead of a full one
    on every turn once the cap is reached. Returns how many exchanges went.
    """
    _, processor = engine

    def tokens() -> int:  # the processor's own count, pictures expanded to their tiles
        return processor(convo.prompt(extra_system, prefill, note), convo.images or None)["input_ids"].shape[-1]

    if tokens() + reserve <= session.capacity:
        return 0
    dropped = 0
    while tokens() + reserve > TRIM_TO * session.capacity and convo.drop_oldest_exchange():
        dropped += 1
    if dropped:
        print(
            f"(forgetting the oldest {dropped} exchange(s) to keep the conversation inside the model's cache)",
            flush=True,
        )
    return dropped


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
    """Generate one assistant reply, streaming words to stdout as they arrive.

    This is a trimmed-down version of the library's own generation loop: greedy
    decoding with a KV cache, stopping at <|end|>. Writing it here (instead of
    calling ``pv.generate``) lets us hide the end-of-turn token and keeps the
    terminal output clean.

    With ``use_tools`` the loop also stops as soon as a complete tool-call JSON
    object has been produced (otherwise small models tend to carry on and make
    up the tool's result), and nothing from the first "{" onward is printed.

    ``on_text`` receives the same pieces of text as they are printed (whole
    words, never tool-call JSON), so the reply can be spoken while it is
    still being generated. With a ``session`` only the part of the prompt
    that is not already cached is encoded. ``note`` is text rendered into the
    last user turn for this call only (the recalled memories).
    """
    model, processor = engine
    tokenizer = processor.tokenizer
    eos_id = tokenizer.encode(END)[-1]

    inputs = processor(convo.prompt(extra_system, prefill, note), convo.images or None)
    if session is not None and inputs["input_ids"].shape[-1] + max_tokens > session.capacity:
        if trim_to_fit(engine, convo, session, max_tokens, extra_system, prefill, note):
            inputs = processor(convo.prompt(extra_system, prefill, note), convo.images or None)
    logits = session.prepare(inputs, convo.images, max_tokens) if session is not None else None
    cache = None
    if logits is None:  # no session, or the prompt does not fit it: the library's one-call cache
        session = None
        logits, cache = model(**inputs, max_tokens=max_tokens)
        logits = logits[:, -1, :]

    tokens: list[int] = []
    printed = 0
    for _ in range(max_tokens):
        if events.CANCEL.is_set():
            raise KeyboardInterrupt  # a display asked to stop: same path as Ctrl-C
        token = mx.argmax(logits, axis=-1)
        mx.eval(token)
        token_id = token.item()
        if token_id == eos_id:
            break
        tokens.append(token_id)
        text = prefill + tokenizer.decode(tokens)
        # Print up to the last space so partial multi-token words are not shown yet,
        # and never print from the first "{" (or a ```json fence) on when it might be a tool call.
        cut = text.rfind(" ")
        brace = text.find("{") if use_tools else -1
        if brace != -1:
            fence = text.find("```")
            hold = brace if fence == -1 else min(brace, fence)
            cut = min(cut, len(text[:hold].rstrip()))
        if cut > printed and not quiet:
            print(text[printed:cut], end="", flush=True)
            if on_text is not None:
                on_text(text[printed:cut])
            printed = cut
        if brace != -1 and text.rstrip().endswith("}") and parse_tool_call(text):
            break
        if session is not None:
            logits = session.forward([token_id])
        else:
            logits, cache = model(input_ids=token[:, None], cache=cache)
            logits = logits[:, -1, :]

    text = prefill + tokenizer.decode(tokens)
    if quiet:
        pass
    elif use_tools and (prefill or parse_tool_call(text)):
        # A tool call (or a botched attempt at one after a prefill) is never shown.
        if printed:
            print(flush=True)
    else:
        print(text[printed:], flush=True)
        print()
        if on_text is not None:
            on_text(text[printed:])
    return clean_reply(text)
