"""Long-term memory for the chatbot: facts it was told, and past conversations.

Two kinds of memory live in one SQLite file (``memory.db`` by default):

* **facts**: short notes such as "the user's name is Adam". The model adds
  them through the ``remember`` tool; every fact is put in the system prompt
  on every turn, which is the most reliable way to get a small model to use
  them. Keep the list short and curated (``/memory`` lists, ``/forget`` prunes).
* **exchanges**: every user message and the reply it got, from every session.
  Before each turn the current message is embedded and the closest past
  exchanges are added to the prompt as background. Embeddings come from
  model2vec (``minishlab/potion-base-8M``, a 30 MB static model, numpy only,
  microseconds per query) and the search is a cosine similarity over a numpy
  array, which is instant at the scale of a personal chat history.

Text only: image turns are stored by their text, the images are not kept.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass

import numpy as np

DEFAULT_DB = "memory.db"
DEFAULT_EMBEDDER = "minishlab/potion-base-8M"
MIN_SIMILARITY = 0.35  # cosine; unrelated text scores near 0, paraphrases ~0.7


@dataclass
class Exchange:
    session: str
    ts: float
    user: str
    assistant: str
    score: float = 0.0

    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))


class Memory:
    def __init__(self, path: str = DEFAULT_DB, embedder: str = DEFAULT_EMBEDDER) -> None:
        from model2vec import StaticModel  # keep the import local so --no-memory stays fast

        self.embedder = StaticModel.from_pretrained(embedder)
        # The chatbot may run its chat loop on a worker thread (--tui) while this was
        # opened on the main one; the connection is only ever used from one thread at a time.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY, ts REAL, text TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY, session TEXT, ts REAL,
                user TEXT, assistant TEXT, embedding BLOB);
            """)
        self.session = self.new_session()
        self._load_index()

    # -- sessions -----------------------------------------------------------
    def new_session(self) -> str:
        """Start a new session; its exchanges are excluded from recall (they are in the prompt)."""
        self.session = uuid.uuid4().hex[:8]
        return self.session

    # -- facts ----------------------------------------------------------------
    def remember(self, fact: str) -> dict:
        fact = " ".join(str(fact).split())
        if not fact:
            return {"ok": False, "error": "empty fact"}
        self.db.execute("INSERT OR IGNORE INTO facts (ts, text) VALUES (?, ?)", (time.time(), fact))
        self.db.commit()
        return {"ok": True, "remembered": fact, "facts_total": len(self.facts())}

    def forget(self, text: str) -> int:
        """Delete facts containing ``text`` (case-insensitive). Returns how many went."""
        cur = self.db.execute("DELETE FROM facts WHERE lower(text) LIKE ?", (f"%{text.lower()}%",))
        self.db.commit()
        return cur.rowcount

    def facts(self) -> list[str]:
        return [row[0] for row in self.db.execute("SELECT text FROM facts ORDER BY id")]

    # -- exchanges ------------------------------------------------------------
    def add_exchange(self, user: str, assistant: str) -> None:
        vec = self._embed(self._exchange_text(user, assistant))
        self.db.execute(
            "INSERT INTO exchanges (session, ts, user, assistant, embedding) VALUES (?, ?, ?, ?, ?)",
            (self.session, time.time(), user, assistant, vec.astype(np.float32).tobytes()),
        )
        self.db.commit()
        self._rows.append(Exchange(self.session, time.time(), user, assistant))
        self._vecs = np.vstack([self._vecs, vec[None]]) if len(self._vecs) else vec[None]

    def recall(self, query: str, k: int = 3) -> list[Exchange]:
        """Closest past exchanges from earlier sessions, best first."""
        if not len(self._vecs):
            return []
        scores = self._vecs @ self._embed(query)
        hits = []
        for i in np.argsort(-scores):
            row = self._rows[i]
            if scores[i] < MIN_SIMILARITY or row.session == self.session:
                continue
            hits.append(Exchange(row.session, row.ts, row.user, row.assistant, float(scores[i])))
            if len(hits) == k:
                break
        return hits

    def count(self) -> int:
        return len(self._rows)

    # -- prompt text ----------------------------------------------------------
    def facts_section(self) -> str:
        """The facts, formatted for the top of the system prompt ("" if none).

        This part of the prompt changes only when a fact is added or removed,
        so the model's cache of the system prompt stays valid between turns.
        """
        facts = self.facts()
        if not facts:
            return ""
        return "Facts about the user that you know for certain and should use when relevant:\n" + "\n".join(
            f"- {f}" for f in facts
        )

    def recall_section(self, query: str) -> str:
        """Earlier exchanges relevant to ``query``, formatted as a note for the user turn ("" if none).

        This changes with every message, so it goes at the end of the prompt,
        not the start, to leave the cached prefix untouched.
        """
        hits = self.recall(query)
        if not hits:
            return ""
        return "Earlier conversations that may be relevant:\n" + "\n".join(
            f"[{h.when()}] user: {h.user}\n  you: {h.assistant}" for h in hits
        )

    def prompt_section(self, query: str) -> str:
        """Facts plus relevant history in one block (the pre-cache layout; kept for callers that want it)."""
        return "\n\n".join(p for p in (self.facts_section(), self.recall_section(query)) if p)

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _exchange_text(user: str, assistant: str) -> str:
        return f"{user}\n{assistant}"

    def _embed(self, text: str) -> np.ndarray:
        vec = self.embedder.encode([text], normalize=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def _load_index(self) -> None:
        rows = self.db.execute("SELECT session, ts, user, assistant, embedding FROM exchanges ORDER BY id").fetchall()
        self._rows = [Exchange(s, ts, u, a) for s, ts, u, a, _ in rows]
        self._vecs = (
            np.vstack([np.frombuffer(e, dtype=np.float32) for *_, e in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
