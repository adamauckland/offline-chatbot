from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass

import numpy as np

DEFAULT_DB = "memory.db"
DEFAULT_EMBEDDER = "minishlab/potion-base-8M"
MIN_SIMILARITY = 0.35
SCHEMA = """
    CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, ts REAL, text TEXT UNIQUE);
    CREATE TABLE IF NOT EXISTS exchanges (
        id INTEGER PRIMARY KEY, session TEXT, ts REAL, user TEXT, assistant TEXT, embedding BLOB);
"""


def open_database(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.executescript(SCHEMA)

    return db


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
        from model2vec import StaticModel

        self.embedder = StaticModel.from_pretrained(embedder)
        self.db = open_database(path)
        self.session = self.new_session()
        self._load_index()

    def new_session(self) -> str:
        self.session = uuid.uuid4().hex[:8]

        return self.session

    def remember(self, fact: str) -> dict:
        fact = " ".join(str(fact).split())

        if not fact:
            return {"ok": False, "error": "empty fact"}

        self.db.execute("INSERT OR IGNORE INTO facts (ts, text) VALUES (?, ?)", (time.time(), fact))
        self.db.commit()

        return {"ok": True, "remembered": fact, "facts_total": len(self.facts())}

    def forget(self, text: str) -> int:
        cur = self.db.execute("DELETE FROM facts WHERE lower(text) LIKE ?", (f"%{text.lower()}%",))
        self.db.commit()

        return cur.rowcount

    def facts(self) -> list[str]:
        return [row[0] for row in self.db.execute("SELECT text FROM facts ORDER BY id")]

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

    def facts_section(self) -> str:
        facts = self.facts()

        if not facts:
            return ""

        return "Facts about the user that you know for certain and should use when relevant:\n" + "\n".join(
            f"- {f}" for f in facts
        )

    def recall_section(self, query: str) -> str:
        hits = self.recall(query)

        if not hits:
            return ""

        return "Earlier conversations that may be relevant:\n" + "\n".join(
            f"[{h.when()}] user: {h.user}\n  you: {h.assistant}" for h in hits
        )

    def prompt_section(self, query: str) -> str:
        return "\n\n".join(p for p in (self.facts_section(), self.recall_section(query)) if p)

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
