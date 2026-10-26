import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

# Relevance scoring: final = W_SIM*similarity + W_REC*recency + W_IMP*importance
W_SIM, W_REC, W_IMP = 0.70, 0.15, 0.15
HALF_LIFE_DAYS = 30.0   # recency halves every 30 days
MIN_SIM = float(os.getenv("MIN_SIM", "0.20"))  # ignore memories below this similarity
DEDUPE_SIM = 0.85       # above this, a new fact updates the old memory instead


@dataclass
class Memory:
    id: str
    text: str
    kind: str            # preference | fact | goal | event | summary
    importance: int      # 1 (trivial) .. 5 (critical)
    created_at: float
    updated_at: float
    last_accessed: float
    access_count: int = 0
    deleted: bool = False  # tombstone, so deletions survive cross-session sync


@dataclass
class Recalled:
    memory: Memory
    score: float
    similarity: float
    recency: float
    importance: float


_FIELDS = [f.name for f in fields(Memory)]


class LongTermMemory:
    """Per-user vector memory persisted as JSON, with merge-based cross-session sync."""

    def __init__(self, user_id: str, embedder, root: str = "data/users"):
        self.embedder = embedder
        self.path = Path(root) / user_id / "memories.json"
        self.items: dict[str, Memory] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.sync()

    # ---------- persistence and sync ----------
    def _read_disk(self):
        if not self.path.exists():
            return {}, {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        items = {d["id"]: Memory(**{k: d[k] for k in _FIELDS}) for d in data["memories"]}
        vecs = {d["id"]: np.array(d["vector"], dtype=np.float32) for d in data["memories"]}
        if items and data.get("embedder") != self.embedder.name:  # embedder changed: re-embed
            ordered = list(items.values())
            matrix = self.embedder.embed([m.text for m in ordered])
            vecs = {m.id: matrix[n] for n, m in enumerate(ordered)}
        return items, vecs

    def sync(self) -> None:
        """Pull changes made by other sessions: the newer updated_at wins per memory."""
        disk_items, disk_vecs = self._read_disk()
        for mid, theirs in disk_items.items():
            mine = self.items.get(mid)
            if mine is None:
                self.items[mid], self.vectors[mid] = theirs, disk_vecs[mid]
            elif theirs.updated_at > mine.updated_at:
                theirs.last_accessed = max(theirs.last_accessed, mine.last_accessed)
                theirs.access_count = max(theirs.access_count, mine.access_count)
                self.items[mid], self.vectors[mid] = theirs, disk_vecs[mid]
            else:
                mine.last_accessed = max(mine.last_accessed, theirs.last_accessed)
                mine.access_count = max(mine.access_count, theirs.access_count)

    def save(self) -> None:
        self.sync()  # merge first so we never overwrite another session's work
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder.name,
            "memories": [dict(asdict(m), vector=[round(float(x), 5) for x in self.vectors[m.id]])
                         for m in self.items.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: readers never see a half-written file

    # ---------- reading ----------
    def all(self) -> list[Memory]:
        return sorted((m for m in self.items.values() if not m.deleted),
                      key=lambda m: m.created_at)

    def _live_matrix(self):
        ids = [i for i, m in self.items.items() if not m.deleted]
        if not ids:
            return [], None
        return ids, np.vstack([self.vectors[i] for i in ids])

    def recall(self, query: str, k: int = 4, now: Optional[float] = None,
               min_sim: float = MIN_SIM) -> list[Recalled]:
        now = time.time() if now is None else now
        ids, matrix = self._live_matrix()
        if matrix is None:
            return []
        sims = matrix @ self.embedder.embed([query])[0]
        results = []
        for mid, sim in zip(ids, sims):
            if sim < min_sim:
                continue
            m = self.items[mid]
            age_days = max(0.0, (now - m.updated_at) / 86400)
            recency = 0.5 ** (age_days / HALF_LIFE_DAYS)
            importance = (m.importance - 1) / 4
            score = W_SIM * float(sim) + W_REC * recency + W_IMP * importance
            results.append(Recalled(m, score, float(sim), recency, importance))
        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]
        for r in top:
            r.memory.last_accessed = now
            r.memory.access_count += 1
        if top:
            self.save()
        return top

    # ---------- writing ----------
    def add(self, text: str, kind: str = "fact", importance: int = 3,
            now: Optional[float] = None):
        """Returns (memory, 'added' | 'updated'). Near-duplicates update the old memory."""
        now = time.time() if now is None else now
        self.sync()
        vec = self.embedder.embed([text])[0]
        ids, matrix = self._live_matrix()
        if matrix is not None:
            sims = matrix @ vec
            best = int(np.argmax(sims))
            if sims[best] >= DEDUPE_SIM:
                m = self.items[ids[best]]
                m.text, m.kind = text, kind              # newer wording wins
                m.importance = max(m.importance, importance)
                m.updated_at = now
                self.vectors[m.id] = vec
                self.save()
                return m, "updated"
        m = Memory(id=uuid.uuid4().hex[:8], text=text, kind=kind, importance=importance,
                   created_at=now, updated_at=now, last_accessed=now)
        self.items[m.id], self.vectors[m.id] = m, vec
        self.save()
        return m, "added"

    def forget(self, memory_id: str, now: Optional[float] = None) -> bool:
        self.sync()
        m = self.items.get(memory_id)
        if m is None or m.deleted:
            return False
        m.deleted = True
        m.updated_at = time.time() if now is None else now
        self.save()
        return True