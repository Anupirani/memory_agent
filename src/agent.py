import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .embeddings import get_embedder
from .extractor import extract_facts
from .llm import LLM
from .memory import LongTermMemory, Recalled
from .short_term import ShortTermBuffer, estimate_tokens

SYSTEM = (
    "You are a friendly assistant with a long-term memory of the user. Use the saved memories "
    "below only when they are relevant, and never invent memories. If memories conflict, trust "
    "the most recent one. If you do not know something about the user, say so."
)


@dataclass
class ChatResult:
    reply: str
    recalled: list[Recalled]
    saved: list[str]
    compressed: bool
    prompt_tokens: int


class MemoryAgent:
    def __init__(self, user_id: str, llm: LLM, extract_llm: Optional[LLM] = None,
                 embedder=None, root: str = "data/users", use_memory: bool = True,
                 top_k: int = 4, max_buffer_tokens: int = 1200):
        self.llm = llm
        self.extract_llm = extract_llm or llm
        self.use_memory = use_memory
        self.top_k = top_k
        self.max_buffer_tokens = max_buffer_tokens
        self.ltm = LongTermMemory(user_id, embedder or get_embedder(), root)
        self.buffer = ShortTermBuffer(max_tokens=max_buffer_tokens)

    def _system_prompt(self, recalled: list[Recalled]) -> str:
        lines = [f"- {r.memory.text} (saved {datetime.fromtimestamp(r.memory.updated_at):%Y-%m-%d})"
                 for r in recalled] or ["- (nothing relevant saved yet)"]
        prompt = SYSTEM + "\n\nRelevant memories:\n" + "\n".join(lines)
        if self.buffer.summary:
            prompt += "\n\nSummary of the earlier conversation:\n" + self.buffer.summary
        return prompt

    def chat(self, user_message: str) -> ChatResult:
        recalled = []
        if self.use_memory:
            self.ltm.sync()                                   # see other sessions' changes
            recalled = self.ltm.recall(user_message, k=self.top_k)
        messages = ([{"role": "system", "content": self._system_prompt(recalled)}]
                    + self.buffer.messages + [{"role": "user", "content": user_message}])
        reply = self.llm(messages)
        self.buffer.add("user", user_message)
        self.buffer.add("assistant", reply)

        saved = []
        if self.use_memory:
            for fact in extract_facts(user_message, self.extract_llm):
                mem, action = self.ltm.add(fact.text, fact.kind, fact.importance)
                saved.append(f"{action}: {mem.text}")

        compressed = False
        if self.buffer.needs_compression():
            self.buffer.compress(self.llm)
            compressed = True

        tokens = sum(estimate_tokens(m["content"]) for m in messages)
        return ChatResult(reply, recalled, saved, compressed, tokens)

    def end_session(self) -> Optional[str]:
        """Fold the finished conversation into one long-term 'summary' memory, then reset."""
        note = None
        if self.use_memory and self.buffer.messages:
            transcript = "\n".join(f"{m['role']}: {m['content']}" for m in self.buffer.messages)
            if self.buffer.summary:
                transcript = f"(earlier summary) {self.buffer.summary}\n" + transcript
            try:
                note = self.llm([
                    {"role": "system", "content": "Summarize this conversation in 1-2 sentences, "
                     "focusing on what was learned about the user and what was decided. Plain text."},
                    {"role": "user", "content": transcript},
                ]).strip()
            except Exception:
                note = None
            if note:
                stamp = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
                self.ltm.add(f"Session summary ({stamp}): {note}", kind="summary", importance=2)
        self.buffer = ShortTermBuffer(max_tokens=self.max_buffer_tokens)
        return note