from typing import Callable

LLM = Callable[[list], str]

SUMMARY_PROMPT = (
    "You compress chat history. Merge the existing summary and the older messages into ONE "
    "short summary (max 120 words). Keep names, numbers, dates, decisions, preferences and open "
    "questions. Drop greetings and filler. Plain text only."
)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)  # rough rule of thumb: about 4 characters per token


class ShortTermBuffer:
    """Recent turns kept verbatim; older turns are folded into a running summary."""

    def __init__(self, max_tokens: int = 1200, keep_recent: int = 4):
        self.max_tokens = max_tokens
        self.keep_recent = keep_recent     # number of most recent messages never compressed
        self.messages: list[dict] = []
        self.summary = ""

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def tokens(self) -> int:
        return estimate_tokens(self.summary) * bool(self.summary) + sum(
            estimate_tokens(m["content"]) for m in self.messages)

    def needs_compression(self) -> bool:
        return self.tokens() > self.max_tokens and len(self.messages) > self.keep_recent

    def compress(self, llm: LLM) -> tuple[int, int]:
        """Returns (tokens_before, tokens_after)."""
        before = self.tokens()
        old, recent = self.messages[:-self.keep_recent], self.messages[-self.keep_recent:]
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        try:
            new_summary = llm([
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": f"Existing summary:\n{self.summary or '(none)'}\n\n"
                                            f"Older messages:\n{transcript}"},
            ]).strip()
        except Exception:
            new_summary = ""
        if not new_summary:  # fallback: crude extract so we still shrink the buffer
            new_summary = (self.summary + " " + " | ".join(
                m["content"][:80] for m in old if m["role"] == "user")).strip()
        self.summary, self.messages = new_summary, recent
        return before, self.tokens()