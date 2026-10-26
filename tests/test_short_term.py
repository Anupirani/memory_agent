from src.short_term import ShortTermBuffer, estimate_tokens


def fill(buf, n=10, size=200):
    for i in range(n):
        buf.add("user" if i % 2 == 0 else "assistant", f"message {i} " + "x" * size)


def test_no_compression_when_small():
    buf = ShortTermBuffer(max_tokens=5000)
    fill(buf, 4, 20)
    assert not buf.needs_compression()


def test_compression_shrinks_and_keeps_recent():
    buf = ShortTermBuffer(max_tokens=300, keep_recent=4)
    fill(buf, 10)
    assert buf.needs_compression()
    last_four = list(buf.messages[-4:])
    before, after = buf.compress(lambda msgs: "Short summary: user discussed topics 0-5.")
    assert after < before
    assert buf.messages == last_four
    assert "Short summary" in buf.summary


def test_summary_is_fed_back_on_next_compression():
    seen = []
    buf = ShortTermBuffer(max_tokens=100, keep_recent=2)
    buf.summary = "EARLIER SUMMARY"
    fill(buf, 6)
    buf.compress(lambda msgs: seen.append(msgs[-1]["content"]) or "new summary")
    assert "EARLIER SUMMARY" in seen[0]


def test_falls_back_when_llm_fails():
    def boom(msgs):
        raise RuntimeError("api down")
    buf = ShortTermBuffer(max_tokens=100, keep_recent=2)
    fill(buf, 6)
    buf.compress(boom)
    assert buf.summary and len(buf.messages) == 2


def test_estimate_tokens_is_positive():
    assert estimate_tokens("") == 1 and estimate_tokens("a" * 40) == 10