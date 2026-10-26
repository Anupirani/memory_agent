import json

from src.agent import MemoryAgent
from src.embeddings import HashEmbedder


class FakeChat:
    def __init__(self):
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if "Summarize this conversation" in messages[0]["content"]:
            return "User said they are vegetarian."
        if "compress chat history" in messages[0]["content"]:
            return "Compressed summary."
        return "Noted."


def fake_extract(messages):
    text = messages[-1]["content"]
    if "vegetarian" in text:
        return json.dumps({"facts": [{"text": "User is vegetarian and avoids eggs",
                                      "kind": "preference", "importance": 4}]})
    return json.dumps({"facts": []})


def agent(tmp_path, chat=None, **kw):
    return MemoryAgent("u1", chat or FakeChat(), fake_extract, HashEmbedder(),
                       root=str(tmp_path), **kw)


def test_fact_from_session_one_is_recalled_in_session_two(tmp_path):
    a1 = agent(tmp_path)
    res = a1.chat("I am vegetarian and I avoid eggs")
    assert res.saved and res.saved[0].startswith("added")
    a1.end_session()

    chat2 = FakeChat()
    a2 = agent(tmp_path, chat2)                 # new object = new session
    res2 = a2.chat("what vegetarian meal should I cook tonight")
    system_prompt = chat2.calls[0][0]["content"]
    assert "User is vegetarian and avoids eggs" in system_prompt
    assert res2.recalled


def test_memory_can_be_switched_off(tmp_path):
    agent(tmp_path).chat("I am vegetarian and I avoid eggs")
    chat = FakeChat()
    res = agent(tmp_path, chat, use_memory=False).chat("what vegetarian meal should I cook")
    assert "vegetarian and avoids eggs" not in chat.calls[0][0]["content"]
    assert res.recalled == [] and res.saved == []


def test_end_session_stores_a_summary_memory(tmp_path):
    a = agent(tmp_path)
    a.chat("I am vegetarian and I avoid eggs")
    note = a.end_session()
    kinds = [m.kind for m in a.ltm.all()]
    assert note and "summary" in kinds
    assert a.buffer.messages == []


def test_long_chat_gets_compressed(tmp_path):
    a = agent(tmp_path, max_buffer_tokens=150)
    flags = [a.chat("tell me about topic number " + str(i) + " " + "word " * 40).compressed
             for i in range(6)]
    assert any(flags)
    assert a.buffer.summary == "Compressed summary."
    assert a.buffer.tokens() < 400


def test_two_parallel_sessions_share_memories(tmp_path):
    a, b = agent(tmp_path), agent(tmp_path)
    a.chat("I am vegetarian and I avoid eggs")
    chat = FakeChat()
    b.llm = chat
    b.chat("what vegetarian meal should I cook tonight")
    assert "avoids eggs" in chat.calls[0][0]["content"]