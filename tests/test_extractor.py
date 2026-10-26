import json

from src.extractor import extract_facts

GOOD = json.dumps({"facts": [{"text": "User is vegetarian", "kind": "preference",
                              "importance": 4}]})


def scripted(responses):
    it = iter(responses)
    return lambda messages: next(it)


def test_extracts_valid_facts():
    facts = extract_facts("I am vegetarian and I avoid eggs", scripted([GOOD]))
    assert [f.text for f in facts] == ["User is vegetarian"]


def test_retries_after_invalid_output():
    bad = json.dumps({"facts": [{"text": "User likes tea", "kind": "hobby", "importance": 9}]})
    facts = extract_facts("I really like tea in the morning", scripted([bad, GOOD]))
    assert len(facts) == 1


def test_gives_empty_list_after_repeated_garbage():
    assert extract_facts("I really like tea in the morning", scripted(["no"] * 3)) == []


def test_skips_tiny_messages_without_calling_the_llm():
    def must_not_run(messages):
        raise AssertionError("LLM should not be called")
    assert extract_facts("ok thanks", must_not_run) == []