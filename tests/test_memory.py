import json
import time

import numpy as np

from src.embeddings import HashEmbedder
from src.memory import LongTermMemory

DAY = 86400


def make(tmp_path, user="u1", embedder=None):
    return LongTermMemory(user, embedder or HashEmbedder(), root=str(tmp_path))


def test_embeddings_are_normalized_and_related_text_is_closer():
    e = HashEmbedder()
    a, b, c = e.embed(["User is allergic to peanuts", "peanut allergy",
                       "User plays chess on Sundays"])
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert a @ b > a @ c


def test_add_and_recall_most_relevant_first(tmp_path):
    m = make(tmp_path)
    m.add("User is allergic to peanuts", "fact", 5)
    m.add("User plays chess on Sundays", "preference", 2)
    top = m.recall("peanut allergy", k=2)
    assert top[0].memory.text == "User is allergic to peanuts"


def test_near_duplicate_updates_instead_of_adding(tmp_path):
    m = make(tmp_path)
    m.add("User is vegetarian", "preference", 3)
    _, action = m.add("User is vegetarian", "preference", 4)
    assert action == "updated"
    assert len(m.all()) == 1 and m.all()[0].importance == 4


def test_recency_decay_prefers_newer_when_equally_similar(tmp_path):
    m = make(tmp_path)
    now = time.time()
    old, _ = m.add("User lives in Pune", "fact", 3, now=now - 90 * DAY)
    new, _ = m.add("User lives in Mumbai now", "fact", 3, now=now)
    top = m.recall("where does the user live", k=2, now=now, min_sim=0.0)
    scores = {r.memory.id: r for r in top}
    assert scores[new.id].recency > scores[old.id].recency


def test_importance_breaks_ties(tmp_path):
    m = make(tmp_path)
    low, _ = m.add("User likes tea number one", "preference", 1)
    high, _ = m.add("User likes tea number two", "preference", 5)
    top = m.recall("tea", k=2, min_sim=0.0)
    assert top[0].memory.id == high.id


def test_min_similarity_filters_unrelated(tmp_path):
    m = make(tmp_path)
    m.add("User is allergic to peanuts", "fact", 5)
    assert m.recall("quantum chromodynamics lecture notes", k=3) == []


def test_persists_across_sessions(tmp_path):
    make(tmp_path).add("User has a daughter named Meera", "fact", 4)
    again = make(tmp_path)                      # brand new object = new session
    assert [x.text for x in again.all()] == ["User has a daughter named Meera"]


def test_two_live_sessions_sync(tmp_path):
    a, b = make(tmp_path), make(tmp_path)
    a.add("User drinks black coffee", "preference", 3)
    b.sync()
    assert [x.text for x in b.all()] == ["User drinks black coffee"]
    b.add("User works in Pune", "fact", 3)
    a.sync()
    assert len(a.all()) == 2


def test_forget_is_a_tombstone_that_syncs(tmp_path):
    a, b = make(tmp_path), make(tmp_path)
    mem, _ = a.add("User dislikes cilantro", "preference", 2)
    b.sync()
    assert b.forget(mem.id)
    a.sync()
    assert a.all() == []
    a.save()                                    # must not resurrect the deleted memory
    assert make(tmp_path).all() == []


def test_changing_embedder_reembeds(tmp_path):
    class Other(HashEmbedder):
        name = "other"
    make(tmp_path).add("User likes sci-fi novels", "preference", 3)
    switched = make(tmp_path, embedder=Other(dim=256))
    assert switched.vectors[switched.all()[0].id].shape[0] == 256
    switched.save()
    saved = json.loads((tmp_path / "u1" / "memories.json").read_text())
    assert saved["embedder"] == "other"