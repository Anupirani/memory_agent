import argparse
import json
import re
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src.agent import MemoryAgent
from src.embeddings import HashEmbedder
from src.llm import groq_chat, groq_json
from src.memory import LongTermMemory

DATA = json.loads(Path("data/eval_set.json").read_text(encoding="utf-8"))
CASES, DISTRACTORS = DATA["cases"], DATA["distractor_memories"]
NO_FACTS = lambda messages: '{"facts": []}'   # used so questions never write memories

UNSURE = ("don't know", "do not know", "don't have", "do not have", "no record", "not sure",
          "haven't", "have not", "can't", "cannot", "no information")


def hit(text: str, expect: list[str]) -> bool:
    """Whole-word match, and the reply must not be hedging (a baseline that says
    'I don't know if you are vegetarian' should not get credit)."""
    if any(u in text.lower() for u in UNSURE):
        return False
    return any(re.search(r"\b" + re.escape(e) + r"\b", text, re.IGNORECASE) for e in expect)


def throttled(fn, pause: float):
    def run(messages):
        out = fn(messages)
        time.sleep(pause)          # stay under free-tier rate limits
        return out
    return run


# ---------- A. retrieval quality (no LLM needed) ----------
def retrieval_eval(embedder) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        ltm = LongTermMemory("eval", embedder, root=tmp)
        for c in CASES:
            ltm.add(c["memory"], "fact", 3)
        for d in DISTRACTORS:
            ltm.add(d, "fact", 3)
        ranks = []
        for c in CASES:
            top = ltm.recall(c["question"], k=10, min_sim=-1.0)
            texts = [r.memory.text for r in top]
            ranks.append(texts.index(c["memory"]) + 1 if c["memory"] in texts else None)
    n = len(CASES)
    return {"recall@1": sum(1 for r in ranks if r == 1) / n,
            "recall@3": sum(1 for r in ranks if r and r <= 3) / n,
            "mrr": sum(1 / r for r in ranks if r) / n,
            "ranks": ranks}


# ---------- B. end-to-end: tell facts in session 1, ask in session 2 ----------
def e2e_eval(embedder, chat, extract) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        s1 = MemoryAgent("eval", chat, extract, embedder, root=tmp)
        for c in CASES:
            s1.chat(c["fact"])
        for q in DATA["chatter"]:
            s1.chat(q)
        s1.end_session()
        extracted = len([m for m in s1.ltm.all() if m.kind != "summary"])

        out = {"facts_extracted": extracted, "facts_expected": len(CASES)}
        for label, use_memory in (("with_memory", True), ("without_memory", False)):
            correct, detail = 0, []
            for c in CASES:
                s2 = MemoryAgent("eval", chat, NO_FACTS, embedder, root=tmp, use_memory=use_memory)
                reply = s2.chat(c["question"]).reply      # fresh session for every question
                ok = hit(reply, c["expect"])
                correct += ok
                detail.append({"question": c["question"], "ok": ok, "reply": reply[:120]})
            out[label] = {"accuracy": correct / len(CASES), "detail": detail}
        return out


# ---------- C. context compression ----------
def compression_eval(chat) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        a = MemoryAgent("compress", chat, NO_FACTS, HashEmbedder(), root=tmp,
                        use_memory=False, max_buffer_tokens=250)
        a.chat("Just so you know, my train to Mysuru leaves at 6:40am from platform 3.")
        raw, peak, compressions = 0, 0, 0
        for text in DATA["filler"]:
            res = a.chat(text)
            compressions += res.compressed
            peak = max(peak, a.buffer.tokens())
        raw = sum(len(m["content"]) // 4 for m in a.buffer.messages) + len(a.buffer.summary) // 4
        answer = a.chat("What time does my train leave, and from which platform?").reply
        return {"compressions": compressions, "peak_buffer_tokens": peak,
                "final_buffer_tokens": a.buffer.tokens(), "budget": 250,
                "detail_survived": "6:40" in answer and "3" in answer, "answer": answer[:160]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="run only the retrieval test")
    ap.add_argument("--pause", type=float, default=1.5)
    args = ap.parse_args()

    embedders = [HashEmbedder()]
    try:
        from src.embeddings import FastEmbedder
        embedders.append(FastEmbedder())
    except Exception as e:  # fastembed not installed
        print("fastembed not available, comparing hash embedder only:", type(e).__name__)

    results = {"retrieval": {}}
    for emb in embedders:
        results["retrieval"][emb.name] = retrieval_eval(emb)
        r = results["retrieval"][emb.name]
        print(f"[retrieval/{emb.name}] recall@1={r['recall@1']:.0%} recall@3={r['recall@3']:.0%} "
              f"MRR={r['mrr']:.2f}")

    best = embedders[-1]   # fastembed if available, otherwise hash
    if not args.skip_llm:
        chat, extract = throttled(groq_chat, args.pause), throttled(groq_json, args.pause)
        results["e2e"] = e2e_eval(best, chat, extract)
        e = results["e2e"]
        print(f"[e2e/{best.name}] facts extracted {e['facts_extracted']}/{e['facts_expected']}")
        print(f"  accuracy WITH memory   : {e['with_memory']['accuracy']:.0%}")
        print(f"  accuracy WITHOUT memory: {e['without_memory']['accuracy']:.0%}")
        results["compression"] = compression_eval(chat)
        print("[compression]", {k: v for k, v in results["compression"].items() if k != "answer"})

    Path("evals").mkdir(exist_ok=True)
    Path("evals/results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2 if "e2e" in results else 1, figsize=(9, 4))
    axes = axes if hasattr(axes, "__len__") else [axes]
    names = list(results["retrieval"])
    for off, metric in ((-0.2, "recall@1"), (0.2, "recall@3")):
        axes[0].bar([i + off for i in range(len(names))],
                    [results["retrieval"][n][metric] * 100 for n in names], 0.4, label=metric)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% of questions")
    axes[0].set_title("Retrieval: right memory in top-k")
    axes[0].legend()
    if "e2e" in results:
        axes[1].bar(["without memory", "with memory"],
                    [results["e2e"]["without_memory"]["accuracy"] * 100,
                     results["e2e"]["with_memory"]["accuracy"] * 100],
                    color=["#d9534f", "#5cb85c"])
        axes[1].set_ylim(0, 100)
        axes[1].set_title("Answers correct in a NEW session")
    plt.tight_layout()
    plt.savefig("evals/results.png", dpi=150)
    print("saved evals/results.json and evals/results.png")


if __name__ == "__main__":
    main()