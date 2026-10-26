import argparse
from datetime import datetime

from .agent import MemoryAgent
from .llm import groq_chat, groq_json

HELP = "Commands: /memories  /forget <id>  /new (end session)  /help  /exit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="default", help="memories are stored per user name")
    ap.add_argument("--no-memory", action="store_true", help="run without any memory")
    args = ap.parse_args()

    agent = MemoryAgent(args.user, groq_chat, groq_json, use_memory=not args.no_memory)
    print(f"Memory agent for user '{args.user}' ({len(agent.ltm.all())} memories). {HELP}")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            text = "/exit"
        if not text:
            continue
        if text == "/exit":
            agent.end_session()
            print("Session saved. Bye.")
            break
        if text == "/help":
            print(HELP)
        elif text == "/new":
            agent.end_session()
            print("Started a new session (long-term memory kept).")
        elif text == "/memories":
            for m in agent.ltm.all():
                when = datetime.fromtimestamp(m.updated_at).strftime("%Y-%m-%d")
                print(f"  [{m.id}] ({m.kind}, imp {m.importance}, {when}, used {m.access_count}x) {m.text}")
        elif text.startswith("/forget "):
            ok = agent.ltm.forget(text.split(maxsplit=1)[1].strip())
            print("Forgotten." if ok else "No memory with that id.")
        else:
            res = agent.chat(text)
            print(f"bot> {res.reply}")
            info = [f"recalled {len(res.recalled)}", f"saved {len(res.saved)}"]
            if res.compressed:
                info.append("context compressed")
            print(f"     [{' | '.join(info)}]")
            for r in res.recalled:
                print(f"     ~ recalled (sim {r.similarity:.2f}, score {r.score:.2f}): {r.memory.text}")
            for line in res.saved:
                print(f"     + {line}")


if __name__ == "__main__":
    main()