"""The command line: ask a question, or measure the thing that answers it.

    ai-coach ask "how do I hand in a session" --pages ./units/en
    ai-coach ask "explain structured outputs" --provider ollama
    ai-coach measure --pages ./units/en --cases data/coach.jsonl
    ai-coach providers

Two commands, because there are two things you do with a coach: use it, and
find out whether your change to it helped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_engineering_coach import corpus
from ai_engineering_coach.coach import answer
from ai_engineering_coach.measure import CaseError, load_cases, run
from ai_engineering_coach.models import PROVIDERS, ModelError, get_client


def _pages(argument: str | None) -> Path:
    return Path(argument) if argument else Path.cwd()


def _ask(question: str, pages: Path, provider: str, model: str, top_k: int) -> int:
    documents = corpus.load(pages)
    if not documents:
        print(f"no pages under {pages}", file=sys.stderr)
        return 2
    try:
        client = get_client(provider, model)
    except ModelError as error:
        print(str(error), file=sys.stderr)
        return 2

    result = answer(question, documents, client=client, top_k=top_k)

    if result.refused:
        print(f"\n  {result.reason}.\n")
        # Not an error: refusing is a correct answer, and a non-zero exit here
        # would make every honest refusal look like a broken tool in a script.
        return 0

    if result.prose:
        print(f"\n{result.prose}\n")
        if result.fabricated:
            # Surfaced, never swallowed. A citation the model invented is the
            # single most useful thing this tool can show you about a model.
            named = ", ".join(result.fabricated)
            print(f"  removed {len(result.fabricated)} invented citation(s): {named}\n")

    for scored in result.passages:
        chunk = scored.chunk
        where = chunk.url or chunk.doc_id
        print(f"  {chunk.title}  ({scored.score:.2f})\n  {where}")
        if not result.prose:
            body = chunk.text.strip()
            print("\n" + "\n".join(f"    {line}" for line in body.splitlines()[:12]))
        print()
    if result.reason and result.prose == "":
        print(f"  note: {result.reason}\n")
    return 0


def _measure(pages: Path, cases_path: Path, top_k: int) -> int:
    documents = corpus.load(pages)
    try:
        cases = load_cases(cases_path)
    except CaseError as error:
        print(str(error), file=sys.stderr)
        return 2
    report = run(cases, documents, top_k=top_k)
    print(f"\n  {len(documents)} pages from {pages}\n")
    print(report.rendered())
    print()
    # Never fail on a low score: a number you are trying to improve must not be
    # a gate you are trying to pass.
    return 0


def _providers() -> int:
    print()
    for name, lane in sorted(PROVIDERS.items()):
        key = lane.key_env or "no key needed"
        model = lane.default_model or "pass --model"
        print(f"  {name:<12} {lane.base_url}")
        print(f"  {'':<12} {key} · {model}")
        if lane.note:
            print(f"  {'':<12} {lane.note}")
        print()
    print("  echo         no model at all: quote the pages and say nothing more\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-coach", description="Ask a course a question, and measure what answers it."
    )
    sub = parser.add_subparsers(dest="command")

    asker = sub.add_parser("ask", help="answer a question from the pages")
    asker.add_argument("question")
    asker.add_argument("--pages", help="the corpus directory (default: the working directory)")
    asker.add_argument("--provider", default="", help="ollama, moonshot, openai, groq, openrouter")
    asker.add_argument("--model", default="", help="the model id for that provider")
    asker.add_argument("--top-k", type=int, default=3)

    measurer = sub.add_parser("measure", help="hit rate against a labelled set")
    measurer.add_argument("--pages")
    measurer.add_argument("--cases", required=True, help="a JSONL labelled set")
    measurer.add_argument("--top-k", type=int, default=3)

    sub.add_parser("providers", help="the model lanes this knows about")

    args = parser.parse_args(argv)
    if args.command == "ask":
        return _ask(args.question, _pages(args.pages), args.provider, args.model, args.top_k)
    if args.command == "measure":
        return _measure(_pages(args.pages), Path(args.cases), args.top_k)
    if args.command == "providers":
        return _providers()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
