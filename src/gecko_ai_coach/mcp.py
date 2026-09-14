"""An MCP server, in stdlib, so a harness can ask the course questions.

    gecko-ai-coach-course --pages ./units/en

WHY WRITE THE PROTOCOL RATHER THAN IMPORT IT. MCP over stdio is JSON-RPC 2.0
with three methods that matter, and the whole of it fits on one screen. A
dependency here would cost more than it saves, and this file is meant to be
copied by somebody adding their own capability -- which is far likelier if they
can read all of it in a sitting.

THE SHAPE OF A CAPABILITY, and it is worth naming because it is the thing being
taught: a **Skill** tells the agent when to reach for this and what the tools
mean; an **MCP server** is the tools themselves. The skill is instructions, the
server is capability, and neither substitutes for the other. `capabilities/`
holds one directory per capability with both halves.

WHAT THIS DELIBERATELY DOES NOT DO. It never writes, never fetches a URL, and
never reads outside the pages directory it was given. A course tool that can
edit a repository is a different and much more dangerous thing, and nothing here
needs it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from gecko_ai_coach import corpus
from gecko_ai_coach.coach import answer
from gecko_ai_coach.measure import CaseError, load_cases, run
from gecko_ai_coach.models import get_client
from gecko_ai_coach.retrieve import Document

#: Echoed back to the client when it asks for one we can speak. A server that
#: insists on its own version breaks against a newer harness for no reason.
DEFAULT_PROTOCOL = "2025-06-18"
SUPPORTED = ("2025-06-18", "2025-03-26", "2024-11-05")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ask_course",
        "description": (
            "Answer a question from the course pages, and name the page each answer came "
            "from. Returns NOT IN THESE PAGES when nothing supports an answer, which is a "
            "correct outcome rather than a failure. Use this before guessing about how the "
            "course works, what a session covers, or how to hand work in."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question, in plain words."},
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to retrieve. Default 3.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "list_pages",
        "description": (
            "Every page the coach can quote, as page ids with titles. Use it to find out "
            "what exists before asking, or to check whether a week has been published yet."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "contains": {
                    "type": "string",
                    "description": "Only pages whose id or title contains this text.",
                }
            },
        },
    },
    {
        "name": "measure_retrieval",
        "description": (
            "Score the retriever against a labelled set of questions and print the hit rate, "
            "naming every miss. This is how a change to retrieval is judged: run it before "
            "and after, and report both numbers including any regression."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cases": {
                    "type": "string",
                    "description": "Path to a JSONL labelled set. Defaults to the bundled one.",
                },
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
    },
]


def _text(body: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": body}]}


def _failed(body: str) -> dict[str, Any]:
    # `isError` rather than a JSON-RPC error: a tool that could not answer is a
    # result the model should read and act on, not a transport fault.
    return {"content": [{"type": "text", "text": body}], "isError": True}


def _ask(documents: list[Document], arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question", "")).strip()
    if not question:
        return _failed("ask_course needs a question")
    result = answer(
        question,
        documents,
        client=get_client(),
        top_k=int(arguments.get("top_k", 3)),
    )
    if result.refused:
        return _text(f"NOT IN THESE PAGES — {result.reason}.")
    lines: list[str] = []
    if result.prose:
        lines.append(result.prose)
        if result.fabricated:
            lines.append(f"(removed invented citations: {', '.join(result.fabricated)})")
        lines.append("")
    for scored in result.passages:
        chunk = scored.chunk
        lines.append(f"--- {chunk.doc_id} · {chunk.title} ({scored.score:.2f})")
        lines.append(chunk.text.strip())
        lines.append("")
    return _text("\n".join(lines).strip())


def _list(documents: list[Document], arguments: dict[str, Any]) -> dict[str, Any]:
    needle = str(arguments.get("contains", "")).lower()
    rows = [
        f"{doc.doc_id}  ·  {doc.title}"
        for doc in documents
        if not needle or needle in doc.doc_id.lower() or needle in doc.title.lower()
    ]
    return _text("\n".join(rows) if rows else "no page matches that")


def _measure(documents: list[Document], arguments: dict[str, Any], default: Path) -> dict[str, Any]:
    path = Path(str(arguments.get("cases", ""))) if arguments.get("cases") else default
    try:
        cases = load_cases(path)
    except CaseError as error:
        return _failed(str(error))
    report = run(cases, documents, top_k=int(arguments.get("top_k", 3)))
    return _text(report.rendered())


def handle(
    request: dict[str, Any], documents: list[Document], cases: Path
) -> dict[str, Any] | None:
    """One JSON-RPC request in, one response out — or None for a notification.

    Returning None matters: a notification has no `id` and a reply to one is a
    protocol violation that some clients treat as fatal.
    """
    method = request.get("method", "")
    identifier = request.get("id")

    if method == "initialize":
        asked = (request.get("params") or {}).get("protocolVersion")
        version = asked if asked in SUPPORTED else DEFAULT_PROTOCOL
        result: dict[str, Any] = {
            "protocolVersion": version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "gecko-ai-coach-course", "version": "0.1.0"},
        }
    elif method in ("notifications/initialized", "notifications/cancelled"):
        return None
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name", "")
        arguments = params.get("arguments") or {}
        if name == "ask_course":
            result = _ask(documents, arguments)
        elif name == "list_pages":
            result = _list(documents, arguments)
        elif name == "measure_retrieval":
            result = _measure(documents, arguments, cases)
        else:
            return {
                "jsonrpc": "2.0",
                "id": identifier,
                "error": {"code": -32602, "message": f"no tool called {name!r}"},
            }
    else:
        return {
            "jsonrpc": "2.0",
            "id": identifier,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": identifier, "result": result}


def serve(documents: list[Document], cases: Path, stdin: Any = None, stdout: Any = None) -> int:
    """Read newline-delimited JSON-RPC from stdin, write answers to stdout.

    stdin and stdout are arguments so the tests can drive this over a pair of
    pipes rather than a subprocess -- a server you can only test by launching it
    is a server nobody tests.
    """
    source = stdin or sys.stdin
    sink = stdout or sys.stdout
    for line in source:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            # No id to answer to, so the only honest move is to say nothing and
            # keep serving. Exiting would take the whole session down.
            continue
        response = handle(request, documents, cases)
        if response is None:
            continue
        sink.write(json.dumps(response) + "\n")
        sink.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gecko-ai-coach-course", description="The course coach, as an MCP server."
    )
    parser.add_argument(
        "--pages",
        default=os.environ.get("COACH_PAGES", ""),
        help="the course pages directory (or COACH_PAGES)",
    )
    parser.add_argument("--cases", default="", help="a labelled set for measure_retrieval")
    args = parser.parse_args(argv)

    if not args.pages:
        print("no pages: pass --pages or set COACH_PAGES", file=sys.stderr)
        return 2
    try:
        documents = corpus.load(Path(args.pages))
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2
    return serve(documents, Path(args.cases) if args.cases else Path("data/dev3pack.jsonl"))


if __name__ == "__main__":
    raise SystemExit(main())


__all__: list[str] = ["TOOLS", "handle", "main", "serve"]
