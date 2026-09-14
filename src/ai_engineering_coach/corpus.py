"""Turn a directory of markdown pages into a corpus, and leave the answers out.

COURSE-AGNOSTIC ON PURPOSE. This knows about markdown and about JSX-ish blocks,
and nothing about any particular course. Point it at a directory and it reads
what is there; a corpus you can only build for one repository is one nobody else
can improve.

TWO THINGS IT REFUSES TO INDEX, and both are the difference between a coach and
a cheat sheet:

  solutions/   -- a worked answer is not a passage to quote back
  <Question/>  -- inline quiz blocks carry `correct: true` in their own markup

A page that is not on disk is not indexed either, which sounds obvious and is
the whole reason a published clone can only answer about what it was given.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from pathlib import Path

from ai_engineering_coach.retrieve import Document

#: Directory names that never enter a corpus, matched on any path segment.
EXCLUDED = frozenset({"solutions", "node_modules", "__pycache__", ".git", ".ipynb_checkpoints"})

_HEADING = re.compile(r"^#\s+(.+?)\s*(?:\[\[[^\]]*\]\])?\s*$", re.M)
_FENCE = re.compile(r"^```", re.M)


def _strip_components(text: str) -> str:
    """Remove self-closing JSX blocks such as an inline quiz.

    Scans through quoted strings so a `/>` inside prose does not end the block
    early, and honours backslash escapes inside those strings. Both are real
    bugs rather than hypothetical ones: course prose is full of slashes.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        start = text.find("<", index)
        if start == -1:
            out.append(text[index:])
            break
        # Only block-level components, i.e. a capitalised tag at a line start.
        following = text[start + 1 : start + 2]
        at_line_start = start == 0 or text[start - 1] == "\n"
        if not (at_line_start and following.isupper()):
            out.append(text[index : start + 1])
            index = start + 1
            continue

        out.append(text[index:start])
        scan = start + 1
        quote = ""
        while scan < length:
            char = text[scan]
            if quote:
                if char == "\\":
                    scan += 2
                    continue
                if char == quote:
                    quote = ""
            elif char in "\"'`":
                quote = char
            elif char == "/" and text.startswith("/>", scan):
                scan += 2
                break
            elif char == ">" and text[scan - 1] == "/":
                scan += 1
                break
            scan += 1
        index = scan
    return "".join(out)


def _title_of(text: str, fallback: str) -> str:
    found = _HEADING.search(text)
    return found.group(1).strip() if found else fallback


def _is_excluded(path: Path, root: Path) -> bool:
    return any(part in EXCLUDED for part in path.relative_to(root).parts)


def load(
    root: Path,
    suffixes: Iterable[str] = (".md", ".mdx"),
    url_for: Callable[[str], str] | None = None,
) -> list[Document]:
    """Every page under `root`, as retrieval Documents, in a stable order.

    `doc_id` is the path under `root` without its suffix, so it is a citation a
    human can act on: `unit1/session-03-structured-outputs/introduction` names a
    file they can open. `url_for` turns that id into a link when the corpus is
    published somewhere; without it a citation is still the path.

    Built fresh every call and never cached to disk. An index that can go stale
    is worse than no index, because it is wrong in a way nobody looks at.
    """
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"no corpus directory at {root}")

    wanted = tuple(suffixes)
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in wanted:
            continue
        if _is_excluded(path, root):
            continue
        raw = path.read_text(encoding="utf-8")
        doc_id = path.relative_to(root).with_suffix("").as_posix()
        body = _strip_components(raw)
        if not body.strip():
            continue
        documents.append(
            Document(
                doc_id=doc_id,
                title=_title_of(raw, doc_id),
                text=body,
                url=url_for(doc_id) if url_for else "",
            )
        )
    return documents
