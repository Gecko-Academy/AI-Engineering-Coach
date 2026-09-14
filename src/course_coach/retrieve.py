"""Lexical retrieval: deterministic, cheap, and readable on sight.

This is deliberately not an embedding index. A keyword baseline is debuggable by
eye, needs no dependencies and no GPU, and sets the bar a fancier retriever must
beat on a labelled set before it earns its place.

IT IS ALSO DELIBERATELY MEDIOCRE. This file is the thing you are invited to
improve. Every knob is in it and nothing is hidden: the stopword list, the chunk
size, how a score is computed, how ties break. `course-coach measure` prints the
number that says how mediocre, and a pull request that moves it -- and says by
how much, on which set -- is the contribution this project wants.

THE ORDER THAT ACTUALLY MOVES THE NUMBER, measured rather than assumed:
retrieval first, prompt shape second, model last. A 7B model holding the right
passage beats a frontier model holding the wrong one, so most of the headroom
is in this file rather than in whichever model you point at it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Words too common to signal relevance. Hand-written, English, and one of the
#: easiest things here to do better -- it knows nothing about the vocabulary of
#: the corpus it is pointed at.
STOPWORDS = frozenset(
    "a an and are as at be but by can could did do does for from has have how i if in "
    "into is it its me my no not of on one only or our over should so some than that "
    "the their then they this to under was we what when where which who why will with "
    "would you your".split()
)


@dataclass(frozen=True)
class Document:
    """A page of the corpus. `doc_id` is what a citation names."""

    doc_id: str
    title: str
    text: str
    url: str = ""


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    text: str
    position: int
    url: str = ""


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def chunk_document(doc: Document, max_chars: int = 800) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most `max_chars` characters.

    Paragraphs are kept whole because a sentence cut in half retrieves badly and
    reads worse. A single oversized paragraph still becomes its own chunk,
    truncated -- losing the tail is better than losing the paragraph.
    """
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current: list[str] = []
    length = 0
    for paragraph in paragraphs:
        if length and length + len(paragraph) + 2 > max_chars:
            chunks.append(Chunk(doc.doc_id, doc.title, "\n\n".join(current), len(chunks), doc.url))
            current, length = [], 0
        current.append(paragraph[:max_chars])
        length += len(paragraph) + 2
    if current:
        chunks.append(Chunk(doc.doc_id, doc.title, "\n\n".join(current), len(chunks), doc.url))
    return chunks


def retrieve(
    query: str,
    documents: Sequence[Document],
    top_k: int = 3,
    max_chars: int = 800,
) -> list[ScoredChunk]:
    """Score chunks by query-token overlap, weighted by inverse chunk frequency.

    Returns `[]` when no chunk shares a token with the query, which is what lets
    a caller refuse before spending a model call. An empty result is a correct
    answer, not a failure.

    The sort key is load-bearing: `(-score, doc_id, position)` makes the same
    question return the same passages on every machine, so a measured
    improvement is a property of the change and not of the hardware.
    """
    query_tokens = set(tokens(query))
    if not query_tokens:
        return []
    all_chunks = [chunk for doc in documents for chunk in chunk_document(doc, max_chars)]
    if not all_chunks:
        return []

    frequency: dict[str, int] = {}
    per_chunk: list[set[str]] = []
    for chunk in all_chunks:
        found = set(tokens(chunk.text))
        per_chunk.append(found)
        for token in found & query_tokens:
            frequency[token] = frequency.get(token, 0) + 1

    total = len(all_chunks)
    scored: list[ScoredChunk] = []
    for chunk, found in zip(all_chunks, per_chunk, strict=True):
        overlap = found & query_tokens
        if not overlap:
            continue
        score = sum(math.log(1 + total / frequency[token]) for token in overlap)
        scored.append(ScoredChunk(chunk=chunk, score=round(score, 6)))

    scored.sort(key=lambda s: (-s.score, s.chunk.doc_id, s.chunk.position))
    return scored[:top_k]
