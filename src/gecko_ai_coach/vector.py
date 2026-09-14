"""An embedding retriever, behind the same seam as the keyword one.

WHY THIS IS OPTIONAL AND NOT THE DEFAULT. It needs a dependency and a model
download, against a corpus that fits in memory. The baseline needs neither and
answers instantly. So this has to earn its place the same way any change here
does: beat the number on a labelled set, and report what it cost.

WHAT IT IS FOR. The baseline's measured failure is vocabulary: "hand in" never
matches "submit", "set up" never matches "install". No amount of tuning a
keyword scorer fixes that, because the words genuinely do not overlap. An
embedding does, and that is the one thing it is bought for.

WHAT IT IS NOT FOR. Being modern. If it does not beat the keyword baseline on
your set, the honest thing is to say so and keep the baseline -- and that
outcome is worth exactly as much as the other one.

    uv pip install "gecko-ai-coach[vector] @ git+https://github.com/Gecko-Academy/gecko-ai-coach"
    ai-coach measure --pages ./units/en --cases data/dev3pack.jsonl --retriever chroma
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from gecko_ai_coach.retrieve import Chunk, Document, ScoredChunk, chunk_document


class VectorError(Exception):
    """Raised when the optional vector extra is missing or unusable."""


def _client():  # type: ignore[no-untyped-def]
    try:
        import chromadb
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise VectorError(
            'the vector retriever needs chromadb: uv pip install "chromadb"'
        ) from error
    # In memory, deliberately: an index on disk outlives the pages it was built
    # from, and a stale index is wrong in a way nobody looks at. The corpus is
    # small enough that rebuilding costs seconds.
    return chromadb.EphemeralClient()


def _fingerprint(documents: Sequence[Document], max_chars: int) -> str:
    """A name that changes whenever the corpus or the chunking changes.

    Reusing a collection built from different text is the subtle failure here:
    everything works, the numbers move, and nothing says why.
    """
    digest = hashlib.sha256(str(max_chars).encode())
    for doc in documents:
        digest.update(doc.doc_id.encode())
        digest.update(doc.text.encode())
    return "coach-" + digest.hexdigest()[:16]


class ChromaRetriever:
    """Embeds the corpus once, then answers queries by cosine distance.

    Built as a class rather than a function because the index is worth keeping
    between questions; `__call__` gives it the same shape as `retrieve`, so it
    drops into `answer(retriever=...)` and into `measure` unchanged.
    """

    def __init__(
        self, documents: Sequence[Document], max_chars: int = 800, min_score: float = 0.0
    ) -> None:
        self.min_score = min_score
        self._chunks: list[Chunk] = [
            chunk for doc in documents for chunk in chunk_document(doc, max_chars)
        ]
        if not self._chunks:
            raise VectorError("no chunks to index")
        name = _fingerprint(documents, max_chars)
        self._collection = _client().get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        if self._collection.count() == 0:
            self._collection.add(
                ids=[str(position) for position in range(len(self._chunks))],
                # The title is prepended to the embedded text. A page titled
                # "How to submit" is the answer to "how do I hand in", and the
                # title carries that signal far more densely than the body does.
                documents=[f"{chunk.title}\n\n{chunk.text}" for chunk in self._chunks],
            )

    def __call__(
        self, query: str, documents: Sequence[Document], top_k: int = 3
    ) -> list[ScoredChunk]:
        """Retrieve. `documents` is accepted and ignored: the index holds them.

        Ignoring an argument is worth a word. The seam passes the corpus on
        every call because the keyword retriever is stateless; re-embedding here
        on every question would make a measurement run take minutes. Build a new
        retriever when the corpus changes -- the fingerprint above is what makes
        that visible rather than silent.
        """
        if not query.strip():
            return []
        found = self._collection.query(query_texts=[query], n_results=min(top_k, len(self._chunks)))
        ids = (found.get("ids") or [[]])[0]
        distances = (found.get("distances") or [[]])[0]
        scored: list[ScoredChunk] = []
        for identifier, distance in zip(ids, distances, strict=True):
            # Cosine distance in [0, 2]; reported as similarity so a bigger
            # number means a better match, as it does for the keyword scorer.
            similarity = round(1.0 - distance, 6)
            # THE FLOOR, and it is the whole reason this class is not a
            # one-liner. A vector index returns its nearest neighbours however
            # far away they are, so without a floor "nothing in these pages" is
            # unreachable and the coach answers questions about Kubernetes with
            # course material. Measured: no floor gains 3 answerable questions
            # and loses both refusals.
            if similarity < self.min_score:
                continue
            scored.append(ScoredChunk(chunk=self._chunks[int(identifier)], score=similarity))
        return scored


#: Chosen ON the Dev3Pack labelled set, which is exactly the thing this project
#: warns against, so it is written down rather than hidden: it is a default, not
#: a finding. Measure it on your own set before trusting it.
DEFAULT_MIN_SCORE = 0.25


def build(
    documents: Sequence[Document], max_chars: int = 800, min_score: float = DEFAULT_MIN_SCORE
) -> ChromaRetriever:
    """The one entry point, so a caller never imports chromadb itself."""
    return ChromaRetriever(documents, max_chars=max_chars, min_score=min_score)
