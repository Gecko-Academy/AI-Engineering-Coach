"""Answer a question from a corpus, and never from anywhere else.

THE SHAPE OF THE ANSWER IS THE PRODUCT. A coach that says something plausible
about a course it has not read is worse than one that says "not in these pages",
because the first kind is only caught by the learner already knowing the answer.
So every layer here is built to make the honest outcome the easy one:

  nothing retrieved            -> refuse, before a model is called at all
  no model configured          -> show the passages, cite the pages, say nothing more
  model configured             -> ground it in those passages, then VERIFY the
                                  citations against what was actually retrieved

WHERE THE HEADROOM IS, IN ORDER. This is the part worth arguing with, and the
order is measured rather than assumed:

  1. retrieval -- whether the right passage is in the prompt at all
  2. how much you give it -- three tight chunks beat ten loose ones, and small
     models degrade with long context faster than large ones do
  3. the output shape -- a narrow schema is followed far more reliably than an
     open instruction, at every model size
  4. making refusal legitimate -- models hallucinate hardest when refusing feels
     forbidden
  5. verification after generation -- cheap, model-independent, and the only
     thing here that catches a confident fabrication
  6. the model itself

Most people arrive believing 6 is the whole job. `ai-coach measure` is there
to disagree with them in numbers.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ai_engineering_coach.models import Client, EchoClient, ModelError
from ai_engineering_coach.retrieve import Document, ScoredChunk, retrieve

Retriever = Callable[[str, Sequence[Document], int], list[ScoredChunk]]

#: Narrow on purpose. Every clause in it earns its place by changing what a 7B
#: model does, and the refusal clause earns its place twice.
SYSTEM = """You answer questions about a course, using only the passages given.

Rules:
- Use only the passages. Do not use anything you know from elsewhere.
- Cite the page id of every passage you used, in square brackets, like [session-03/introduction].
- If the passages do not answer the question, reply exactly: NOT IN THESE PAGES
- Answer in at most four sentences. Do not add a preamble.

Replying NOT IN THESE PAGES is a correct and expected answer. It is better than
a plausible guess."""

REFUSAL = "NOT IN THESE PAGES"

_CITATION_RE = re.compile(r"\[([^\]\s]+)\]")


@dataclass(frozen=True)
class Answer:
    """What the coach found, what it said, and whether it stands behind it."""

    question: str
    passages: tuple[ScoredChunk, ...] = ()
    prose: str = ""
    refused: bool = False
    reason: str = ""
    #: Pages the model cited that were never retrieved. Stripped from `prose`,
    #: kept here, because a fabrication you delete silently is a fabrication you
    #: will not notice again.
    fabricated: tuple[str, ...] = field(default_factory=tuple)

    @property
    def pages(self) -> tuple[str, ...]:
        """The page ids behind the answer, in the order they were retrieved."""
        seen: list[str] = []
        for scored in self.passages:
            if scored.chunk.doc_id not in seen:
                seen.append(scored.chunk.doc_id)
        return tuple(seen)


def prompt_for(question: str, passages: Sequence[ScoredChunk]) -> str:
    """The user half of the call: the passages, then the question.

    Each passage is labelled with the page id the model is asked to cite, so
    citing correctly requires no memory and no inference -- the id is on the
    page in front of it. Making the right behaviour mechanical is most of what
    prompt engineering is for.
    """
    blocks = [
        f"--- page: {scored.chunk.doc_id} ({scored.chunk.title})\n{scored.chunk.text}"
        for scored in passages
    ]
    return "\n\n".join(blocks) + f"\n\n--- question\n{question}"


def verify(prose: str, allowed: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Strip citations naming a page that was not retrieved, and report them.

    Model-independent and cheap, which is the point: it catches a confident
    fabrication from any model, including one much larger than the one you
    tested with. A citation that survives this has a page behind it.
    """
    permitted = set(allowed)
    fabricated: list[str] = []

    def keep(match: re.Match[str]) -> str:
        cited = match.group(1)
        if cited in permitted:
            return match.group(0)
        if cited not in fabricated:
            fabricated.append(cited)
        return ""

    cleaned = _CITATION_RE.sub(keep, prose)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip(), tuple(fabricated)


def answer(
    question: str,
    documents: Sequence[Document],
    client: Client | None = None,
    retriever: Retriever = retrieve,
    top_k: int = 3,
) -> Answer:
    """Retrieve, then ground a model in what came back — or quote it directly.

    `retriever` is injected for the same reason the client is: replacing it is
    the exercise. Anything with the signature `(query, documents, top_k)` works,
    so a better retriever is a one-line experiment rather than a fork.
    """
    question = question.strip()
    if not question:
        return Answer(question=question, refused=True, reason="no question was asked")

    passages = tuple(retriever(question, documents, top_k))
    if not passages:
        # Before the model, not after. A refusal that costs a call is a refusal
        # that will be skipped the first time somebody is counting calls.
        return Answer(
            question=question,
            refused=True,
            reason="nothing in these pages shares a word with that question",
        )

    if client is None or isinstance(client, EchoClient):
        return Answer(question=question, passages=passages)

    try:
        raw = client.complete(SYSTEM, prompt_for(question, passages)).strip()
    except ModelError as error:
        # The passages are still worth having. A model that is down degrades the
        # coach to its quoting lane rather than taking it away.
        return Answer(question=question, passages=passages, reason=str(error))

    if REFUSAL in raw.upper():
        return Answer(
            question=question,
            passages=passages,
            refused=True,
            reason="the model found no answer in the retrieved pages",
        )

    prose, fabricated = verify(raw, [scored.chunk.doc_id for scored in passages])
    return Answer(question=question, passages=passages, prose=prose, fabricated=fabricated)
