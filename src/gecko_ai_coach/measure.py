"""The number that says how good the coach is, so "better" is not an opinion.

WHY THIS FILE EXISTS AT ALL. Every change to a retriever feels like an
improvement while you are making it. The only way to know is a labelled set and
a number before and afterwards, and the only way to be honest about it is to
report the regression too -- a rule that is easy to write down and unpleasant to
follow, which is exactly why it is the contribution bar for this project.

WHAT IS MEASURED. Hit rate at k: the share of questions whose expected page
appears in the top k retrieved. It measures RETRIEVAL, not prose, on purpose --
retrieval is where the headroom is, it needs no model, and it is the same number
on every machine. Fluency is not measured here because nobody has a cheap
honest way to measure it, and a metric you cannot trust is worse than none.

THE TRAP THIS CANNOT SAVE YOU FROM. Tune on these questions and the number goes
up while the coach gets worse for everyone else. Keep a second set you did not
tune on, report both, and expect the second one to move less.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gecko_ai_coach.coach import Retriever
from gecko_ai_coach.retrieve import Document, retrieve


class CaseError(Exception):
    """Raised when a labelled set is missing or malformed."""


@dataclass(frozen=True)
class Case:
    """One labelled question.

    `expect_refusal` cases are the ones the corpus cannot answer, and they carry
    their weight: a retriever that returns something for everything scores well
    on the others and is useless in practice.
    """

    question: str
    expected: tuple[str, ...] = ()
    expect_refusal: bool = False


@dataclass(frozen=True)
class Outcome:
    case: Case
    passed: bool
    got: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    outcomes: tuple[Outcome, ...]
    top_k: int

    @property
    def hit_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(o.passed for o in self.outcomes) / len(self.outcomes)

    def rendered(self) -> str:
        lines = [
            f"  hit rate @{self.top_k}: {self.hit_rate:.0%}  ({len(self.outcomes)} questions)",
            "",
        ]
        for outcome in self.outcomes:
            mark = "ok  " if outcome.passed else "MISS"
            lines.append(f"  {mark} {outcome.case.question}")
            if not outcome.passed:
                wanted = ", ".join(outcome.case.expected) or "a refusal"
                found = ", ".join(outcome.got) or "nothing"
                lines.append(f"         wanted {wanted}")
                lines.append(f"         got    {found}")
        return "\n".join(lines)


def load_cases(path: Path) -> list[Case]:
    """Read a JSONL labelled set: one object per line.

    {"question": "how do I hand in a session", "expected": ["unit0/how-to-submit"]}
    {"question": "what is the airspeed of a swallow", "expect_refusal": true}
    """
    if not path.is_file():
        raise CaseError(f"no labelled set at {path}")
    cases: list[Case] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CaseError(f"{path}:{number} is not JSON: {error}") from error
        question = str(row.get("question", "")).strip()
        if not question:
            raise CaseError(f"{path}:{number} has no question")
        cases.append(
            Case(
                question=question,
                expected=tuple(row.get("expected", ())),
                expect_refusal=bool(row.get("expect_refusal", False)),
            )
        )
    if not cases:
        raise CaseError(f"{path} holds no cases")
    return cases


def run(
    cases: Sequence[Case],
    documents: Sequence[Document],
    retriever: Retriever = retrieve,
    top_k: int = 3,
) -> Report:
    """Score a retriever against a labelled set. No model is called."""
    outcomes: list[Outcome] = []
    for case in cases:
        scored = retriever(case.question, documents, top_k)
        got: list[str] = []
        for hit in scored:
            if hit.chunk.doc_id not in got:
                got.append(hit.chunk.doc_id)
        if case.expect_refusal:
            passed = not scored
        else:
            passed = any(page in got for page in case.expected)
        outcomes.append(Outcome(case=case, passed=passed, got=tuple(got)))
    return Report(outcomes=tuple(outcomes), top_k=top_k)
