"""What must stay true, each pinned because it is a way the coach goes wrong.

The doubles here are hand-written. A coach is a thing that talks to a model, and
a test suite that mocks the model into agreeing with it proves nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from course_coach import corpus
from course_coach.coach import answer, verify
from course_coach.measure import Case, load_cases, run
from course_coach.models import EchoClient, ModelError, get_client
from course_coach.retrieve import Document, retrieve


class Recorder:
    """A model that records what it was asked and answers what it was told to."""

    def __init__(self, reply: str = "an answer [one]") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.reply


class Broken:
    def complete(self, system: str, user: str) -> str:
        raise ModelError("no model server at http://localhost:11434/v1")


PAGES = [
    Document("one", "Handing work in", "Open a pull request against the submissions repository."),
    Document("two", "Structured outputs", "A strict parser rejects an unknown field by name."),
]


def test_a_question_nothing_matches_refuses_without_calling_the_model() -> None:
    """The refusal must happen BEFORE the call, not be produced by it.

    Asserted on the recorder rather than on the printed text, because a coach
    that calls a model and then throws the answer away still costs a call, still
    takes the latency, and still fails closed only by luck.
    """
    model = Recorder()
    result = answer("xylophone quarterly dividend", PAGES, client=model)

    assert result.refused
    assert model.calls == [], "the model was called for a question nothing matched"


def test_no_model_configured_still_answers_with_passages() -> None:
    """The default lane needs no key, no download and no network."""
    result = answer("open a pull request", PAGES, client=EchoClient())

    assert not result.refused
    assert result.prose == ""
    assert result.pages == ("one",)


def test_an_invented_citation_is_stripped_and_reported() -> None:
    """A page the model cited that retrieval never returned is a fabrication.

    Removing it silently would be worse than leaving it: the whole value is in
    seeing that the model did it.
    """
    model = Recorder(reply="Open a pull request [one]. It is graded nightly [invented-page].")
    result = answer("open a pull request", PAGES, client=model)

    assert "[one]" in result.prose
    assert "invented-page" not in result.prose
    assert result.fabricated == ("invented-page",)


def test_the_model_is_given_the_page_ids_it_is_asked_to_cite() -> None:
    """Citing correctly must require no memory: the id is in front of it."""
    model = Recorder()
    answer("open a pull request", PAGES, client=model)

    _, user = model.calls[0]
    assert "--- page: one" in user
    assert "--- question" in user


def test_a_model_that_is_down_degrades_to_the_passages() -> None:
    """A broken lane must not take the coach away, only its prose."""
    result = answer("open a pull request", PAGES, client=Broken())

    assert not result.refused
    assert result.pages == ("one",)
    assert "no model server" in result.reason


def test_the_same_question_returns_the_same_passages() -> None:
    first = answer("strict parser unknown field", PAGES, client=EchoClient())
    second = answer("strict parser unknown field", PAGES, client=EchoClient())

    assert [p.chunk.text for p in first.passages] == [p.chunk.text for p in second.passages]


def test_the_retriever_is_replaceable_without_editing_the_coach() -> None:
    """The seam the bonus unit exists to use."""
    called: list[str] = []

    def mine(query, documents, top_k):  # noqa: ANN001, ANN202 - the seam is duck-typed
        called.append(query)
        return retrieve(query, documents, top_k)

    result = answer("open a pull request", PAGES, client=EchoClient(), retriever=mine)

    assert called == ["open a pull request"]
    assert result.pages == ("one",)


def test_verify_leaves_a_citation_that_was_retrieved() -> None:
    cleaned, fabricated = verify("see [one] and [two]", ["one", "two"])

    assert cleaned == "see [one] and [two]"
    assert fabricated == ()


# --- the corpus ------------------------------------------------------------


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_quiz_blocks_never_enter_the_corpus(tmp_path: Path) -> None:
    """An inline quiz carries `correct: true` in its own markup.

    Indexing it would let the coach quote the answer key back, which is the one
    thing a course tool must never do.
    """
    _write(
        tmp_path,
        "quiz.mdx",
        "# Quiz\n\nPick one.\n\n"
        '<Question choices={[{ text: "Rejects it", correct: true,'
        ' explain: "the gate is equality" }]} />\n\n'
        "After the quiz.\n",
    )
    documents = corpus.load(tmp_path)

    assert len(documents) == 1
    assert "correct" not in documents[0].text
    assert "the gate is equality" not in documents[0].text
    assert "After the quiz." in documents[0].text


def test_a_slash_in_prose_does_not_end_a_component_early(tmp_path: Path) -> None:
    """Course prose is full of slashes; `and/or` must not close the block."""
    _write(
        tmp_path,
        "quiz.mdx",
        "# Quiz\n\n"
        '<Question choices={[{ text: "read and/or write", correct: true }]} />\n\n'
        "Kept.\n",
    )
    documents = corpus.load(tmp_path)

    assert "read and/or write" not in documents[0].text
    assert "Kept." in documents[0].text


def test_solutions_are_never_indexed(tmp_path: Path) -> None:
    _write(tmp_path, "page.mdx", "# A page\n\nProse.\n")
    _write(tmp_path, "solutions/answer.md", "# The answer\n\nIt is 42.\n")
    documents = corpus.load(tmp_path)

    assert [d.doc_id for d in documents] == ["page"]


def test_the_doc_id_is_a_path_a_human_can_open(tmp_path: Path) -> None:
    _write(tmp_path, "unit1/session-03/introduction.mdx", "# Structured outputs\n\nProse.\n")
    documents = corpus.load(tmp_path)

    assert documents[0].doc_id == "unit1/session-03/introduction"
    assert documents[0].title == "Structured outputs"


def test_a_missing_corpus_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        corpus.load(tmp_path / "not-here")


# --- measuring -------------------------------------------------------------


def test_hit_rate_counts_the_expected_page_in_the_top_k() -> None:
    cases = [
        Case(question="open a pull request", expected=("one",)),
        Case(question="strict parser unknown field", expected=("two",)),
        Case(question="xylophone quarterly dividend", expect_refusal=True),
    ]
    report = run(cases, PAGES)

    assert report.hit_rate == 1.0


def test_a_retriever_that_answers_everything_fails_the_refusal_case() -> None:
    """The case that stops "return something always" from scoring well."""

    def greedy(query, documents, top_k):  # noqa: ANN001, ANN202
        return retrieve("pull request parser", documents, top_k)

    report = run([Case(question="xylophone", expect_refusal=True)], PAGES, retriever=greedy)

    assert report.hit_rate == 0.0


def test_a_labelled_set_that_is_not_json_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text('{"question": "fine", "expected": ["one"]}\nnot json\n', encoding="utf-8")

    with pytest.raises(Exception) as error:
        load_cases(path)

    assert ":2" in str(error.value)


# --- providers -------------------------------------------------------------


def test_no_provider_configured_is_the_echo_lane_not_an_error() -> None:
    assert isinstance(get_client(env={}), EchoClient)


def test_a_provider_that_needs_a_key_says_which_variable(tmp_path: Path) -> None:
    with pytest.raises(ModelError) as error:
        get_client("moonshot", "kimi-latest", env={})

    assert "MOONSHOT_API_KEY" in str(error.value)


def test_an_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ModelError) as error:
        get_client("hal9000", env={})

    assert "moonshot" in str(error.value)
    assert "ollama" in str(error.value)


def test_the_known_weakness_is_a_vocabulary_mismatch() -> None:
    """A question in the learner's words, about a page in ours, retrieves nothing.

    "hand work in" and "open a pull request" mean the same thing and share no
    token, so the baseline returns `[]` and the coach refuses. This is not a bug
    to hide in a test that avoids it: it is the single biggest thing wrong with
    keyword retrieval, it is why `measure` exists, and closing it -- with query
    expansion, or title weighting, or something better -- is the first issue
    anybody should pick up.

    If your change makes this pass, say so in the pull request, and say what it
    cost on the questions it was not tuned on.
    """
    result = answer("how do I hand work in", PAGES, client=EchoClient())

    assert result.refused
    assert "shares a word" in result.reason
