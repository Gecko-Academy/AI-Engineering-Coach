# Contributing

This project exists to be improved, and a first pull request here is meant to be
a genuinely good one to have made. There is one rule.

## The rule: bring the number

**A pull request that changes retrieval must report the hit rate before and
after, on a named set.** Paste the two `measure` outputs.

```bash
course-coach measure --pages ./units/en --cases data/dev3pack.jsonl
```

That is the whole bar. It is not bureaucracy: without it, neither you nor anyone
reviewing can tell an improvement from a change, and a retriever tuned by feel
gets worse in ways nobody notices for weeks.

**Report the regression too.** If your change helps ten questions and breaks
two, say so and say which. A pull request that reports only the gain is the
single most common way a retrieval project rots, and the two broken questions
are usually more interesting than the ten fixed ones.

**Measure on a set you did not tune on.** If you added questions to make your
change look good, that is fine — add them, and also report the number on the set
as it was before you touched it. Tuning on the set you are measuring is the
oldest mistake in retrieval and it is invisible from the inside.

## The baseline, measured

Against the public Dev3Pack cohort corpus, 67 pages, 2026-09-13:

```
  hit rate @3: 47%  (17 questions)
```

That is what you are beating. It is low on purpose: the questions are phrased
the way a learner asks them, not the way the pages are worded.

## Good first issues, each taken from a real miss

Every one of these is an actual failure in the run above, not a hypothetical.

1. **Weight the page title.** `how do I hand in a session` returns
   `unit0/introduction` and misses `unit0/how-to-submit` — a page whose title is
   almost the question. Title tokens are high signal and nearly free.
2. **Query expansion for the course's own synonyms.** `hand in` never matches
   `submit`; `set up` never matches `install`. One dictionary closes several
   misses. Then measure what it costs on questions it was not written for — in
   this course's own session 7, exactly this change took a probe set from 75% to
   0%, so the regression is the interesting half.
3. **A minimum score, so nothing is returned for nothing.** `how do I deploy a
   Kubernetes ingress controller` currently returns two pages. A question the
   corpus cannot answer should retrieve nothing, and refusing is a correct
   answer here.
4. **Dense pages drown specific ones.** Week-0 `slides` pages are keyword soup
   and outrank the page that actually answers the question. Length normalisation
   is the usual fix; try it and show both numbers.
5. **Chunk size.** `max_chars` is 800 because it had to be something. Nobody has
   measured 400 or 1200.

Bigger, and worth discussing in an issue first: an embedding retriever behind the
same `(query, documents, top_k)` seam. It must stay optional — the baseline runs
with no dependencies, no download and no network, and that is a feature.

## Anything else

- Tests are expected for a behaviour change. The suite is `pytest`, there are no
  mocks of the model beyond a hand-written recorder, and each test's docstring
  says which real failure it pins.
- `ruff check .` and `ruff format .` before you push.
- No new runtime dependencies. A tool people are asked to fork should install
  anywhere in one step.
- Never index a `solutions/` directory, an answer key, or a quiz block. Two tests
  pin this and they are not negotiable: a course tool that can quote the answers
  is a cheat sheet.

## Licence

Apache-2.0. By contributing you agree your contribution is licensed under it.
