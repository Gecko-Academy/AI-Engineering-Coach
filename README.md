# gecko-ai-coach

Ask a course a question, get the passage that answers it, and a citation you can
open. Then measure the thing that found it, and make it better.

```bash
pip install gecko-ai-coach

ai-coach ask "how do I hand a session in" --pages ./units/en
ai-coach measure --pages ./units/en --cases data/coach.jsonl
```

No key, no download, no network: the default lane retrieves from the pages and
quotes them. Point it at a model when you want prose.

---

## It is meant to be improved

This is a teaching tool, and the teaching is the improving. The retriever is a
deliberately plain keyword baseline — every knob is in one readable file, and
the number that says how mediocre it is ships with it:

```
  hit rate @3: 74%  (23 questions)

  ok   how do I hand a session in
  MISS what does the strict parser reject
         wanted unit1/session-03-structured-outputs/introduction
         got    unit1/session-04-bounded-tools/introduction
```

A pull request that moves that number — **and says by how much, on which set** —
is the contribution this project wants. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Where the headroom actually is

Measured rather than assumed, and in this order:

1. **Retrieval.** Whether the right passage is in the prompt at all. A 7B model
   holding the right passage beats a frontier model holding the wrong one.
2. **How much you give it.** Three tight chunks beat ten loose ones. Small
   models degrade with long context faster than large ones.
3. **The output shape.** A narrow schema is followed far more reliably than an
   open instruction, at every model size.
4. **Making refusal legitimate.** Models hallucinate hardest when refusing feels
   forbidden. `NOT IN THESE PAGES` is a correct answer here.
5. **Verification after generation.** Every citation is checked against what was
   actually retrieved; invented ones are stripped and reported.
6. **The model.** Last, and usually by less than people expect.

Most people arrive believing 6 is the whole job. `measure` is there to disagree
in numbers.

## Models

Any OpenAI-shaped endpoint, over stdlib `urllib` — no SDK, no wheel to build.

```bash
ai-coach providers
```

| Lane | Key | Notes |
|---|---|---|
| `ollama` | none | local, free, offline. `ollama serve` |
| `moonshot` | `MOONSHOT_API_KEY` | `api.moonshot.cn/v1` from mainland China |
| `openai` | `OPENAI_API_KEY` | |
| `groq` | `GROQ_API_KEY` | |
| `openrouter` | `OPENROUTER_API_KEY` | many models behind one key |
| `echo` | none | no model at all: quote the pages, say nothing more |

Swap one, run `measure`, and see how much it moved. That comparison is the point
of having a seam.

## What it will not do

- **Index a `solutions/` directory or an inline `<Question />` block.** A course
  tool that can quote the answer key is a cheat sheet.
- **Cache an index to disk.** One that can go stale is wrong in a way nobody
  looks at.
- **Read a key from anywhere but the environment**, or put one in an error.

## Library

```python
from pathlib import Path
from gecko_ai_coach import corpus, coach

documents = corpus.load(Path("units/en"))
result = coach.answer("how do I hand a session in", documents)
print(result.pages)          # ('unit0/how-to-submit',)
```

`coach.answer` takes `retriever=` and `client=`, so replacing either is one line
rather than a fork.

## Licence

MIT.
