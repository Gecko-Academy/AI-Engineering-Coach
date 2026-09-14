"""A course coach you are meant to improve.

from gecko_ai_coach import corpus, coach
documents = corpus.load(Path("units/en"))
print(coach.answer("how do I hand in a session", documents).pages)
"""

from gecko_ai_coach.coach import Answer, answer
from gecko_ai_coach.corpus import load
from gecko_ai_coach.measure import Case, Report, run
from gecko_ai_coach.retrieve import Chunk, Document, ScoredChunk, retrieve

__all__ = [
    "Answer",
    "Case",
    "Chunk",
    "Document",
    "Report",
    "ScoredChunk",
    "answer",
    "load",
    "retrieve",
    "run",
]
__version__ = "0.1.0"
