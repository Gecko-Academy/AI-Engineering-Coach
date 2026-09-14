"""A course coach you are meant to improve.

from ai_engineering_coach import corpus, coach
documents = corpus.load(Path("units/en"))
print(coach.answer("how do I hand in a session", documents).pages)
"""

from ai_engineering_coach.coach import Answer, answer
from ai_engineering_coach.corpus import load
from ai_engineering_coach.measure import Case, Report, run
from ai_engineering_coach.retrieve import Chunk, Document, ScoredChunk, retrieve

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
