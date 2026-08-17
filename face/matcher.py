"""Cosine similarity between two face embeddings, compared against a configurable match threshold."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    """The outcome of comparing two face embeddings: the raw similarity score and the threshold-based pass/fail label."""

    similarity_score: float
    is_match: bool


def compute_cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two face embeddings.

    InsightFace's normed_embedding is already L2-normalized, so this reduces to a plain dot
    product -- using np.dot directly (rather than a generic formula with explicit norm division)
    keeps that assumption visible at the call site.
    """
    return float(np.dot(embedding_a, embedding_b))


def match_faces(embedding_a: np.ndarray, embedding_b: np.ndarray, threshold: float) -> MatchResult:
    """Compare two face embeddings and return both the raw similarity score and the automated pass/fail label.

    The automated label is only a starting point for the operator: the UI must always show the
    raw score alongside it and allow a manual override, since this threshold has not yet been
    calibrated against real webcam-vs-ID-photo capture pairs (see config/settings.py).
    """
    score = compute_cosine_similarity(embedding_a, embedding_b)
    return MatchResult(similarity_score=score, is_match=score >= threshold)
