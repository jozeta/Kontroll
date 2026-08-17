"""Produces a 512-d face embedding from a detected face."""

import numpy as np
from insightface.app.common import Face


def get_best_face_embedding(faces: list[Face]) -> np.ndarray | None:
    """Return the L2-normalized 512-d embedding of the most confidently detected face, or None if no face was found.

    When multiple faces appear in a frame (e.g. someone passes behind the operator), the
    highest detection-confidence face is assumed to be the intended subject -- v1 has no
    requirement to prompt the operator to choose among multiple detected faces.
    """
    if len(faces) == 0:
        return None
    best_face = max(faces, key=lambda face: face.det_score)
    return best_face.normed_embedding
