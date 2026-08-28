"""Produces a 512-d face embedding from a detected face."""

import numpy as np
from insightface.app.common import Face


def _face_bbox_area(face: Face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def get_best_face_embedding(faces: list[Face]) -> np.ndarray | None:
    """Return the L2-normalized 512-d embedding of the largest detected face, or None if no face was found.

    When multiple faces appear in a frame (e.g. someone passes behind the operator, or another
    person in the background), the largest bounding box -- a direct proxy for "closest to the
    camera" -- is assumed to be the intended subject, since this app's cameras are pointed at one
    visitor/document standing right in front of them.

    Deliberately NOT the highest detection-confidence face (an earlier version of this function
    picked that instead): diagnosed against a real capture with two people in frame, RetinaFace
    scored a small, off-center background face at det_score=0.877 and the actual subject -- a
    large, centered, much-closer face nearly filling the frame -- at only 0.812. Detector
    confidence reflects how "textbook" a face crop looks to the model (an extreme close-up scores
    slightly lower than a conventionally-framed one), not which face belongs to the person standing
    in front of the camera -- bounding-box size directly encodes that instead. v1 has no
    requirement to prompt the operator to choose among multiple detected faces.
    """
    if len(faces) == 0:
        return None
    best_face = max(faces, key=_face_bbox_area)
    return best_face.normed_embedding
