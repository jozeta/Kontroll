"""Wraps InsightFace face detection on a captured frame."""

import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face


def load_face_analysis_model() -> FaceAnalysis:
    """Load the buffalo_l detection+recognition model pack. Expensive -- call once at app startup.

    InsightFace's buffalo_l pack bundles a RetinaFace detector and an ArcFace-family recognition
    model behind one FaceAnalysis object; there is no supported way to load detection and
    embedding as two independent models, so this single load is shared by both detector.py and
    embedder.py rather than repeated. ctx_id=-1 forces CPU execution, matching this project's
    plain (non-GPU) onnxruntime dependency.
    """
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detect_faces(app: FaceAnalysis, image: np.ndarray) -> list[Face]:
    """Run face detection (and, inseparably, recognition) on a BGR image, returning all detected faces."""
    return app.get(image)
