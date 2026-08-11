"""
face_recognition (dlib-based) engine implementation for PicSort.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from picsort.detectors.base import BaseFaceDetector

logger = logging.getLogger("picsort")

try:
    import face_recognition
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class DlibEngine(BaseFaceDetector):
    """face_recognition (dlib-based) face detection & embedding extraction engine."""

    def __init__(self, model: str = "hog", num_jitters: int = 1):
        if not DLIB_AVAILABLE:
            raise ImportError(
                "face_recognition library is not installed. "
                "Please install dlib and face_recognition via: pip install face_recognition"
            )
        self.model = model  # 'hog' or 'cnn'
        self.num_jitters = num_jitters
        logger.info(f"DlibEngine initialized using model='{self.model}'.")

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces and extract 128D embeddings using face_recognition."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        # face_recognition requires RGB image format
        image_rgb = image_bgr[:, :, ::-1]

        # Find face locations: returns list of (top, right, bottom, left)
        face_locations = face_recognition.face_locations(image_rgb, model=self.model)
        if not face_locations:
            return []

        # Get 128D face encodings
        encodings = face_recognition.face_encodings(
            image_rgb, known_face_locations=face_locations, num_jitters=self.num_jitters
        )

        results = []
        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            bbox = [left, top, right, bottom]  # Convert to [xmin, ymin, xmax, ymax]
            
            # L2 normalize
            norm = np.linalg.norm(encoding)
            if norm > 0:
                encoding = encoding / norm
            encoding = encoding.astype(np.float32)

            results.append(
                {
                    "bbox": bbox,
                    "embedding": encoding,
                    "score": 1.0,
                }
            )

        return results
