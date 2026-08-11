"""
Base abstract interface for face detection and embedding extraction engines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class BaseFaceDetector(ABC):
    """Abstract Base Class for face detection engines."""

    @abstractmethod
    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a BGR uint8 numpy image and return facial bounding boxes and embeddings.

        Returns:
            List of dicts:
            [
                {
                    "bbox": [xmin, ymin, xmax, ymax],  # int coordinates
                    "embedding": np.ndarray,           # 1D float32 normalized embedding
                    "score": float                    # confidence score (0.0 to 1.0)
                },
                ...
            ]
        """
        pass
