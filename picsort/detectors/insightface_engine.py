"""
InsightFace detection engine implementation using ONNX Runtime with CUDA / CPU auto-detection.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from picsort.detectors.base import BaseFaceDetector

logger = logging.getLogger("picsort")

try:
    import insightface
    from insightface.app import FaceAnalysis
    import onnxruntime as ort
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class InsightFaceEngine(BaseFaceDetector):
    """InsightFace face detection & embedding extraction engine."""

    def __init__(
        self,
        name: str = "buffalo_l",
        det_thresh: float = 0.5,
        det_size: tuple = (640, 640),
        use_gpu: bool = True,
    ):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "InsightFace or ONNX Runtime is not installed. "
                "Please install them via: pip install insightface onnxruntime"
            )

        self.name = name
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.use_gpu = use_gpu

        # Determine execution providers (CUDA vs CPU)
        available_providers = ort.get_available_providers()
        if self.use_gpu and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
            logger.info("InsightFace: CUDA Execution Provider detected and enabled.")
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            logger.info("InsightFace: Using CPU Execution Provider.")

        # Initialize FaceAnalysis app
        self.app = FaceAnalysis(
            name=self.name,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=self.det_thresh, det_size=self.det_size)

    def detect_faces(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces and extract 512D embeddings using InsightFace."""
        if image_bgr is None or image_bgr.size == 0:
            return []

        raw_faces = self.app.get(image_bgr)
        results = []

        for face in raw_faces:
            bbox = face.bbox.astype(int).tolist()  # [xmin, ymin, xmax, ymax]
            score = float(face.det_score) if hasattr(face, "det_score") else 1.0
            
            embedding = face.embedding
            if embedding is not None:
                # L2 normalize vector
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                embedding = embedding.astype(np.float32)

                results.append(
                    {
                        "bbox": bbox,
                        "embedding": embedding,
                        "score": score,
                    }
                )

        return results
