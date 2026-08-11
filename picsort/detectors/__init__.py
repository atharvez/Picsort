"""
Detector factory for creating face detection engine instances.
"""

from picsort.config import DetectorEngine
from picsort.detectors.base import BaseFaceDetector


def get_detector_engine(
    engine_type: DetectorEngine = DetectorEngine.INSIGHTFACE,
    use_gpu: bool = True,
) -> BaseFaceDetector:
    """
    Factory function to instantiate the requested face detection engine.
    """
    if engine_type == DetectorEngine.INSIGHTFACE:
        from picsort.detectors.insightface_engine import InsightFaceEngine
        return InsightFaceEngine(use_gpu=use_gpu)
    elif engine_type == DetectorEngine.DLIB:
        from picsort.detectors.dlib_engine import DlibEngine
        return DlibEngine()
    else:
        raise ValueError(f"Unsupported detector engine: {engine_type}")
