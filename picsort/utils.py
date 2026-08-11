"""
Utility functions for file handling, image loading (with HEIC & EXIF support), SHA256 hashing, and logging.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps

# Attempt to register pillow-heif decoder for HEIC files
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configure and return the root logger for PicSort."""
    logger = logging.getLogger("picsort")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger


def calculate_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for change detection and caching."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_image_files(directory: Path, extensions: List[str]) -> List[Path]:
    """Recursively scan a directory for supported image extensions."""
    if not directory.exists() or not directory.is_dir():
        return []
    
    ext_set = {e.lower() for e in extensions}
    image_paths = []
    
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in ext_set:
            image_paths.append(path)
            
    return sorted(image_paths)


def load_image_rgb(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load an image file safely into an RGB uint8 numpy array (H, W, 3).
    Handles EXIF auto-rotation and HEIC formats.
    Returns (image_array, error_message).
    """
    try:
        with Image.open(file_path) as img:
            # Auto-rotate image according to EXIF Orientation tag
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
            return arr, None
    except Exception as e:
        return None, f"Failed to load image '{file_path.name}': {str(e)}"


def load_image_bgr(file_path: Path) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load an image file safely into a BGR uint8 numpy array (H, W, 3) for OpenCV/InsightFace.
    Returns (image_array, error_message).
    """
    rgb_arr, err = load_image_rgb(file_path)
    if err or rgb_arr is None:
        return None, err
    # Convert RGB to BGR
    bgr_arr = rgb_arr[:, :, ::-1].copy()
    return bgr_arr, None
