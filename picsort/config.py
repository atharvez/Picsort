"""
Configuration structures and Enums for PicSort.
"""

from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


class SortMode(str, Enum):
    KNOWN = "known"
    CLUSTER = "cluster"


class FileAction(str, Enum):
    COPY = "copy"
    MOVE = "move"
    SYMLINK = "symlink"


class DetectorEngine(str, Enum):
    INSIGHTFACE = "insightface"
    DLIB = "dlib"


@dataclass
class AppConfig:
    source_dir: Path
    output_dir: Path
    mode: SortMode = SortMode.KNOWN
    people_dir: Optional[Path] = None
    action: FileAction = FileAction.COPY
    detector: DetectorEngine = DetectorEngine.INSIGHTFACE
    similarity_threshold: float = 0.5
    cluster_eps: float = 0.4
    cluster_min_samples: int = 2
    force_reprocess: bool = False
    cache_db_path: Path = field(default_factory=lambda: Path(".picsort_cache.db"))
    unknown_folder_name: str = "unknown_or_no_face"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".heic"]
    )
