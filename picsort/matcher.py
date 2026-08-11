"""
Known face reference matcher for PicSort.
Matches query face embeddings against reference embeddings stored in /people/<name>/ folder structure.
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from picsort.config import AppConfig
from picsort.detectors.base import BaseFaceDetector
from picsort.utils import scan_image_files, load_image_bgr

logger = logging.getLogger("picsort")


class KnownFaceMatcher:
    """Manages loading reference faces for known people and matching new face embeddings."""

    def __init__(self, detector: BaseFaceDetector, similarity_threshold: float = 0.5):
        self.detector = detector
        self.similarity_threshold = similarity_threshold
        # Dict mapping person_name -> list of normalized reference embeddings
        self.known_people_embeddings: Dict[str, List[np.ndarray]] = {}

    def load_reference_people(self, people_dir: Path, image_extensions: List[str]):
        """
        Scan /people/<name>/ folders, detect faces, and extract reference embeddings.
        """
        if not people_dir or not people_dir.exists():
            logger.warning(f"People reference directory '{people_dir}' does not exist.")
            return

        person_folders = [p for p in people_dir.iterdir() if p.is_dir()]
        logger.info(f"Loading reference photos for {len(person_folders)} people from '{people_dir}'...")

        for p_dir in person_folders:
            person_name = p_dir.name
            image_files = scan_image_files(p_dir, image_extensions)
            
            embeddings = []
            for img_path in image_files:
                img_bgr, err = load_image_bgr(img_path)
                if err or img_bgr is None:
                    logger.warning(f"Reference image error ({img_path}): {err}")
                    continue

                faces = self.detector.detect_faces(img_bgr)
                for f in faces:
                    emb = f["embedding"]
                    # Normalize vector
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    embeddings.append(emb)

            if embeddings:
                self.known_people_embeddings[person_name] = embeddings
                logger.info(
                    f"Loaded person '{person_name}' with {len(embeddings)} reference face embedding(s)."
                )
            else:
                logger.warning(f"No valid faces detected for person '{person_name}' in '{p_dir}'.")

    def match_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match a query face embedding against known people.
        
        Returns:
            Tuple of (matched_person_name, max_similarity_score)
            If max similarity is below threshold, returns (None, score).
        """
        if not self.known_people_embeddings or face_embedding is None:
            return None, 0.0

        # Ensure query embedding is L2 normalized
        norm = np.linalg.norm(face_embedding)
        if norm > 0:
            face_embedding = face_embedding / norm

        best_person: Optional[str] = None
        best_score: float = -1.0

        for person_name, ref_embeddings in self.known_people_embeddings.items():
            # Calculate cosine similarities against all reference vectors of this person
            similarities = [np.dot(face_embedding, ref) for ref in ref_embeddings]
            max_sim = float(max(similarities)) if similarities else 0.0

            if max_sim > best_score:
                best_score = max_sim
                best_person = person_name

        if best_score >= self.similarity_threshold:
            return best_person, best_score
        else:
            return None, best_score
