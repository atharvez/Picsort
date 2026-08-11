"""
Auto-clustering module for grouping unidentified face embeddings into person_1, person_2, etc. using DBSCAN.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger("picsort")


class FaceClusterer:
    """Cluster face embeddings using DBSCAN with Cosine metric."""

    def __init__(self, eps: float = 0.4, min_samples: int = 2):
        """
        eps: Maximum cosine distance between two samples for one to be considered in the neighborhood of the other.
        min_samples: Minimum number of samples in a neighborhood for a point to be considered as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples

    def cluster_faces(
        self, face_records: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
        """
        Perform DBSCAN clustering on a list of face records.
        Each face_record must contain: {'file_hash': str, 'embedding': np.ndarray, ...}

        Returns:
            Tuple of:
            - cluster_name_map: Dict[cluster_label, person_folder_name] (e.g. 0 -> "person_1", -1 -> "unknown_or_no_face")
            - updated_face_records: List of face records with an added 'assigned_person' key.
        """
        if not face_records:
            return {}, []

        # Extract normalized embeddings matrix
        embeddings = []
        valid_indices = []
        for idx, record in enumerate(face_records):
            emb = record["embedding"]
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
            valid_indices.append(idx)

        embeddings_matrix = np.array(embeddings, dtype=np.float32)

        logger.info(
            f"Running DBSCAN auto-clustering on {len(embeddings)} faces with eps={self.eps}, min_samples={self.min_samples}..."
        )

        # Run DBSCAN using cosine metric
        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="cosine"
        ).fit(embeddings_matrix)

        labels = clustering.labels_

        # Map integer cluster labels to human-readable person directory names
        unique_labels = set(labels)
        cluster_name_map: Dict[int, str] = {}
        person_counter = 1

        for label in sorted(unique_labels):
            if label == -1:
                # Noise / unclustered faces
                continue
            cluster_name_map[label] = f"person_{person_counter}"
            person_counter += 1

        # Assign cluster person names back to face records
        updated_records = []
        for idx, label in enumerate(labels):
            record = dict(face_records[idx])
            if label in cluster_name_map:
                record["assigned_person"] = cluster_name_map[label]
            else:
                record["assigned_person"] = None  # Noise/unassigned
            updated_records.append(record)

        num_clusters = len(cluster_name_map)
        num_noise = sum(1 for l in labels if l == -1)
        logger.info(
            f"Clustering complete: Found {num_clusters} person cluster(s) and {num_noise} unassigned face(s)."
        )

        return cluster_name_map, updated_records
