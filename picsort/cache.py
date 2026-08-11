"""
SQLite-backed persistent caching system for face detection & embeddings.
Keyed by SHA-256 hash of image file content.
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


class CacheManager:
    """Manages SQLite cache database for storing image hashes and extracted face embeddings."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize SQLite tables if they do not exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    score REAL NOT NULL,
                    FOREIGN KEY (file_hash) REFERENCES images (file_hash) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_faces_file_hash ON faces(file_hash);"
            )
            conn.commit()
        finally:
            conn.close()

    def get_cached_faces(self, file_hash: str) -> Optional[List[Dict[str, Any]]]:
        """
        Check if file_hash exists in cache.
        Returns list of face dicts: [{'bbox': [...], 'embedding': np.ndarray, 'score': float}]
        or None if file is not in cache.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT face_count FROM images WHERE file_hash = ?", (file_hash,)
            )
            row = cursor.fetchone()
            if row is None:
                return None

            cursor.execute(
                "SELECT bbox_json, embedding_blob, score FROM faces WHERE file_hash = ?",
                (file_hash,),
            )
            face_rows = cursor.fetchall()
            
            faces = []
            for fr in face_rows:
                bbox = json.loads(fr["bbox_json"])
                embedding = np.frombuffer(fr["embedding_blob"], dtype=np.float32)
                score = float(fr["score"])
                faces.append(
                    {
                        "bbox": bbox,
                        "embedding": embedding,
                        "score": score,
                    }
                )
            return faces
        finally:
            conn.close()

    def save_faces(
        self, file_hash: str, file_path: str, faces: List[Dict[str, Any]]
    ):
        """Save image hash and associated face embeddings to database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Upsert into images table
            cursor.execute(
                """
                INSERT OR REPLACE INTO images (file_hash, file_path, face_count)
                VALUES (?, ?, ?)
                """,
                (file_hash, file_path, len(faces)),
            )
            # Clear old faces for this hash
            cursor.execute("DELETE FROM faces WHERE file_hash = ?", (file_hash,))

            for face in faces:
                bbox_json = json.dumps(face.get("bbox", []))
                embedding = np.asarray(face["embedding"], dtype=np.float32)
                embedding_blob = embedding.tobytes()
                score = float(face.get("score", 1.0))

                cursor.execute(
                    """
                    INSERT INTO faces (file_hash, bbox_json, embedding_blob, score)
                    VALUES (?, ?, ?, ?)
                    """,
                    (file_hash, bbox_json, embedding_blob, score),
                )
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, int]:
        """Return total cached images and faces count."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images")
            img_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM faces")
            face_count = cursor.fetchone()[0]
            return {"cached_images": img_count, "cached_faces": face_count}
        finally:
            conn.close()
