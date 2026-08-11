"""
Basic sanity tests for utils and cache modules.
"""

import tempfile
from pathlib import Path
import numpy as np
from picsort.utils import calculate_file_hash
from picsort.cache import CacheManager


def test_cache_and_utils():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello PicSort!")

        # 1. Test Hash
        h1 = calculate_file_hash(test_file)
        assert len(h1) == 64, "SHA-256 hash length should be 64 characters"

        # 2. Test Cache
        db_path = tmp_path / "test_cache.db"
        cache = CacheManager(db_path)

        dummy_embedding = np.random.rand(512).astype(np.float32)
        dummy_faces = [
            {"bbox": [10, 20, 100, 200], "embedding": dummy_embedding, "score": 0.98}
        ]

        cache.save_faces(h1, str(test_file), dummy_faces)

        retrieved = cache.get_cached_faces(h1)
        assert retrieved is not None, "Cache should return saved entry"
        assert len(retrieved) == 1, "Should retrieve 1 face"
        assert np.allclose(retrieved[0]["embedding"], dummy_embedding), "Embeddings should match"
        assert retrieved[0]["bbox"] == [10, 20, 100, 200], "BBox should match"

        stats = cache.get_stats()
        assert stats["cached_images"] == 1
        assert stats["cached_faces"] == 1

        print("SUCCESS: Utils & Cache unit test passed successfully.")


if __name__ == "__main__":
    test_cache_and_utils()
