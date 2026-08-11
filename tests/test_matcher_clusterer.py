"""
Unit tests for matcher and clusterer modules using synthetic vector embeddings.
"""

import numpy as np
from picsort.matcher import KnownFaceMatcher
from picsort.clusterer import FaceClusterer


class MockDetector:
    pass


def test_matcher_and_clusterer():
    # 1. Test Matcher
    matcher = KnownFaceMatcher(detector=MockDetector(), similarity_threshold=0.5)

    # Define synthetic reference vectors for "Alice" and "Bob"
    v_alice = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_bob = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    matcher.known_people_embeddings = {
        "Alice": [v_alice],
        "Bob": [v_bob],
    }

    # Query vector close to Alice
    q_alice = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    person, score = matcher.match_face(q_alice)
    assert person == "Alice", f"Should match Alice, got {person}"
    assert score > 0.5

    # Query vector close to Bob
    q_bob = np.array([0.1, 0.95, 0.0, 0.0], dtype=np.float32)
    person, score = matcher.match_face(q_bob)
    assert person == "Bob", f"Should match Bob, got {person}"

    # Query vector orthogonal to both
    q_unknown = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    person, score = matcher.match_face(q_unknown)
    assert person is None, f"Should be unmatched (None), got {person}"

    print("Matcher test passed.")

    # 2. Test Clusterer
    # Group 1: near [1, 0, 0, 0]
    g1_1 = np.array([1.0, 0.05, 0.0, 0.0], dtype=np.float32)
    g1_2 = np.array([0.98, 0.1, 0.0, 0.0], dtype=np.float32)

    # Group 2: near [0, 1, 0, 0]
    g2_1 = np.array([0.0, 1.0, 0.05, 0.0], dtype=np.float32)
    g2_2 = np.array([0.0, 0.95, 0.1, 0.0], dtype=np.float32)

    # Noise vector
    g_noise = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    records = [
        {"embedding": g1_1, "file_path": "photo1.jpg"},
        {"embedding": g1_2, "file_path": "photo2.jpg"},
        {"embedding": g2_1, "file_path": "photo3.jpg"},
        {"embedding": g2_2, "file_path": "photo4.jpg"},
        {"embedding": g_noise, "file_path": "photo5.jpg"},
    ]

    clusterer = FaceClusterer(eps=0.3, min_samples=2)
    cmap, updated = clusterer.cluster_faces(records)

    assert len(cmap) == 2, f"Should find 2 clusters, found {len(cmap)}"
    assert updated[0]["assigned_person"] == updated[1]["assigned_person"]
    assert updated[2]["assigned_person"] == updated[3]["assigned_person"]
    assert updated[4]["assigned_person"] is None

    print("Clusterer test passed.")
    print("SUCCESS: Matcher & Clusterer unit tests passed successfully.")


if __name__ == "__main__":
    test_matcher_and_clusterer()
