"""
Unit tests for PhotoOrganizer.
"""

import tempfile
from pathlib import Path
from picsort.config import FileAction
from picsort.organizer import PhotoOrganizer


def test_organizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_dir = tmp_path / "source"
        out_dir = tmp_path / "output"
        src_dir.mkdir()

        # Create dummy photos
        p1 = src_dir / "photo1.jpg"
        p2 = src_dir / "photo2.jpg"
        p3 = src_dir / "photo3.jpg"

        p1.write_text("photo 1 content")
        p2.write_text("photo 2 content")
        p3.write_text("photo 3 content")

        assignments = {
            p1: {"Alice"},
            p2: {"Alice", "Bob"},  # Multi-person photo!
            p3: set(),            # Unmatched / faceless
        }

        organizer = PhotoOrganizer(output_dir=out_dir, action=FileAction.COPY)
        stats = organizer.organize_photos(assignments)

        assert stats["Alice"] == 2
        assert stats["Bob"] == 1
        assert stats["unknown_or_no_face"] == 1

        assert (out_dir / "Alice" / "photo1.jpg").exists()
        assert (out_dir / "Alice" / "photo2.jpg").exists()
        assert (out_dir / "Bob" / "photo2.jpg").exists()
        assert (out_dir / "unknown_or_no_face" / "photo3.jpg").exists()

        print("SUCCESS: PhotoOrganizer unit test passed successfully.")


if __name__ == "__main__":
    test_organizer()
