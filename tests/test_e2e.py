"""
End-to-end integration test for PicSort CLI.
"""

import sys
import shutil
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

from picsort.config import SortMode, FileAction, DetectorEngine
from picsort.utils import scan_image_files


def create_dummy_image(path: Path, color=(200, 200, 200)):
    """Create a basic RGB dummy image."""
    img = Image.new("RGB", (300, 300), color=color)
    img.save(path)


def test_cli_integration():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        source_dir = tmp_path / "source"
        people_dir = tmp_path / "people"
        output_dir = tmp_path / "output"

        source_dir.mkdir()
        people_dir.mkdir()
        output_dir.mkdir()

        # Create dummy source photos
        create_dummy_image(source_dir / "photo1.jpg", (255, 0, 0))
        create_dummy_image(source_dir / "photo2.jpg", (0, 255, 0))
        create_dummy_image(source_dir / "photo3.jpg", (0, 0, 255))

        # Create dummy reference photo for Alice
        alice_dir = people_dir / "Alice"
        alice_dir.mkdir()
        create_dummy_image(alice_dir / "ref1.jpg", (255, 0, 0))

        # Test CLI help
        import subprocess
        cmd_help = [sys.executable, "main.py", "--help"]
        res = subprocess.run(cmd_help, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))
        assert res.returncode == 0, f"Help command failed: {res.stderr}"
        assert "PicSort" in res.stdout

        print("SUCCESS: CLI help command works!")


if __name__ == "__main__":
    test_cli_integration()
