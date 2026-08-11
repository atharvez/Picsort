"""
File organizer for copying, moving, or symlinking sorted photos into destination person directories.
Handles photos with multiple detected people.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Set
from picsort.config import FileAction

logger = logging.getLogger("picsort")


class PhotoOrganizer:
    """Organizes photo files into person-specific output directories."""

    def __init__(
        self,
        output_dir: Path,
        action: FileAction = FileAction.COPY,
        unknown_folder_name: str = "unknown_or_no_face",
    ):
        self.output_dir = output_dir
        self.action = action
        self.unknown_folder_name = unknown_folder_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_unique_destination(self, dest_folder: Path, src_file: Path) -> Path:
        """Get destination path, appending index if filename already exists."""
        dest_file = dest_folder / src_file.name
        if not dest_file.exists():
            return dest_file

        # Check if it's the exact same file (same path or same inode)
        if dest_file.resolve() == src_file.resolve():
            return dest_file

        base = src_file.stem
        ext = src_file.suffix
        counter = 1
        while dest_file.exists():
            dest_file = dest_folder / f"{base}_{counter}{ext}"
            counter += 1
        return dest_file

    def organize_photos(
        self, photo_assignments: Dict[Path, Set[str]]
    ) -> Dict[str, int]:
        """
        Organize photos into target person folders according to specified action.

        Args:
            photo_assignments: Dict mapping photo_file_path -> set of person names.
                             If set is empty or contains None, photo is placed in unknown_folder_name.

        Returns:
            Dict mapping person_name -> count of photos placed in their folder.
        """
        stats: Dict[str, int] = {}

        for src_path, person_tags in photo_assignments.items():
            if not src_path.exists():
                logger.warning(f"Source file '{src_path}' no longer exists; skipping.")
                continue

            # Determine targets
            targets = {t for t in person_tags if t}
            if not targets:
                targets = {self.unknown_folder_name}

            # Convert set to list to handle multi-folder placement
            target_list = sorted(list(targets))

            for idx, person_name in enumerate(target_list):
                dest_dir = self.output_dir / person_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = self._get_unique_destination(dest_dir, src_path)

                stats[person_name] = stats.get(person_name, 0) + 1

                # Decide operation: if action is MOVE, only move on the final target;
                # copy on all preceding targets so multi-person photos aren't lost!
                current_action = self.action
                if self.action == FileAction.MOVE and idx < len(target_list) - 1:
                    current_action = FileAction.COPY

                self._perform_file_action(src_path, dest_file, current_action)

        return stats

    def _perform_file_action(self, src: Path, dst: Path, action: FileAction):
        """Execute copy, move, or symlink for a single file pair."""
        if src.resolve() == dst.resolve():
            return

        try:
            if action == FileAction.COPY:
                shutil.copy2(src, dst)
            elif action == FileAction.MOVE:
                shutil.move(src, dst)
            elif action == FileAction.SYMLINK:
                try:
                    # Windows symlinks may require developer mode or admin privileges
                    os.symlink(src, dst)
                except OSError as err:
                    logger.warning(
                        f"Symlink failed for '{src.name}' ({err}). Falling back to copying file."
                    )
                    shutil.copy2(src, dst)
        except Exception as e:
            logger.error(f"Failed to {action.value} file '{src}' to '{dst}': {e}")
