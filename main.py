"""
PicSort - Local, Offline Face Recognition Photo Sorter CLI Entrypoint.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from tqdm import tqdm

from picsort.config import AppConfig, SortMode, FileAction, DetectorEngine
from picsort.utils import setup_logger, scan_image_files, calculate_file_hash, load_image_bgr
from picsort.cache import CacheManager
from picsort.detectors import get_detector_engine
from picsort.matcher import KnownFaceMatcher
from picsort.clusterer import FaceClusterer
from picsort.organizer import PhotoOrganizer
from picsort.reporter import ExecutionReporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PicSort: Local, offline photo organizer powered by face recognition."
    )
    parser.add_argument(
        "-s", "--source", required=True, type=Path, help="Path to source photos folder"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Path to output sorted folder"
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["known", "cluster"],
        default="known",
        help="Sorting mode: 'known' (match against reference photos) or 'cluster' (auto-cluster unidentified faces)",
    )
    parser.add_argument(
        "-p",
        "--people-dir",
        type=Path,
        default=None,
        help="Directory containing reference photos in /people/<name>/ folders (used in 'known' mode)",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        choices=["copy", "move", "symlink"],
        default="copy",
        help="File action: 'copy', 'move', or 'symlink'",
    )
    parser.add_argument(
        "-d",
        "--detector",
        type=str,
        choices=["insightface", "dlib"],
        default="insightface",
        help="Face detection engine: 'insightface' (preferred) or 'dlib' (face_recognition fallback)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Cosine similarity threshold for known face matching (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.4,
        help="DBSCAN eps cosine distance parameter for auto-clustering (default: 0.4)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples parameter for auto-clustering (default: 2)",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Bypass cache and force reprocessing of all photos",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration and run strictly on CPU",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to output execution summary JSON report",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose debug logging"
    )

    return parser.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    logger = setup_logger(args.verbose)

    source_dir: Path = args.source.resolve()
    output_dir: Path = args.output.resolve()

    if not source_dir.exists():
        logger.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    # Initialize Config
    config = AppConfig(
        source_dir=source_dir,
        output_dir=output_dir,
        mode=SortMode(args.mode),
        people_dir=args.people_dir.resolve() if args.people_dir else (source_dir / "people"),
        action=FileAction(args.action),
        detector=DetectorEngine(args.detector),
        similarity_threshold=args.threshold,
        cluster_eps=args.eps,
        cluster_min_samples=args.min_samples,
        force_reprocess=args.force_reprocess,
    )

    logger.info("Initializing PicSort...")
    logger.info(f"Mode: {config.mode.value.upper()} | Action: {config.action.value.upper()} | Engine: {config.detector.value}")

    # Initialize Cache
    cache = CacheManager(output_dir / ".picsort_cache.db")

    # Scan Source Photos
    image_paths = scan_image_files(config.source_dir, config.image_extensions)
    if not image_paths:
        logger.warning(f"No supported image files found in '{config.source_dir}'.")
        sys.exit(0)

    logger.info(f"Scanned {len(image_paths)} image file(s) in source folder.")

    # Instantiate Face Detector Engine
    use_gpu = not args.no_gpu
    try:
        detector = get_detector_engine(config.detector, use_gpu=use_gpu)
    except Exception as err:
        logger.error(f"Failed to initialize detector engine '{config.detector.value}': {err}")
        sys.exit(1)

    # Storage for photo face extraction results
    # Dict[Path, List[Dict[str, Any]]] mapping photo path -> list of face dicts
    photo_faces_map: Dict[Path, List[Dict[str, Any]]] = {}
    cached_count = 0
    processed_count = 0
    failed_count = 0
    total_faces_count = 0

    logger.info("Extracting faces and embeddings...")
    for img_path in tqdm(image_paths, desc="Processing Photos", unit="img"):
        try:
            file_hash = calculate_file_hash(img_path)
            
            cached_faces = None
            if not config.force_reprocess:
                cached_faces = cache.get_cached_faces(file_hash)

            if cached_faces is not None:
                faces = cached_faces
                cached_count += 1
            else:
                img_bgr, err = load_image_bgr(img_path)
                if err or img_bgr is None:
                    logger.warning(f"Skipping unreadable photo '{img_path.name}': {err}")
                    failed_count += 1
                    faces = []
                else:
                    faces = detector.detect_faces(img_bgr)
                    cache.save_faces(file_hash, str(img_path), faces)
                    processed_count += 1

            photo_faces_map[img_path] = faces
            total_faces_count += len(faces)

        except Exception as e:
            logger.error(f"Error processing image '{img_path.name}': {e}")
            failed_count += 1
            photo_faces_map[img_path] = []

    # Map photo_path -> set of target person folder names
    photo_assignments: Dict[Path, Set[str]] = {p: set() for p in image_paths}

    if config.mode == SortMode.KNOWN:
        matcher = KnownFaceMatcher(
            detector=detector, similarity_threshold=config.similarity_threshold
        )
        if config.people_dir and config.people_dir.exists():
            matcher.load_reference_people(config.people_dir, config.image_extensions)
        else:
            logger.warning(
                f"Reference people directory '{config.people_dir}' not found. All photos will be categorized as '{config.unknown_folder_name}'."
            )

        for img_path, faces in photo_faces_map.items():
            if not faces:
                photo_assignments[img_path].add(config.unknown_folder_name)
                continue

            matched_any = False
            for f in faces:
                person, score = matcher.match_face(f["embedding"])
                if person:
                    photo_assignments[img_path].add(person)
                    matched_any = True

            if not matched_any:
                photo_assignments[img_path].add(config.unknown_folder_name)

    elif config.mode == SortMode.CLUSTER:
        # Prepare list of all face records across the photo library
        all_face_records = []
        for img_path, faces in photo_faces_map.items():
            for face in faces:
                all_face_records.append(
                    {
                        "photo_path": img_path,
                        "embedding": face["embedding"],
                    }
                )

        clusterer = FaceClusterer(
            eps=config.cluster_eps, min_samples=config.cluster_min_samples
        )
        _, updated_records = clusterer.cluster_faces(all_face_records)

        # Map cluster assignments back to original photos
        for rec in updated_records:
            img_path = rec["photo_path"]
            person = rec.get("assigned_person")
            if person:
                photo_assignments[img_path].add(person)

        # Handle photos without any assigned cluster
        for img_path in photo_faces_map.keys():
            if not photo_assignments[img_path]:
                photo_assignments[img_path].add(config.unknown_folder_name)

    # Perform File Organization
    logger.info(f"Organizing photos into output directory '{config.output_dir}'...")
    organizer = PhotoOrganizer(
        output_dir=config.output_dir,
        action=config.action,
        unknown_folder_name=config.unknown_folder_name,
    )
    person_counts = organizer.organize_photos(photo_assignments)

    elapsed = time.time() - start_time

    # Display Report
    ExecutionReporter.print_summary(
        total_photos=len(image_paths),
        processed_photos=processed_count,
        cached_photos=cached_count,
        failed_photos=failed_count,
        total_faces=total_faces_count,
        person_counts=person_counts,
        elapsed_seconds=elapsed,
    )

    if args.report_json:
        report_data = {
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "mode": config.mode.value,
            "action": config.action.value,
            "detector": config.detector.value,
            "total_photos": len(image_paths),
            "processed_photos": processed_count,
            "cached_photos": cached_count,
            "failed_photos": failed_count,
            "total_faces": total_faces_count,
            "person_counts": person_counts,
            "elapsed_seconds": round(elapsed, 2),
        }
        ExecutionReporter.save_report_json(args.report_json.resolve(), report_data)


if __name__ == "__main__":
    main()
