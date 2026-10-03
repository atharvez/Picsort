# PicSort - Local, Offline Face Recognition Photo Sorter

**PicSort** is a high-performance, 100% local, offline Python tool that scans photo libraries and organizes images into person-specific folders using state-of-the-art face detection and embedding recognition.

---

## Key Features

- **100% Local & Offline**: No cloud APIs, no network requests, and zero data leaving your machine.
- **Dual Detection Engines**:
  - **InsightFace (Preferred)**: Fast, accurate ONNX Runtime-powered detection with automatic GPU (CUDA) acceleration and CPU fallback.
  - **dlib / `face_recognition`**: Reliable 128D embedding fallback engine.
- **Two Sorting Modes**:
  1. **Known People Mode (`known`)**: Matches face embeddings against reference images provided in a `/people/<name>/` directory structure.
  2. **Auto-Cluster Mode (`cluster`)**: Uses DBSCAN machine learning clustering to automatically group unidentified faces into folders (`person_1`, `person_2`, etc.) so you can review and rename them later.
- **Multi-Person Support**: Photos containing multiple people are filed into each matched person's folder without file duplication when using symlinks.
- **HEIC & EXIF Auto-Rotation**: Full support for `.jpg`, `.jpeg`, `.png`, and iPhone `.heic` photos with proper EXIF orientation handling.
- **Persistent SQLite Caching**: Keyed by SHA-256 file content hash to bypass re-processing unchanged photos on subsequent runs.
- **Flexible File Actions**: Choose between `--action copy`, `move`, or `symlink`.
- **Rich Terminal UI & Summary Reports**: Real-time progress bars and category breakdown tables with optional JSON report export.

---

## Installation & Setup

### 1. Prerequisites
- **Python**: 3.9, 3.10, 3.11, or 3.12.
- *(Optional for GPU acceleration)*: NVIDIA CUDA Toolkit & cuDNN installed on your system.

### 2. Basic Installation (InsightFace Engine)
Clone or download the repository, then install core dependencies:

```bash
git clone https://github.com/your-repo/picsort.git
cd picsort

pip install -r requirements.txt
```

### 3. Installing Dependencies & Troubleshooting

#### InsightFace & ONNX Runtime (Default & Recommended)
- **CPU (Default)**: Included in `requirements.txt`.
- **GPU (CUDA)**: If you have an NVIDIA GPU, install `onnxruntime-gpu`:
  ```bash
  pip install onnxruntime-gpu
  ```

#### dlib / `face_recognition` (Optional Fallback Engine)
To use the `--detector dlib` fallback option:
- **Windows**: Installing dlib requires CMake and Visual Studio C++ Build Tools:
  ```bash
  pip install cmake
  pip install dlib
  pip install face_recognition
  ```
- **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev
  pip install face_recognition
  ```
- **macOS**:
  ```bash
  brew install cmake
  pip install face_recognition
  ```

---

## Usage Examples

### Mode 1: Known People Mode

1. Prepare your reference photos directory structure:
   ```
   people/
   |-- Alice/
   |   |-- alice_ref1.jpg
   |   \-- alice_ref2.jpg
   |-- Bob/
   |   \-- bob_ref.jpg
   \-- Charlie/
       \-- charlie.png
   ```

2. Run PicSort in **known** mode:
   ```bash
   python main.py --source /path/to/my_photos --output /path/to/sorted_photos --mode known --people-dir /path/to/people --action copy
   ```

### Mode 2: Auto-Cluster Mode

If you don't have reference photos, PicSort will cluster faces automatically:

```bash
python main.py --source /path/to/my_photos --output /path/to/clustered_photos --mode cluster --action symlink
```

This will output folders named `person_1`, `person_2`, ..., and `unknown_or_no_face`. After reviewing, simply rename `person_1` to the actual person's name!

---

## Command Line Arguments Reference

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `-s` | *Required* | Path to the source photos directory. |
| `--output` | `-o` | *Required* | Path to the output directory where sorted folders will be created. |
| `--mode` | `-m` | `known` | Processing mode: `known` or `cluster`. |
| `--people-dir` | `-p` | `<source>/people` | Directory containing reference photos for known mode (`/people/<name>/`). |
| `--action` | `-a` | `copy` | Organization action: `copy`, `move`, or `symlink`. |
| `--detector` | `-d` | `insightface` | Face detection engine: `insightface` or `dlib`. |
| `--threshold` | `-t` | `0.5` | Cosine similarity threshold for matching (0.0 to 1.0). Higher = stricter match. |
| `--eps` | | `0.4` | DBSCAN cosine distance parameter for clustering. Lower = tighter clusters. |
| `--min-samples` | | `2` | Minimum faces required to form a cluster in `cluster` mode. |
| `--force-reprocess` | | `False` | Ignore SQLite cache and re-run face detection on all photos. |
| `--no-gpu` | | `False` | Force CPU execution even if CUDA GPU is available. |
| `--report-json` | | `None` | Path to save an execution summary JSON report (e.g. `report.json`). |
| `--verbose` | `-v` | `False` | Enable detailed debug logging. |

---

## Architecture Overview

```
picsort/
|-- picsort/
|   |-- config.py           # Configuration data structures & Enums
|   |-- utils.py            # HEIC & EXIF image loader, SHA256 hashing, logging
|   |-- cache.py            # SQLite cache for face embeddings and hashes
|   |-- detectors/
|   |   |-- base.py         # Abstract BaseFaceDetector interface
|   |   |-- insightface_engine.py  # InsightFace (CUDA / CPU auto-detection)
|   |   \-- dlib_engine.py         # face_recognition / dlib fallback
|   |-- matcher.py          # Known face matching against /people/<name>/
|   |-- clusterer.py        # DBSCAN clustering for unidentified faces
|   |-- organizer.py        # Copy / Move / Symlink file organizer
|   \-- reporter.py         # Terminal summary table & JSON report export
|-- main.py                 # CLI entrypoint
|-- requirements.txt        # Package dependencies
\-- README.md               # Documentation
```

---

## License

MIT License. Free for personal and commercial use.
