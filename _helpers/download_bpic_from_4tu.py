"""
Download BPIC 2017 and BPIC 2012 from 4TU.ResearchData.
"""

from __future__ import annotations
import gzip
import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import TypedDict

# Resolve the root filepath.
ROOT = Path(__file__).resolve().parents[1]
CHUNK_SIZE = 1024 * 1024

class DatasetConfig(TypedDict):
    name: str
    doi: str
    url: str
    archive_name: str
    archive_md5: str
    target_dir: Path
    xes_name: str
    xes_md5: str

# Define datasets to download save their MD5 hashes.
DATASETS: list[DatasetConfig] = [
    {
        "name": "BPI Challenge 2017",
        "doi": "10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b",
        "url": "https://data.4tu.nl/file/34c3f44b-3101-4ea9-8281-e38905c68b8d/f3aec4f7-d52c-4217-82f4-57d719a8298c",
        "archive_name": "BPI Challenge 2017.xes.gz",
        "archive_md5": "10b37a2f78e870d78406198403ff13d2",
        "target_dir": ROOT / "E_main_BPIC_2017" / "BPI Challenge 2017",
        "xes_name": "BPI Challenge 2017.xes",
        "xes_md5": "3b8eefc5a5981c48451af0513e1669d3",
    },
    {
        "name": "BPI Challenge 2012",
        "doi": "10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
        "url": "https://data.4tu.nl/file/533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893",
        "archive_name": "BPI_Challenge_2012.xes.gz",
        "archive_md5": "74c7ba9aba85bfcb181a22c9d565e5b5",
        "target_dir": ROOT / "E_ablation_BPIC_2012" / "BPI Challenge 2012",
        "xes_name": "BPI_Challenge_2012.xes",
        "xes_md5": "b815ef03ebae63407bc09de191a748f6",
    },
]

# Configure logging.
logger = logging.getLogger("download_bpic_from_4tu")

# Return the hexadecimal MD5 checksum for a file.
def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""): digest.update(chunk)
    return digest.hexdigest()

# Validate a file checksum and fail loudly on mismatch.
def assert_md5(path: Path, expected: str) -> None:
    actual = md5sum(path)
    if actual != expected: raise ValueError(f"MD5 mismatch for {path}: expected {expected}, got {actual}")

# Validate a produced file and remove it on mismatch, so the next invocation starts from a clean state.
def verify_or_remove(path: Path, expected: str) -> None:
    try:
        assert_md5(path, expected)
    except ValueError:
        path.unlink(missing_ok=True)
        raise

# Download a file to a temporary path before replacing the target.
def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fl-ppm-thesis-pipeline/1.0"})

    # The "finally" block removes the temporary after the atomic "replace" and after any aborted transfer alike.
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk: break
                handle.write(chunk)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)

# Unpack a gzip archive into the expected XES file.
def unpack_gzip(archive_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".part")

    # The "finally" block removes the temporary after the atomic "replace" and after any aborted extraction alike.
    try:
        with gzip.open(archive_path, "rb") as source, temporary.open("wb") as target:
            while True:
                chunk = source.read(CHUNK_SIZE)
                if not chunk: break
                target.write(chunk)
        temporary.replace(target_path)
    finally:
        temporary.unlink(missing_ok=True)

# Download, verify and unpack one 4TU event log.
def prepare_dataset(config: DatasetConfig) -> None:
    target_dir = config["target_dir"]
    archive_path = target_dir / config["archive_name"]
    xes_path = target_dir / config["xes_name"]

    # Check if the dataset already exists.
    if xes_path.exists():
        assert_md5(xes_path, config["xes_md5"])
        logger.info("%s already exists: %s", config["name"], xes_path)
        return

    # Download and unpack the dataset when the XES file is missing.
    if not archive_path.exists():
        logger.info("Downloading %s from 4TU.ResearchData", config["name"])
        download_file(config["url"], archive_path)

    verify_or_remove(archive_path, config["archive_md5"])
    unpack_gzip(archive_path, xes_path)
    verify_or_remove(xes_path, config["xes_md5"])
    logger.info("Prepared %s: %s", config["name"], xes_path)

# Prepare all source event logs required by the pipeline.
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for config in DATASETS: prepare_dataset(config)

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb  |  Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────