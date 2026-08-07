from __future__ import annotations
import gzip
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

# HELPER: Load the downloader by path, because the helpers folder is not an importable package.
def _load_downloader() -> Any:
    spec = importlib.util.spec_from_file_location(
        "download_bpic_from_4tu", REPO_ROOT / "_helpers" / "download_bpic_from_4tu.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

downloader = _load_downloader()

# HELPER: Build one gzip archive and return the payload with both checksums.
def _sample_archive(root: Path) -> tuple[Path, bytes, str, str]:
    payload = b"<log>synthetic event log payload</log>\n" * 200
    archive = root / "sample.xes.gz"
    archive.write_bytes(gzip.compress(payload, mtime=0))
    return archive, payload, hashlib.md5(archive.read_bytes()).hexdigest(), hashlib.md5(payload).hexdigest()

# HELPER: Build one dataset configuration pointing at a local archive instead of 4TU.
def _config(root: Path, archive: Path, archive_md5: str, xes_md5: str) -> dict[str, Any]:
    return {
        "name": "SAMPLE", "doi": "n/a", "url": archive.as_uri(),
        "archive_name": "sample.xes.gz", "archive_md5": archive_md5,
        "target_dir": root / "target", "xes_name": "sample.xes", "xes_md5": xes_md5,
    }

class DownloadHelperTests(unittest.TestCase):
    # Verify the downloader resolves its targets where the preprocessing scripts read them.
    def test_target_paths_match_where_preprocessing_looks(self) -> None:
        by_name = {config["name"]: config for config in downloader.DATASETS}

        self.assertEqual(by_name["BPI Challenge 2017"]["target_dir"] / by_name["BPI Challenge 2017"]["xes_name"],
                         REPO_ROOT / "E_main_BPIC_2017" / "BPI Challenge 2017" / "BPI Challenge 2017.xes")
        self.assertEqual(by_name["BPI Challenge 2012"]["target_dir"] / by_name["BPI Challenge 2012"]["xes_name"],
                         REPO_ROOT / "E_ablation_BPIC_2012" / "BPI Challenge 2012" / "BPI_Challenge_2012.xes")

    # Verify a complete run creates the folder, extracts the log and is idempotent on a second call.
    def test_successful_run_creates_the_folder_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, payload, archive_md5, xes_md5 = _sample_archive(root)
            config = _config(root, archive, archive_md5, xes_md5)
            self.assertFalse(config["target_dir"].exists())

            downloader.prepare_dataset(config)
            extracted = config["target_dir"] / config["xes_name"]
            self.assertEqual(extracted.read_bytes(), payload)

            # A second call verifies the checksum and returns without downloading again.
            downloader.prepare_dataset(config)
            self.assertEqual(extracted.read_bytes(), payload)

    # Verify an unreachable source leaves no partial file and no unverified artifact behind.
    def test_unreachable_source_leaves_nothing_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _payload, archive_md5, xes_md5 = _sample_archive(root)
            config = _config(root, root / "missing.xes.gz", archive_md5, xes_md5)

            with self.assertRaises(Exception):
                downloader.prepare_dataset(config)

            leftovers = sorted(path.name for path in config["target_dir"].iterdir())
            self.assertEqual(leftovers, [], f"failed download left {leftovers} behind")

    # Verify a corrupt download is rejected and never lands where the pipeline would read it.
    def test_corrupt_download_is_rejected_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _payload, _archive_md5, xes_md5 = _sample_archive(root)
            config = _config(root, archive, "0" * 32, xes_md5)

            with self.assertRaises(ValueError):
                downloader.prepare_dataset(config)

            self.assertFalse((config["target_dir"] / config["archive_name"]).exists())
            self.assertFalse((config["target_dir"] / config["xes_name"]).exists())
            self.assertEqual(sorted(path.name for path in config["target_dir"].iterdir()), [])

    # Verify a pre-existing archive whose checksum no longer matches stops the run before unpacking.
    def test_existing_archive_with_a_wrong_checksum_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _payload, archive_md5, xes_md5 = _sample_archive(root)
            config = _config(root, archive, archive_md5, xes_md5)
            config["target_dir"].mkdir(parents=True)
            (config["target_dir"] / config["archive_name"]).write_bytes(b"not the archive we recorded")

            with self.assertRaises(ValueError):
                downloader.prepare_dataset(config)

            self.assertFalse((config["target_dir"] / config["xes_name"]).exists())

    # Verify a corrupt extracted log is rejected instead of being handed to preprocessing.
    def test_corrupt_extracted_log_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, _payload, archive_md5, _xes_md5 = _sample_archive(root)
            config = _config(root, archive, archive_md5, "0" * 32)

            with self.assertRaises(ValueError):
                downloader.prepare_dataset(config)

            self.assertFalse((config["target_dir"] / config["xes_name"]).exists())
            self.assertFalse((config["target_dir"] / (config["xes_name"] + ".part")).exists())

if __name__ == "__main__":
    unittest.main()