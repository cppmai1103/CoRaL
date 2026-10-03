"""Offline checks for upload packaging and private-repository enforcement."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import upload_to_huggingface as upload


class UploadTests(unittest.TestCase):
    def setUp(self):
        scratch = upload.ROOT / ".scratch/hf_upload_tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_packaging_excludes_hidden_files_and_launchers(self):
        source = self.root / "source"
        for name in ["fil/fil_clean.csv", "fil/original.xlsx", "fil/chart.png",
                     ".cache/huggingface/token.json", ".env", "script_cpu.sh", "__pycache__/module.pyc"]:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        destination = self.root / "bundle"
        upload.copy_artifacts(source, destination, upload.DATA_SUFFIXES)
        self.assertEqual({p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()},
                         {"fil/fil_clean.csv", "fil/original.xlsx", "fil/chart.png"})

    def test_all_destinations_are_private_before_upload(self):
        api = Mock()
        private = {"data": False, "model": True}
        api.repo_info.side_effect = lambda **kw: SimpleNamespace(private=private[kw["repo_id"]])
        api.update_repo_settings.side_effect = lambda **kw: private.update({kw["repo_id"]: kw["private"]})

        def do_upload(**kwargs):
            self.assertTrue(all(private.values()))
            self.assertNotIn("delete_patterns", kwargs)
            return SimpleNamespace(commit_url="https://example.test/commit")

        api.upload_folder.side_effect = do_upload
        upload.upload_bundles([("dataset", "data", self.root), ("model", "model", self.root)], api)
        self.assertEqual(api.upload_folder.call_count, 2)
        api.update_repo_settings.assert_called_once_with(repo_id="data", repo_type="dataset", private=True)
        self.assertTrue(all(c.kwargs["private"] for c in api.create_repo.call_args_list))

    def test_refuses_upload_if_visibility_does_not_change(self):
        api = Mock()
        api.repo_info.return_value = SimpleNamespace(private=False)
        with self.assertRaisesRegex(RuntimeError, "not private"):
            upload.upload_bundles([("dataset", "data", self.root)], api)
        api.upload_folder.assert_not_called()

    def test_privacy_permission_failure_prevents_all_uploads(self):
        api = Mock()
        api.repo_info.side_effect = lambda **kw: SimpleNamespace(private=kw["repo_id"] == "data")
        api.update_repo_settings.side_effect = PermissionError("access denied")
        with self.assertRaises(PermissionError):
            upload.upload_bundles([("dataset", "data", self.root), ("model", "model", self.root)], api)
        api.upload_folder.assert_not_called()

    def test_dry_run_does_not_call_upload(self):
        def prepare(args, directory):
            directory.mkdir(parents=True)
            (directory / "README.md").write_text("fixture")

        with patch.object(upload, "prepare_dataset", side_effect=prepare), \
             patch.object(upload, "prepare_model", side_effect=prepare), \
             patch.object(upload, "upload_bundles") as remote:
            upload.main(["--dry-run", "--staging-dir", str(self.root / "staging")])
        remote.assert_not_called()
        self.assertEqual(len(list(self.root.rglob("upload_manifest.json"))), 2)

    def test_prepared_split_mismatch_is_rejected(self):
        fields = ["doc_id", "language"] + upload.DIMENSIONS + [f"split_{d}" for d in upload.DIMENSIONS]
        documents = []
        # The same ID in different languages is a valid document identity.
        for lang, split in [("fil", "train"), ("vie", "validation"), ("thai", "test")]:
            documents.append({"doc_id": "same-id", "language": lang,
                              **{d: "2.5" for d in upload.DIMENSIONS},
                              **{f"split_{d}": split for d in upload.DIMENSIONS}})
        with (self.root / "document_table.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(documents)
        for dim in upload.DIMENSIONS:
            with (self.root / f"split_manifest_{dim}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["doc_id", "language", "score", "split"])
                writer.writeheader()
                writer.writerows({"doc_id": r["doc_id"], "language": r["language"], "score": r[dim],
                                  "split": r[f"split_{dim}"]} for r in documents)
        self.assertEqual(upload.validate_prepared(self.root)[0], 3)
        path = self.root / "split_manifest_reasoning.csv"
        path.write_text(path.read_text().replace("2.5", "3.0", 1))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            upload.validate_prepared(self.root)


if __name__ == "__main__":
    unittest.main()
