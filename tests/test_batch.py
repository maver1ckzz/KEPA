import json
import tempfile
import unittest
from pathlib import Path

from kepa.execution.batch import (
    load_batch_manifest,
    make_paired_manifest,
    parse_gpu_ids,
    start_backend_with_gpu_fallback,
)


class BatchManifestTests(unittest.TestCase):
    def test_named_manifest_resolves_relative_and_axial_list_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("sag.nii.gz", "ax1.nii.gz", "ax2.nii.gz"):
                (root / name).touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({
                "manifest_version": "1.0",
                "cases": [{
                    "case_id": "case-1",
                    "inputs": {"sag": "sag.nii.gz", "axl": ["ax1.nii.gz", "ax2.nii.gz"]},
                }],
            }), encoding="utf-8")
            case = load_batch_manifest(manifest)[0]
            self.assertEqual(case.case_id, "case-1")
            self.assertEqual(case.sagittal_nii, (root / "sag.nii.gz").resolve())
            self.assertEqual(case.axial_niis, ((root / "ax1.nii.gz").resolve(), (root / "ax2.nii.gz").resolve()))

    def test_legacy_manifest_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sag.nii.gz").touch()
            (root / "ax.nii.gz").touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps([{
                "case_id": "old", "sag_nii": "sag.nii.gz", "ax_nii": "ax.nii.gz",
            }]), encoding="utf-8")
            self.assertEqual(len(load_batch_manifest(manifest)), 1)

    def test_manifest_rejects_duplicate_ids_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sag.nii.gz").touch()
            (root / "ax.nii.gz").touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({"cases": [
                {"case_id": "dup", "inputs": {"sag": "sag.nii.gz", "axl": "ax.nii.gz"}},
                {"case_id": "dup", "inputs": {"sag": "sag.nii.gz", "axl": "ax.nii.gz"}},
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                load_batch_manifest(manifest)

    def test_gpu_ids_are_ordered_and_unique(self):
        self.assertEqual(parse_gpu_ids("3, 5,6"), ("3", "5", "6"))
        with self.assertRaises(ValueError):
            parse_gpu_ids("3,3")

    def test_backend_falls_back_in_user_order(self):
        attempted = []

        class Backend:
            def __init__(self, gpu):
                self.gpu = gpu

            def start(self):
                attempted.append(self.gpu)
                if self.gpu == "3":
                    raise RuntimeError("not enough GPU memory")

            def close(self):
                pass

        backend, selected = start_backend_with_gpu_fallback(Backend, ("3", "4"))
        self.assertEqual(attempted, ["3", "4"])
        self.assertEqual(backend.gpu, "4")
        self.assertEqual(selected, "4")



    def test_directory_scan_generates_paired_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sag_dir, axial_root = root / "sag", root / "axial"
            sag_dir.mkdir()
            (axial_root / "case-1").mkdir(parents=True)
            (sag_dir / "case-1.nii.gz").touch()
            (axial_root / "case-1" / "part-1.nii.gz").touch()
            (axial_root / "case-1" / "part-2.nii.gz").touch()
            output = make_paired_manifest(sag_dir, axial_root, root / "manifest.json")
            cases = load_batch_manifest(output)
            self.assertEqual(cases[0].case_id, "case-1")
            self.assertEqual(len(cases[0].axial_niis), 2)
