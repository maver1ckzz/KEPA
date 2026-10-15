import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from kepa.execution.single_nifti import build_single_nifti_workflow


class SingleNiftiWorkflowTests(unittest.TestCase):
    def test_full_workflow_reuses_segmentation_and_exports_both_views(self):
        repo = Path.cwd()
        workflow = repo / "tests" / "fixtures" / "single_nifti_dual_view.json"
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            affine = np.eye(4)
            intensities = np.arange(16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
            input_path = root / "input.nii.gz"
            nib.save(nib.Nifti1Image(intensities, affine), str(input_path))

            labels = np.zeros((16, 16, 16), dtype=np.int16)
            labels[3:6, 8:10, 8:12] = 2  # Canal: sagittal maximum-area reference.
            labels[9:11, 8:10, 9:11] = 64  # C3-C4 disc used for sag2tra.
            segmentation_path = root / "mock_totalspineseg_all.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), str(segmentation_path))

            calls = []

            def fake_segmentation_runner(input_nii, output_dir):
                calls.append((input_nii, output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                return segmentation_path, ["mock-sct", "totalspineseg"]

            output_dir = root / "run"
            manifest = build_single_nifti_workflow(
                workflow,
                input_path,
                "C3-4",
                output_dir,
                segmentation_runner=fake_segmentation_runner,
            )

            self.assertEqual(len(calls), 1, "same-volume TotalSpineSeg should be cached/reused")
            self.assertEqual({item["view"] for item in manifest["evidence"]}, {"sagittal", "axial"})
            self.assertTrue((output_dir / "evidence.json").is_file())
            for item in manifest["evidence"]:
                self.assertTrue((output_dir / item["image"]).is_file())
            saved = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["query"]["level"], "C3-4")


if __name__ == "__main__":
    unittest.main()
