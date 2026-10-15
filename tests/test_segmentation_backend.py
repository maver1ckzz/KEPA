import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from kepa.tools.segmentation import SegmentationGeometry, SegmentationRequest, TotalSpineSegCliBackend
from kepa.tools.segmentation.totalspineseg_python import _serial_upstream_helpers


class SegmentationBackendTests(unittest.TestCase):
    def test_totalspineseg_adapter_runs_one_sagittal_case_with_installed_weights(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            input_path = root / "case_sag.nii.gz"
            input_path.write_bytes(b"input")
            data_dir = root / "weights"
            for dataset in ("Dataset101_TotalSpineSeg_step1", "Dataset102_TotalSpineSeg_step2"):
                fold = data_dir / "nnUNet" / "results" / dataset / "trainer" / "fold_0"
                fold.mkdir(parents=True)
                (fold / "checkpoint_best.pth").write_bytes(b"weights")

            backend = TotalSpineSegCliBackend("tss-mock", data_dir)
            request = SegmentationRequest(
                input_image=input_path,
                output_dir=root / "step-segmentation",
                view="sagittal_t2",
                structures=("vertebrae", "spinal_canal"),
                model_id="totalspineseg",
                case_id="case-001",
            )

            def fake_run(command, **kwargs):
                staged_data = Path(command[command.index("--data-dir") + 1])
                staged_results = staged_data / "nnUNet" / "results"
                self.assertTrue((staged_results / "Dataset101_TotalSpineSeg_step1").is_dir())
                self.assertTrue((staged_results / "Dataset102_TotalSpineSeg_step2").is_dir())
                output = request.output_dir / "step2_output"
                output.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.int16), np.eye(4)),
                    str(output / "case_sag_step2_output.nii.gz"),
                )

                class Completed:
                    returncode = 0
                    stdout = "ok"
                    stderr = ""

                return Completed()

            with patch("kepa.tools.segmentation.totalspineseg.shutil.which", return_value="/fake/totalspineseg"), patch(
                "kepa.tools.segmentation.totalspineseg.subprocess.run", side_effect=fake_run
            ) as run:
                with backend:
                    result = backend.segment(request)
                self.assertEqual(result.label_map.name, "case_sag_step2_output.nii.gz")
                self.assertEqual(result.label_map_semantics[2], "spinal_canal")
                self.assertIn("--data-dir", run.call_args.args[0])
                self.assertEqual(backend._started, False)

    def test_backend_rejects_non_sagittal_view_before_inference(self):
        backend = TotalSpineSegCliBackend()
        request = SegmentationRequest(
            input_image=Path("not-needed.nii.gz"),
            output_dir=Path("unused-output"),
            view="axial_t2",
            structures=("spinal_canal",),
            model_id="totalspineseg",
        )
        with self.assertRaisesRegex(ValueError, "does not support view"):
            backend.segment(request)

    def test_geometry_rejects_invalid_affine(self):
        with self.assertRaisesRegex(ValueError, "4x4"):
            SegmentationGeometry((1, 2, 3), ((1.0, 0.0),), "NIFTI_RAS+")

    def test_backend_requires_preinstalled_weights_and_does_not_infer_download(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            backend = TotalSpineSegCliBackend("tss-mock", Path(temporary))
            with patch("kepa.tools.segmentation.totalspineseg.shutil.which", return_value="/fake/totalspineseg"):
                with self.assertRaisesRegex(FileNotFoundError, "checkpoints are incomplete"):
                    backend.start()

    def test_serial_helper_patch_is_temporary_even_on_failure(self):
        helper_module = types.ModuleType("fake_tss_helper")
        original_map = lambda function, values, **kwargs: [function(value) for value in values]
        helper_module.process_map = original_map
        exec("def transform_mp(values):\n    return process_map(lambda value: value + 1, values)\n", helper_module.__dict__)
        upstream = types.SimpleNamespace(transform_mp=helper_module.transform_mp)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with _serial_upstream_helpers(upstream) as count:
                self.assertEqual(count, 1)
                self.assertEqual(helper_module.transform_mp([1, 2]), [2, 3])
                raise RuntimeError("test failure")
        self.assertIs(helper_module.process_map, original_map)


if __name__ == "__main__":
    unittest.main()
