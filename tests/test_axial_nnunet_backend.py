"""Weight-free contract tests for the two axial nnU-Net models."""

import unittest
import json
from pathlib import Path

import numpy as np
import nibabel as nib
from types import SimpleNamespace
from unittest.mock import patch

from kepa.tools.segmentation.axial_nnunet import AxialNnunetBackend, _largest_component, _predict_native_labels
from kepa.tools.segmentation.contract import SegmentationRequest
from kepa.tools.segmentation.factory import backend_specs, create_backend
from kepa.workflow.validator import validate_workflow


class AxialNnunetBackendTests(unittest.TestCase):
    def test_registry_declares_two_distinct_anatomical_outputs(self):
        spec = backend_specs()["axial_nnunet_python"]
        self.assertEqual(spec.views, frozenset({"axial_t2"}))
        self.assertEqual(spec.models["canal_seg"], frozenset({"spinal_canal"}))
        self.assertEqual(spec.models["sc_epi"], frozenset({"spinal_cord"}))
        backend = create_backend("axial_nnunet_python", data_dir=Path("model-data"), device="cpu")
        self.assertIsInstance(backend, AxialNnunetBackend)

    def test_request_rejects_wrong_view_before_loading(self):
        backend = AxialNnunetBackend(Path("model-data"), device="cpu")
        request = SegmentationRequest(
            input_image=Path("not-needed.nii.gz"), output_dir=Path("unused"),
            view="sagittal_t2", model_id="canal_seg", structures=("spinal_canal",),
        )
        with self.assertRaisesRegex(ValueError, "does not support view"):
            backend.segment(request)

    def test_largest_component_discards_isolated_false_positives(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[6, 6, 6] = True
        result = _largest_component(mask)
        self.assertEqual(int(result.sum()), 8)
        self.assertFalse(result[6, 6, 6])

    def test_serial_worker_only(self):
        with self.assertRaisesRegex(ValueError, "max_workers=1"):
            AxialNnunetBackend(Path("model-data"), max_workers=2)

    def test_native_orientation_roundtrip_for_both_axial_models(self):
        class FakePredictor:
            def __init__(self, orientation):
                self.dataset_json = {} if orientation is None else {"image_orientation": orientation}
                self.plans_manager = SimpleNamespace(dataset_name="Dataset001_Test")

        data = np.zeros((4, 5, 6), dtype=np.float32)
        data[1, 2, 3] = 1
        # Input has nibabel LAS axes; canal's SCT LAS means nibabel RPI,
        # while EPISeg's default SCT LPI means nibabel RAS.
        affine = np.diag([-0.5, 0.6, 3.0, 1.0])
        image = nib.Nifti1Image(data, affine)
        for orientation in ("LAS", None):
            predictor = FakePredictor(orientation)
            captured = {}

            def fake_predict(model_array, spacing, _predictor, _stage_seconds):
                captured["shape"] = model_array.shape
                captured["spacing"] = spacing
                return (model_array[0] > 0).astype(np.uint8)

            with patch("kepa.tools.segmentation.axial_nnunet._predict_single_npy_array_timed", fake_predict):
                output = _predict_native_labels(image, predictor)
            self.assertEqual(output.shape, data.shape)
            self.assertTrue(np.array_equal(output, data))
            self.assertEqual(len(captured["shape"]), 4)
            self.assertEqual(captured["shape"][0], 1)
            np.testing.assert_allclose(captured["spacing"], (3.0, 0.6, 0.5), rtol=1e-6)

    def test_two_axial_masks_can_be_declared_as_supporting_artifacts(self):
        path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cervical_scs_paired_with_axial_masks.json"
        workflow = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(validate_workflow(workflow), [])


if __name__ == "__main__":
    unittest.main()
