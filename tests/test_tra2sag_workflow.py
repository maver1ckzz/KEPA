import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from kepa.execution.paired_nifti import build_paired_nifti_workflow
from kepa.workflow.validator import validate_workflow


class Tra2SagWorkflowTests(unittest.TestCase):
    def _make_case(self, root):
        sagittal = root / "sag.nii.gz"
        axial = root / "axial_full.nii.gz"
        shape = (32, 32, 12)
        affine = np.eye(4)
        sagittal_data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        axial_data = sagittal_data + 1
        nib.save(nib.Nifti1Image(sagittal_data, affine), sagittal)
        nib.save(nib.Nifti1Image(axial_data, affine), axial)

        config = Path(__file__).parents[1] / "configs" / "workflows" / "rsna_lumbar_nfn_left_dual_view.json"
        document = json.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(validate_workflow(document), [])
        workflow = root / "nfn_workflow.json"
        workflow.write_text(json.dumps(document), encoding="utf-8")

        sagittal_labels = np.zeros(shape, dtype=np.int16)
        sagittal_labels[16, 14:19, 1:11] = 1
        for label_id, z in zip((92, 93, 94, 95, 100), (2, 4, 6, 8, 10)):
            sagittal_labels[13:19, 8:14, z] = label_id

        def fake_segmentation_runner(input_nii, output_dir):
            input_nii = Path(input_nii)
            image = nib.load(str(input_nii))
            if input_nii == sagittal:
                labels = sagittal_labels
            else:
                labels = np.zeros(image.shape, dtype=np.int16)
                labels[12:20, 12:20, :] = 1
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / "mock_labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, image.affine), output)
            return output, ["synthetic-segmentation"]

        return sagittal, axial, workflow, fake_segmentation_runner

    def test_single_axial_workflow_projects_each_requested_point_and_crops_roi(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sagittal, axial, workflow, fake_runner = self._make_case(root)
            run_dir, manifest = build_paired_nifti_workflow(
                workflow, sagittal, axial, root / "runs",
                case_id="synthetic_nfn", run_id="single",
                segmentation_runner=fake_runner,
            )
            self.assertEqual(manifest["status"], "succeeded")
            result_path = run_dir / manifest["result"]["path"]
            result = json.loads(result_path.read_text(encoding="utf-8"))
            rois = result["views"]["sagittal_t2"]
            self.assertEqual(set(rois), {"L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"})
            self.assertTrue(all(roi is not None for roi in rois.values()))
            self.assertTrue(all(roi["bbox_xyxy"] is not None for roi in rois.values()))
            axial_rois = result["views"]["axial_t2"]
            self.assertEqual(set(axial_rois), {"L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"})
            self.assertTrue(all(roi is not None for roi in axial_rois.values()))
            match_step = next(step for step in manifest["steps"] if step["id"] == "match_axial_points_to_sagittal")
            match_artifact = next(item for item in match_step["artifacts"] if item["type"] == "SliceMatchSet")
            matches = json.loads((run_dir / match_artifact["path"]).read_text(encoding="utf-8"))["regions"]
            self.assertTrue(all(item["status"] == "selected" for item in matches.values()))
            self.assertTrue(all(item["details"]["target_point_xy"] for item in matches.values()))

    def test_split_axial_candidates_keep_sagittal_roi_from_winning_axial_plane(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sagittal, full_axial, workflow, fake_runner = self._make_case(root)
            image = nib.load(str(full_axial))
            full_data = np.asarray(image.dataobj)
            split_paths = []
            for part, (start, stop) in enumerate(((0, 6), (6, 12)), 1):
                part_affine = image.affine.copy()
                part_affine[:3, 3] = image.affine[:3, :3] @ np.asarray((0.0, 0.0, float(start)))
                path = root / f"axial_part_{part}.nii.gz"
                nib.save(nib.Nifti1Image(full_data[:, :, start:stop], part_affine), path)
                split_paths.append(path)
            run_dir, manifest = build_paired_nifti_workflow(
                workflow, sagittal, split_paths, root / "runs",
                case_id="synthetic_nfn_split", run_id="split",
                segmentation_runner=fake_runner,
            )
            result = json.loads((run_dir / manifest["result"]["path"]).read_text(encoding="utf-8"))
            sagittal_rois = result["views"]["sagittal_t2"]
            self.assertEqual(set(sagittal_rois), {"L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1"})
            self.assertTrue(all(roi is not None for roi in sagittal_rois.values()))
            axial_rois = result["views"]["axial_t2"]
            self.assertEqual(set(axial_rois), set(sagittal_rois))
            self.assertTrue(all(roi is not None for roi in axial_rois.values()))
            for region, roi in sagittal_rois.items():
                selection = manifest["region_selection"][region]
                self.assertEqual(selection["sagittal_roi_status"], "selected")
                self.assertEqual(selection["sagittal_source_nifti"], selection["source_nifti"])
                self.assertEqual(selection["sagittal_source_axial_slice_ref"], selection["slice_ref"])
                self.assertEqual(selection["sagittal_roi_status"], "selected")


if __name__ == "__main__":
    unittest.main()
