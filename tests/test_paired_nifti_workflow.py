import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from kepa.execution.paired_nifti import _public_region_name, _roi_level_name, build_paired_nifti_workflow
from kepa.tools.matching.tra_map import LevelSliceMatch, match_anatomical_levels, match_anatomical_levels_soft_distance
from kepa.tools.segmentation import (
    SegmentationBackend,
    SegmentationGeometry,
    SegmentationRequest,
    SegmentationResult,
)
from kepa.workflow.validator import validate_workflow


class PairedNiftiWorkflowTests(unittest.TestCase):
    def test_slash_region_alias_maps_to_compact_roi_preset_level(self):
        self.assertEqual(_roi_level_name("C3/C4"), "C3-4")
        self.assertEqual(_roi_level_name("L5/S1"), "L5-S1")
        self.assertEqual(_roi_level_name("C3"), "C3")
        self.assertEqual(_public_region_name("C3-4"), "C3/C4")
        self.assertEqual(_public_region_name("L5-S1"), "L5/S1")

    def test_soft_distance_maps_each_labeled_vertebra_and_disc_to_nearest_plane(self):
        workflow = json.loads(
            (Path.cwd() / "tests" / "fixtures" / "cervical_scs_paired_soft_distance.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(validate_workflow(workflow), [])
        sag_affine = np.eye(4)
        sag_affine[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        labels = np.zeros((40, 40, 3), dtype=np.int16)
        for label_id, low in zip(range(13, 18), (8, 14, 20, 26, 32)):
            labels[8:28, low:low + 4, :] = label_id
        for label_id, low in zip(range(64, 68), (12, 18, 24, 30)):
            labels[8:28, low:low + 2, :] = label_id
        axial_affine = np.eye(4)
        matches = match_anatomical_levels_soft_distance(
            labels,
            sag_affine,
            (40, 40, 40),
            axial_affine,
            sagittal_slice_axis=2,
            axial_slice_axis=2,
            sigma_mm=3.0,
        )
        self.assertEqual(len(matches), 9)
        self.assertEqual(matches["C3"].slice_index, 9)
        self.assertEqual(matches["C3/C4"].slice_index, 12)
        self.assertEqual(matches["C7"].slice_index, 33)
        self.assertEqual(matches["C3/C4"].method, "soft_distance")

    def test_missing_requested_ratio_regions_are_explicit_null_matches(self):
        labels = np.zeros((20, 20, 3), dtype=np.int16)
        matches = match_anatomical_levels(
            labels, np.eye(4), (20, 20, 20), np.eye(4),
            sagittal_slice_axis=2, axial_slice_axis=2, reference_sagittal_index=1,
            requested_regions=["C4", "C4/C5"],
        )
        self.assertEqual(list(matches), ["C4", "C4/C5"])
        self.assertTrue(all(match.status == "missing" for match in matches.values()))
        self.assertTrue(all(match.slice_index is None for match in matches.values()))

    def test_ratio_matcher_accepts_requested_vertebra_and_disc_regions(self):
        labels = np.zeros((12, 12, 3), dtype=np.int16)
        matches = match_anatomical_levels(
            labels, np.eye(4), (12, 12, 12), np.eye(4),
            sagittal_slice_axis=2, axial_slice_axis=2, reference_sagittal_index=1,
            requested_regions=["L3", "L3/L4"],
            region_label_ids={"L3": [43], "L4": [44], "L3/L4": [93]},
        )
        self.assertEqual(list(matches), ["L3", "L3/L4"])
        self.assertTrue(all(match.status == "missing" for match in matches.values()))
        self.assertTrue(all(match.slice_index is None for match in matches.values()))

    def test_missing_requested_soft_distance_regions_are_explicit_null_matches(self):
        labels = np.zeros((20, 20, 3), dtype=np.int16)
        region_ids = {"L3": [23], "L3/L4": [81]}
        matches = match_anatomical_levels_soft_distance(
            labels, np.eye(4), (20, 20, 20), np.eye(4),
            sagittal_slice_axis=2, axial_slice_axis=2,
            requested_regions=["L3", "L3/L4"], region_label_ids=region_ids,
        )
        self.assertEqual(list(matches), ["L3", "L3/L4"])
        self.assertTrue(all(match.status == "missing" for match in matches.values()))
        self.assertTrue(all(match.slice_index is None for match in matches.values()))

    def test_current_case_gt_orientation_regression(self):
        fixture_path = Path(__file__).parent / "fixtures" / "paired_sag_ax_gt_regression.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        count = fixture["axial_slice_count"]
        gt_nifti = {level: count - 1 - index for level, index in fixture["dicom_gt_indices"].items()}
        predicted = fixture["predicted_nifti_indices"]
        differences = {level: predicted[level] - gt_nifti[level] for level in predicted}
        self.assertEqual(sum(value == 0 for value in differences.values()), fixture["expected_exact_matches"])
        self.assertAlmostEqual(
            sum(abs(value) for value in differences.values()) / len(differences),
            fixture["expected_mean_absolute_error_slices"],
        )
        self.assertEqual(max(abs(value) for value in differences.values()), fixture["expected_max_absolute_error_slices"])
        soft_predicted = fixture["soft_distance_predicted_nifti_indices"]
        soft_differences = {level: soft_predicted[level] - gt_nifti[level] for level in soft_predicted}
        self.assertEqual(sum(value == 0 for value in soft_differences.values()), fixture["expected_exact_matches"])
        self.assertAlmostEqual(
            sum(abs(value) for value in soft_differences.values()) / len(soft_differences),
            fixture["expected_mean_absolute_error_slices"],
        )

    def test_paired_workflow_validates_and_executes_to_manifest(self):
        repo = Path.cwd()
        workflow = repo / "tests" / "fixtures" / "cervical_scs_paired_series.json"
        document = json.loads(workflow.read_text(encoding="utf-8"))
        self.assertEqual(validate_workflow(document), [])

        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            root = Path(temporary)
            sag_affine = np.eye(4)
            sag_affine[:3, :3] = np.array(
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float
            )  # axis 2 is left-right; axis 1 is superior-inferior
            axial_affine = np.eye(4)
            axial_affine[:3, :3] = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
            )  # axis 2 is superior-inferior
            sag_path, axial_path = root / "sag.nii.gz", root / "ax.nii.gz"
            nib.save(nib.Nifti1Image(np.arange(40 * 40 * 3, dtype=np.float32).reshape(40, 40, 3), sag_affine), sag_path)
            nib.save(nib.Nifti1Image(np.arange(40 * 40 * 40, dtype=np.float32).reshape(40, 40, 40), axial_affine), axial_path)

            labels = np.zeros((40, 40, 3), dtype=np.int16)
            labels[8:28, 8:12, :] = 13
            labels[8:28, 14:18, :] = 14
            labels[8:28, 20:24, :] = 15
            labels[8:28, 26:30, :] = 16
            labels[8:28, 32:36, :] = 17
            labels[4:9, 4:7, 0] = 1
            labels[4:9, 4:7, 1] = 1
            labels[4:9, 4:7, 2] = 1
            labels[12:20, 4:9, 1] = 2
            labels_path = root / "labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, sag_affine), labels_path)

            class MockBackend(SegmentationBackend):
                backend_id = "mock"
                supported_views = frozenset({"sagittal_t2"})
                supported_models = frozenset({"totalspineseg"})

                def __init__(self):
                    super().__init__()
                    self.starts = self.calls = self.closes = 0

                def _start(self):
                    self.starts += 1

                def _segment(self, request: SegmentationRequest):
                    self.calls += 1
                    output = request.output_dir / "labels.nii.gz"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    nib.save(nib.Nifti1Image(labels, sag_affine), str(output))
                    return SegmentationResult(
                        label_map=output,
                        view=request.view,
                        model_id=request.model_id,
                        backend_id=self.backend_id,
                        backend_version="test",
                        geometry=SegmentationGeometry(
                            tuple(labels.shape),
                            tuple(tuple(float(value) for value in row) for row in sag_affine),
                            "NIFTI_RAS+",
                        ),
                        label_map_semantics={1: "spinal_cord", 2: "spinal_canal"},
                        command=("mock-segmentation",),
                    )

                def _close(self):
                    self.closes += 1

            mock_backend = MockBackend()

            run_dir, manifest = build_paired_nifti_workflow(
                workflow,
                sag_path,
                axial_path,
                root / "runs",
                case_id="test-case",
                run_id="run-001",
                segmentation_backend_factory=lambda _options: mock_backend,
            )
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual((mock_backend.starts, mock_backend.calls, mock_backend.closes), (1, 1, 1))
            self.assertEqual(len(manifest["workflow_outputs"][1]["artifacts"]), 9)
            persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "succeeded")
            self.assertEqual(persisted["steps"][0]["status"], "succeeded")
            for output in persisted["workflow_outputs"]:
                for item in output["artifacts"]:
                    self.assertTrue((run_dir / item["image"]).is_file())
            with self.assertRaises(FileExistsError):
                build_paired_nifti_workflow(
                    workflow, sag_path, axial_path, root / "runs",
                    case_id="test-case", run_id="run-001",
                )


    def test_multi_axial_volumes_are_segmented_independently_then_aggregated(self):
        repo = Path.cwd()
        workflow_doc = json.loads((repo / "tests" / "fixtures" / "cervical_scs_paired_series.json").read_text(encoding="utf-8"))
        workflow_doc["steps"].insert(1, {
            "id": "segment_each_axial_series",
            "tool": "segmentation",
            "operation": "segment",
            "inputs": {"series": "$input.exam.series.axial_t2"},
            "options": {"view": "axial_t2", "backend": "axial_nnunet_python", "model_id": "canal_seg", "structures": ["spinal_canal"]},
        })
        workflow_doc.setdefault("outputs", {}).setdefault("artifacts", []).append("$steps.segment_each_axial_series.mask")
        self.assertEqual(validate_workflow(workflow_doc), [])
        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            root = Path(temporary)
            affine = np.eye(4)
            sag = root / "sag.nii.gz"
            ax1, ax2 = root / "ax_level_a.nii.gz", root / "ax_level_b.nii.gz"
            nib.save(nib.Nifti1Image(np.arange(125, dtype=np.float32).reshape(5, 5, 5), affine), sag)
            nib.save(nib.Nifti1Image(np.arange(125, dtype=np.float32).reshape(5, 5, 5) + 1, affine), ax1)
            nib.save(nib.Nifti1Image(np.arange(125, dtype=np.float32).reshape(5, 5, 5) + 2, affine), ax2)
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[2, 2, 2] = 1
            labels_path = root / "mock_labels.nii.gz"
            nib.save(nib.Nifti1Image(labels, affine), labels_path)
            calls = []

            def fake_segmentation_runner(input_nii, output_dir):
                calls.append(Path(input_nii))
                output_dir.mkdir(parents=True, exist_ok=True)
                return labels_path, ["mock-segmentation"]

            workflow_path = root / "multi_workflow.json"
            workflow_path.write_text(json.dumps(workflow_doc), encoding="utf-8")
            run_dir, manifest = build_paired_nifti_workflow(
                workflow_path, sag, [ax1, ax2], root / "runs",
                case_id="multi_ax_test", run_id="multi", segmentation_runner=fake_segmentation_runner,
            )
            self.assertEqual(manifest["status"], "succeeded")
            self.assertEqual(manifest["execution_mode"], "global_physical_plane_ratio")
            self.assertEqual(len(manifest["candidates"]), 2)
            self.assertEqual(calls.count(sag), 1, "the unchanged sagittal segmentation should be reused")
            self.assertEqual(calls.count(ax1), 1)
            self.assertEqual(calls.count(ax2), 1)
            for candidate in manifest["candidates"]:
                child_manifest = json.loads((run_dir / candidate["manifest"]).read_text(encoding="utf-8"))
                axial_segment = next(step for step in child_manifest["steps"] if step["id"] == "segment_each_axial_series")
                self.assertEqual(axial_segment["status"], "succeeded")
                self.assertTrue((run_dir / candidate["match_artifact"]).is_file())

    def test_ratio_matching_is_invariant_to_splitting_axial_nifti(self):
        repo = Path.cwd()
        workflow_doc = json.loads((repo / "tests" / "fixtures" / "cervical_scs_paired_series.json").read_text(encoding="utf-8"))
        workflow_doc["steps"].insert(1, {
            "id": "segment_each_axial_series",
            "tool": "segmentation",
            "operation": "segment",
            "inputs": {"series": "$input.exam.series.axial_t2"},
            "options": {"view": "axial_t2", "backend": "axial_nnunet_python", "model_id": "canal_seg", "structures": ["spinal_canal"]},
        })
        workflow_doc.setdefault("outputs", {}).setdefault("artifacts", []).append("$steps.segment_each_axial_series.mask")
        workflow_doc["requested_regions"] = {"axial_t2": ["C3", "C3/C4", "C4", "C4/C5", "C5"]}
        self.assertEqual(validate_workflow(workflow_doc), [])

        with tempfile.TemporaryDirectory(dir=repo) as temporary:
            root = Path(temporary)
            affine = np.eye(4)
            angle = np.deg2rad(11.0)
            axial_affine = np.eye(4)
            axial_affine[:3, :3] = np.asarray([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ])
            sag = root / "sag.nii.gz"
            full_axial = root / "axial_full.nii.gz"
            workflow_path = root / "split_invariance_workflow.json"

            sagittal_labels = np.zeros((64, 64, 24), dtype=np.int16)
            sagittal_labels[32, 45:55, 1:23] = 1
            for label_id, (z0, z1) in zip((12, 13, 14, 15, 16, 17), ((2, 5), (5, 8), (8, 11), (11, 14), (14, 17), (17, 20))):
                sagittal_labels[31:34, 5:35, z0:z1] = label_id
            sag_labels_path = root / "sag_labels.nii.gz"
            nib.save(nib.Nifti1Image(sagittal_labels, affine), sag_labels_path)
            nib.save(nib.Nifti1Image(np.arange(64 * 64 * 24, dtype=np.float32).reshape(64, 64, 24), affine), sag)
            full_data = np.arange(64 * 64 * 24, dtype=np.float32).reshape(64, 64, 24)
            nib.save(nib.Nifti1Image(full_data, axial_affine), full_axial)
            workflow_path.write_text(json.dumps(workflow_doc), encoding="utf-8")

            def fake_segmentation_runner(input_nii, output_dir):
                input_nii = Path(input_nii)
                output_dir.mkdir(parents=True, exist_ok=True)
                if input_nii == sag:
                    labels = sagittal_labels
                    image_affine = affine
                else:
                    image = nib.load(str(input_nii))
                    labels = np.zeros(image.shape, dtype=np.int16)
                    labels[32, 32, :] = 1
                    image_affine = image.affine
                output_path = output_dir / "labels.nii.gz"
                nib.save(nib.Nifti1Image(labels, image_affine), output_path)
                return output_path, ["mock-segmentation"]

            single_run, single_manifest = build_paired_nifti_workflow(
                workflow_path, sag, full_axial, root / "single_runs",
                case_id="split_invariance", run_id="single",
                segmentation_runner=fake_segmentation_runner,
            )
            split_paths = []
            for part, (start, stop) in enumerate(((0, 7), (7, 15), (15, 24)), 1):
                part_affine = axial_affine.copy()
                part_affine[:3, 3] = axial_affine[:3, :3] @ np.asarray((0.0, 0.0, float(start)))
                path = root / f"axial_part_{part}.nii.gz"
                nib.save(nib.Nifti1Image(full_data[:, :, start:stop], part_affine), path)
                split_paths.append(path)
            split_run, split_manifest = build_paired_nifti_workflow(
                workflow_path, sag, split_paths, root / "split_runs",
                case_id="split_invariance", run_id="split",
                segmentation_runner=fake_segmentation_runner,
            )

            single_step = next(step for step in single_manifest["steps"] if step["id"] == "match_sagittal_to_axial_levels")
            single_artifact = next(a for a in single_step["artifacts"] if a["type"] == "SliceMatchSet")
            single_matches = json.loads((single_run / single_artifact["path"]).read_text(encoding="utf-8"))["regions"]
            for region, multi in split_manifest["region_selection"].items():
                single = single_matches[region]
                if single["status"] != "selected" or multi["status"] == "missing":
                    self.assertEqual(single["status"], multi["status"])
                    continue
                idx = single["slice_index_0based"]
                single_voxel = np.asarray((31.5, 31.5, idx, 1.0))
                single_world = (axial_affine @ single_voxel)[:3]
                ref = multi["slice_ref"]
                source_path = next(path for path in split_paths if path.name == ref["nifti_file"])
                split_image = nib.load(str(source_path))
                split_voxel = np.asarray((31.5, 31.5, ref["slice_index_0based"], 1.0))
                split_world = (split_image.affine @ split_voxel)[:3]
                np.testing.assert_allclose(single_world, split_world, atol=1e-8, err_msg=f"physical slice differs for {region}")
                self.assertAlmostEqual(single["score"] or 0.0, multi["score"] or 0.0, places=12)

if __name__ == "__main__":
    unittest.main()
