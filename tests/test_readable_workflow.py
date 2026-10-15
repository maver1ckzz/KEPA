from __future__ import annotations

import json
from pathlib import Path
import unittest

from kepa.workflow.validator import normalize_workflow, validate_workflow


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "configs" / "workflows" / "cervical_scs_dual_view.json"


class ReadableWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(WORKFLOW.read_text(encoding="utf-8"))

    def test_current_dual_view_workflow_normalizes_and_validates(self) -> None:
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        self.assertEqual(normalized["inputs"], {"exam": "MRIExam", "query": "DiagnosticQuery"})
        sagittal = normalized["steps"][0]
        self.assertEqual(sagittal["tool"], "segmentation")
        self.assertEqual(sagittal["inputs"]["series"], "$input.exam.series.sagittal_t2")
        self.assertEqual(sagittal["options"]["backend"], "totalspineseg_python")
        self.assertEqual(sagittal["options"]["label_map"]["spinal_canal"], [1, 2])
        axial = normalized["steps"][1]
        self.assertEqual(axial["options"]["backend"], "axial_nnunet_python")
        self.assertEqual(axial["inputs"]["series"], "$input.exam.series.axial_t2")
        self.assertEqual(normalized["steps"][2]["options"]["target_label_ids"], [13, 14, 15, 16, 17, 64, 65, 66, 67])
        self.assertEqual(normalized["steps"][2]["options"]["reference_label_ids"], [1, 2])
        self.assertEqual(normalized["steps"][2]["inputs"], {"mask": "$steps.segment_sagittal.mask"})
        self.assertEqual(normalized["steps"][4]["options"]["label_ids"], [1, 2])
        self.assertEqual(normalized["steps"][8]["options"]["label_ids"], [1])
        self.assertEqual(normalized["steps"][3]["options"]["target_label_ids"], [1])
        self.assertEqual(normalized["steps"][3]["options"]["minimum_area_fraction"], 0.5)
        self.assertEqual(
            normalized["steps"][8]["inputs"]["mask"],
            "$steps.postprocess_axial_canal.mask",
        )
        self.assertNotIn("view", normalized["steps"][4]["options"])
        self.assertNotIn("structure", normalized["steps"][4]["options"])
        self.assertEqual(
            normalized["steps"][5]["inputs"]["reference_slice"],
            "$steps.select_sagittal_reference.slice",
        )
        self.assertEqual(
            normalized["steps"][7]["inputs"]["selection"],
            "$steps.select_sagittal_roi_slices.slices",
        )
        self.assertEqual(normalized["steps"][3]["inputs"]["mask"], "$steps.segment_axial_canal.mask")
        self.assertEqual(normalized["steps"][4]["inputs"]["mask"], "$steps.postprocess_sagittal.mask")
        self.assertEqual(normalized["steps"][5]["inputs"]["source_mask"], "$steps.postprocess_sagittal.mask")
        self.assertEqual(normalized["outputs"]["results"], [
            "$steps.build_sagittal_rois.rois",
            "$steps.build_axial_rois.rois",
        ])
        self.assertEqual(normalized["steps"][7]["inputs"]["selection"], "$steps.select_sagittal_roi_slices.slices")
        self.assertFalse(normalized["save_qc"])
        self.assertNotIn("render_overlay", normalized["steps"][7]["options"])

    def test_requested_regions_and_named_match_output_normalize(self) -> None:
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        self.assertEqual(
            normalized["requested_regions"]["sagittal_t2"],
            ["C3", "C3/C4", "C4", "C4/C5", "C5", "C5/C6", "C6", "C6/C7", "C7"],
        )
        match_step = normalized["steps"][5]
        self.assertNotIn("output", match_step)
        self.assertEqual(
            normalized["steps"][8]["inputs"]["selection"],
            "$steps.match_sagittal_to_axial_levels.matches",
        )
        self.assertEqual(normalized["steps"][8]["options"]["requested_regions"], normalized["requested_regions"]["axial_t2"])

    def test_top_level_requested_regions_selects_vertebra_and_disc_matches(self) -> None:
        document = json.loads(json.dumps(self.document))
        document["requested_regions"]["axial_t2"] = ["C3", "C3/C4"]
        self.assertEqual(validate_workflow(document), [])
        normalized = normalize_workflow(document)
        self.assertEqual(normalized["requested_regions"]["axial_t2"], ["C3", "C3/C4"])
        match = next(step for step in normalized["steps"] if step["tool"] == "3d_matching")
        self.assertEqual(match["options"]["requested_regions"], ["C3", "C3/C4"])
        self.assertEqual(match["options"]["region_label_ids"]["C3"], [13])
        self.assertEqual(match["options"]["region_label_ids"]["C4"], [14])

    def test_step_must_reference_a_bound_alias(self) -> None:
        self.document["steps"][0]["input"] = "missing_series"
        self.assertTrue(any("not bound by workflow inputs" in error for error in validate_workflow(self.document)))

    def test_null_input_cannot_be_referenced(self) -> None:
        self.document["inputs"]["sag"] = None
        self.assertTrue(any("requested_regions key" in error for error in validate_workflow(self.document)))

    def test_aliases_must_be_unique(self) -> None:
        self.document["inputs"]["axl"] = "sagittal_t2"
        self.assertTrue(any("duplicate workflow input alias" in error for error in validate_workflow(self.document)))

    def test_postprocess_requires_a_previously_named_mask(self) -> None:
        self.document["steps"][2]["input"] = "unknown_mask"
        self.assertTrue(any("not a previously named output" in error for error in validate_workflow(self.document)))

    def test_named_outputs_must_be_unique(self) -> None:
        self.document["steps"][1]["output"] = "sagittal_labels"
        self.assertTrue(any("duplicate workflow output name" in error for error in validate_workflow(self.document)))

    def test_selected_slice_can_be_named_for_downstream_steps(self) -> None:
        self.document["steps"][4]["output"] = "sagittal_reference_slice"
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        self.assertEqual(
            normalized["steps"][5]["inputs"]["reference_slice"],
            "$steps.select_sagittal_reference.slice",
        )

    def test_redundant_selector_options_are_removed_but_checked(self) -> None:
        options = self.document["steps"][4]["options"]
        options["view"] = "sagittal_t2"
        options["structure"] = "spinal_canal"
        normalized = normalize_workflow(self.document)
        self.assertNotIn("view", normalized["steps"][4]["options"])
        self.assertNotIn("structure", normalized["steps"][4]["options"])
        self.assertEqual(validate_workflow(self.document), [])

        options["structure"] = "vertebrae"
        self.assertTrue(any("must be included in labels" in error for error in validate_workflow(self.document)))

    def test_flat_anatomical_label_names_and_disc_slash_convention(self) -> None:
        self.document["steps"][0]["labels"] = {
            "spinal_canal": [1, 2],
            "C3": 13, "C4": 14, "C5": 15, "C6": 16, "C7": 17,
            "C3/C4": 64, "C4/C5": 65, "C5/C6": 66, "C6/C7": 67,
        }
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        self.assertEqual(
            normalized["steps"][0]["options"]["structures"],
            ["intervertebral_discs", "spinal_canal", "vertebrae"],
        )
        self.assertEqual(normalized["steps"][2]["options"]["target_label_ids"], [13, 14, 15, 16, 17, 64, 65, 66, 67])

    def test_region_selector_resolves_direct_and_adjacent_body_aliases(self) -> None:
        self.document["steps"][0]["labels"] = {
            "spinal_canal": [1, 2],
            "C3": 13, "C4": 14, "C5": 15, "C6": 16, "C7": 17,
            "C3/C4": 64, "C4/C5": 65, "C5/C6": 66, "C6/C7": 67,
        }
        self.document["steps"].append({
            "id": "compare_region_selection",
            "tool": "slice_selection",
            "operation": "select_sagittal_regions",
            "output": "region_slices",
            "inputs": {"series": "sagittal_t2", "mask": "clean_sagittal_labels"},
            "options": {
                "selector": "max_label_union_area",
                "regions": {"C3/C4": ["C3", "C4"], "C5/C6": ["C5/C6"]},
            },
        })
        self.document.setdefault("outputs", {}).setdefault("artifacts", []).append("region_slices")
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        self.assertEqual(
            normalized["steps"][-1]["options"]["region_label_ids"],
            {"C3/C4": [13, 14], "C5/C6": [66]},
        )
        self.assertEqual(
            normalized["outputs"]["artifacts"][-1],
            "$steps.compare_region_selection.slices",
        )

    def test_composite_roi_accepts_per_region_slice_selection(self) -> None:
        self.document["steps"][0]["labels"] = {
            "spinal_canal": [1, 2],
            "C3": 13, "C4": 14, "C5": 15, "C6": 16, "C7": 17,
            "C3/C4": 64, "C4/C5": 65, "C5/C6": 66, "C6/C7": 67,
        }
        region_step = next(step for step in self.document["steps"] if step["id"] == "select_sagittal_roi_slices")
        region_step["output"] = "region_slices"
        region_step["options"]["regions"] = {"C3": ["C3"], "C3/C4": ["C3", "C4"]}
        roi_step = next(step for step in self.document["steps"] if step["id"] == "build_sagittal_rois")
        roi_step["inputs"]["slice_selection"] = "region_slices"
        self.document.setdefault("outputs", {}).setdefault("artifacts", []).append("region_slices")
        self.assertEqual(validate_workflow(self.document), [])
        normalized = normalize_workflow(self.document)
        selected = next(step for step in normalized["steps"] if step["id"] == "select_sagittal_roi_slices")
        self.assertEqual(selected["options"]["region_label_ids"], {"C3": [13], "C3/C4": [13, 14]})
        roi_step = next(step for step in normalized["steps"] if step["id"] == "build_sagittal_rois")
        self.assertEqual(
            roi_step["inputs"]["selection"],
            "$steps.select_sagittal_roi_slices.slices",
        )


if __name__ == "__main__":
    unittest.main()

