"""Typed operation registry used by workflow validation and execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationSpec:
    inputs: dict[str, str]
    outputs: dict[str, str]
    required_options: dict[str, str]
    optional_options: dict[str, str]
    choices: dict[str, set[str]]


REGISTRY: dict[tuple[str, str], OperationSpec] = {
    ("segmentation", "segment"): OperationSpec(
        inputs={"series": "MRIImageSeries"},
        outputs={"mask": "SegmentationMask"},
        required_options={"view": "view", "backend": "backend", "model_id": "string", "structures": "strings"},
        optional_options={"label_map": "json_object"},
        choices={
            "view": {"sagittal_t2", "axial_t2"},
            "backend": {"sct_cli", "totalspineseg", "totalspineseg_cli", "totalspineseg_python", "axial_nnunet_python", "local_unet"},
            "model_id": {
                "spinalcord",
                "totalspineseg",
                "canal_seg",
                "sc_epi",
                "cervical_vertebra_unet",
            },
        },
    ),
    ("segmentation", "postprocess"): OperationSpec(
        inputs={"mask": "SegmentationMask"},
        outputs={"mask": "SegmentationMask"},
        required_options={"method": "string"},
        optional_options={
            "target_label_ids": "positive_integers",
            "reference_label_ids": "positive_integers",
            "minimum_area_fraction": "positive_number",
        },
        choices={"method": {"anterior_largest_component", "center_nearest_largest_component"}},
    ),
    ("3d_matching", "match"): OperationSpec(
        inputs={
            "source_series": "MRIImageSeries",
            "target_series": "MRIImageSeries",
            "source_mask": "SegmentationMask",
            "level": "AnatomicalLevel",
        },
        outputs={"match": "SliceMatch"},
        required_options={"direction": "string"},
        optional_options={"sigma_mm": "positive_number"},
        choices={"direction": {"sagittal_to_axial", "axial_to_sagittal"}},
    ),
    ("3d_matching", "match_anatomical_levels"): OperationSpec(
        inputs={
            "source_series": "MRIImageSeries",
            "target_series": "MRIImageSeries",
            "source_mask": "SegmentationMask",
            "reference_slice": "ImageSlice",
        },
        outputs={"matches": "SliceMatchSet"},
        required_options={"direction": "string"},
        optional_options={
            "method": "string",
            "neighborhood_radius": "nonnegative_integer",
            "min_voxels_per_sagittal_slice": "positive_integer",
            "sigma_mm": "positive_number",
            "requested_regions": "strings",
            "region_label_ids": "json_object",
        },
        choices={
            "direction": {"sagittal_to_axial"},
            "method": {"ratio", "soft_distance"},
        },
    ),
    ("3d_matching", "tra2sag_regions"): OperationSpec(
        inputs={"source_series": "MRIImageSeries", "target_series": "MRIImageSeries", "source_mask": "SegmentationMask", "source_slices": "SliceMatchSet"},
        outputs={"matches": "SliceMatchSet"},
        required_options={"direction": "string", "side": "string", "point_rules": "json_array"},
        optional_options={"requested_regions": "strings", "region_label_ids": "json_object", "sigma_mm": "positive_number"},
        choices={"direction": {"axial_to_sagittal"}, "side": {"left", "right"}},
    ),
    ("slice_selection", "select_sagittal_reference"): OperationSpec(
        inputs={"series": "MRIImageSeries", "mask": "SegmentationMask"},
        outputs={"slice": "ImageSlice"},
        required_options={"selector": "string"},
        optional_options={"label_ids": "positive_integers"},
        choices={"selector": {"max_structure_area", "max_label_union_area"}},
    ),
    ("slice_selection", "select_sagittal_regions"): OperationSpec(
        inputs={"series": "MRIImageSeries", "mask": "SegmentationMask"},
        outputs={"slices": "SliceSelectionSet"},
        required_options={"selector": "string", "region_label_ids": "json_object"},
        optional_options={},
        choices={"selector": {"max_label_union_area"}},
    ),
    ("slice_selection", "select_matched_slice"): OperationSpec(
        inputs={"series": "MRIImageSeries", "match": "SliceMatch"},
        outputs={"slice": "ImageSlice"},
        required_options={"view": "view", "selector": "string"},
        optional_options={},
        choices={"view": {"sagittal_t2", "axial_t2"}, "selector": {"best_match"}},
    ),
    ("slice_selection", "export_reference_slice"): OperationSpec(
        inputs={"series": "MRIImageSeries", "slice": "ImageSlice"},
        outputs={"evidence": "EvidenceItem"},
        required_options={"view": "view", "filename": "string"},
        optional_options={},
        choices={"view": {"sagittal_t2"}},
    ),
    ("slice_selection", "export_matched_slices"): OperationSpec(
        inputs={"series": "MRIImageSeries", "matches": "SliceMatchSet"},
        outputs={"evidence": "EvidenceBundle"},
        required_options={"view": "view", "filename_prefix": "string"},
        optional_options={},
        choices={"view": {"axial_t2"}},
    ),
    ("roi_extraction", "construct"): OperationSpec(
        inputs={"image": "ImageSlice", "mask": "SegmentationMask", "level": "AnatomicalLevel"},
        outputs={"evidence": "EvidenceItem"},
        required_options={"rule": "string"},
        optional_options={
            "structure": "string",
            "structures": "strings",
            "offset_mm": "number_pair",
            "padding_mm": "nonnegative_number",
        },
        choices={
            "rule": {"mask_bbox", "adjacent_structure_union", "shifted_mask_bbox"}
        },
    ),
    ("roi_extraction", "mask_bbox"): OperationSpec(
        inputs={"series": "MRIImageSeries", "mask": "SegmentationMask", "selection": "SliceSelection"},
        outputs={"rois": "ROIBundle", "evidence": "EvidenceBundle"},
        required_options={"label_ids": "positive_integers"},
        optional_options={"padding_px": "json_object", "reference_size": "positive_size_pair", "render_overlay": "boolean", "requested_regions": "strings"},
        choices={},
    ),
    ("roi_extraction", "composite_bbox"): OperationSpec(
        inputs={"series": "MRIImageSeries", "mask": "SegmentationMask", "selection": "SliceSelection"},
        outputs={"rois": "ROIBundle", "evidence": "EvidenceBundle"},
        required_options={"preset_file": "string"},
        optional_options={"reference_size": "positive_size_pair", "render_overlay": "boolean", "requested_regions": "strings"},
        choices={},
    ),
    ("roi_extraction", "mask_centered_bbox"): OperationSpec(
        inputs={"series": "MRIImageSeries", "mask": "SegmentationMask", "selection": "SliceSelection"},
        outputs={"rois": "ROIBundle", "evidence": "EvidenceBundle"},
        required_options={"label_ids": "positive_integers", "center_offset_px": "number_pair", "half_size_px": "positive_size_pair"},
        optional_options={"reference_size": "positive_size_pair", "render_overlay": "boolean", "requested_regions": "strings"},
        choices={},
    ),
    ("roi_extraction", "point_bbox"): OperationSpec(
        inputs={"series": "MRIImageSeries", "selection": "SliceSelection"},
        outputs={"rois": "ROIBundle", "evidence": "EvidenceBundle"},
        required_options={"margin_px": "json_object"},
        optional_options={"points": "json_array", "points_from_match": "boolean", "reference_size": "positive_size_pair", "render_overlay": "boolean", "requested_regions": "strings"},
        choices={},
    ),
}


def list_operations() -> list[str]:
    return [f"{tool}.{operation}" for tool, operation in sorted(REGISTRY)]
