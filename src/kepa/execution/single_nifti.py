"""Execute the initial two-view SCS evidence workflow from one 3D NIfTI.

The input is treated as one physical 3D volume. Sagittal and axial outputs are
selected from voxel planes whose normals best align with RAS left-right and
superior-inferior directions, respectively. This does not invent a separately
acquired axial series; the output manifest records its voxel spacing and plane
orientation so thick-slice reformats are not mistaken for native acquisitions.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from PIL import Image
from nibabel.processing import resample_from_to

from kepa.tools.matching import ImageSeriesGeometry, MatchResult, match_sag2tra
from kepa.tools.segmentation import TotalSpineSegBackend
from kepa.workflow.validator import validate_workflow

TSS_LABELS = {
    "spinal_cord": 1,
    "spinal_canal": 2,
    "vertebrae": {f"C{i}": 10 + i for i in range(1, 8)},
    "discs": {
        "C2-C3": 63,
        "C3-C4": 64,
        "C4-C5": 65,
        "C5-C6": 66,
        "C6-C7": 67,
        "C7-T1": 71,
    },
}


@dataclass(frozen=True)
class VolumeView:
    key: str
    view: str
    image_data: np.ndarray
    affine: np.ndarray
    geometry: ImageSeriesGeometry
    slice_axis: int


@dataclass(frozen=True)
class SegmentationVolume:
    labels: np.ndarray
    affine: np.ndarray
    path: Path


@dataclass(frozen=True)
class SelectedSlice:
    view: VolumeView
    index: int
    selector: str
    score: float
    score_name: str
    match: MatchResult | None = None


@dataclass(frozen=True)
class BuiltEvidence:
    view: str
    image_path: str
    slice_index: int
    slice_axis: int
    score: float
    score_name: str
    bbox_xyxy: tuple[int, int, int, int] | None
    match: MatchResult | None = None


def _canonical_volume(path: Path):
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"expected a 3D NIfTI volume, got shape {image.shape}")
    canonical = nib.as_closest_canonical(image)
    data = np.asarray(canonical.dataobj, dtype=np.float32)
    if not np.isfinite(data).any() or float(np.nanmax(data) - np.nanmin(data)) <= 0:
        raise ValueError("input NIfTI is empty or has no intensity range")
    affine = np.asarray(canonical.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
        raise ValueError("input NIfTI has invalid voxel-to-world geometry")
    return canonical, data, affine


def _plane_normal(affine: np.ndarray, slice_axis: int) -> np.ndarray:
    in_plane = [axis for axis in range(3) if axis != slice_axis]
    normal = np.cross(affine[:3, in_plane[0]], affine[:3, in_plane[1]])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-10:
        raise ValueError("NIfTI affine has degenerate slice-plane geometry")
    return normal / norm


def _best_slice_axis(affine: np.ndarray, target_normal_ras: tuple[float, float, float]) -> tuple[int, float]:
    target = np.asarray(target_normal_ras, dtype=np.float64)
    target /= np.linalg.norm(target)
    scores = [abs(float(np.dot(_plane_normal(affine, axis), target))) for axis in range(3)]
    axis = int(np.argmax(scores))
    angle = math.degrees(math.acos(min(1.0, max(-1.0, scores[axis]))))
    return axis, angle


def _make_view(key: str, view: str, data: np.ndarray, affine: np.ndarray, target_axis: tuple[float, float, float]):
    axis, angle = _best_slice_axis(affine, target_axis)
    geometry = ImageSeriesGeometry(
        view=view,
        shape=tuple(int(size) for size in data.shape),
        voxel_to_patient=tuple(tuple(float(v) for v in row) for row in affine),
        slice_axis=axis,
        coordinate_system="NIFTI_RAS+",
    )
    return VolumeView(key, view, data, affine, geometry, axis), angle


def _load_segmentation(path: Path, canonical_input) -> SegmentationVolume:
    seg_img = nib.load(str(path))
    same_shape = tuple(seg_img.shape) == tuple(canonical_input.shape)
    same_affine = np.allclose(seg_img.affine, canonical_input.affine, atol=1e-4)
    if not (same_shape and same_affine):
        seg_img = resample_from_to(seg_img, canonical_input, order=0)
    canonical_seg = nib.as_closest_canonical(seg_img)
    labels = np.rint(np.asarray(canonical_seg.dataobj)).astype(np.int16)
    if labels.ndim != 3 or not np.any(labels):
        raise ValueError("TotalSpineSeg returned an empty or non-3D label volume")
    return SegmentationVolume(labels, np.asarray(canonical_seg.affine, dtype=np.float64), path)


def _normalize_level(level: str) -> str:
    text = level.upper().strip().replace("/", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(C[1-7])-(C?[1-7]|T1)", text)
    if match:
        first, second = match.groups()
        second = second if second.startswith("C") or second == "T1" else f"C{second}"
        return f"{first}-{second}"
    match = re.fullmatch(r"C([1-7])", text)
    if match:
        return f"C{match.group(1)}"
    raise ValueError(f"unsupported cervical level {level!r}; use e.g. C5 or C5-6")


def _level_label(level: str) -> tuple[int, str]:
    normalized = _normalize_level(level)
    if "-" in normalized:
        label = TSS_LABELS["discs"].get(normalized)
        if label is None:
            raise ValueError(f"TotalSpineSeg does not define a disc label for {normalized}")
        return label, f"disc_{normalized.replace('-', '_')}"
    return TSS_LABELS["vertebrae"][normalized], f"vertebrae_{normalized}"


def _mask_for_structure(segmentation: SegmentationVolume, structure: str, level: str | None = None) -> np.ndarray:
    if structure == "spinal_canal":
        label = int(TSS_LABELS["spinal_canal"])
    elif structure in {"vertebrae", "intervertebral_discs"}:
        if level is None:
            raise ValueError(f"structure {structure!r} requires a query level")
        label, label_name = _level_label(level)
        if structure == "vertebrae" and not label_name.startswith("vertebrae_"):
            raise ValueError(f"level {level!r} resolves to a disc, not a vertebra")
        if structure == "intervertebral_discs" and not label_name.startswith("disc_"):
            raise ValueError(f"level {level!r} resolves to a vertebra, not a disc")
    else:
        raise ValueError(f"unsupported TotalSpineSeg structure {structure!r}")
    mask = segmentation.labels == label
    if not np.any(mask):
        raise ValueError(f"TotalSpineSeg output contains no voxels for label {label} ({structure}, level={level})")
    return mask


def _select_max_area(view: VolumeView, mask: np.ndarray) -> SelectedSlice:
    reduce_axes = tuple(axis for axis in range(3) if axis != view.slice_axis)
    areas = np.count_nonzero(mask, axis=reduce_axes)
    maximum = int(np.max(areas))
    if maximum == 0:
        raise ValueError("structure mask has zero area on every candidate slice")
    indices = np.flatnonzero(areas == maximum)
    center = (len(areas) - 1) / 2.0
    index = int(min(indices, key=lambda candidate: (abs(float(candidate) - center), int(candidate))))
    return SelectedSlice(view, index, "max_structure_area", float(maximum), "mask_voxels")


def _slice_bbox(mask: np.ndarray, index: int, axis: int) -> tuple[int, int, int, int] | None:
    plane = np.take(mask, index, axis=axis)
    coords = np.argwhere(plane)
    if coords.size == 0:
        return None
    # Coordinates are [axis-ordered row, axis-ordered column], half-open xyxy.
    low = coords.min(axis=0)
    high = coords.max(axis=0) + 1
    return int(low[1]), int(low[0]), int(high[1]), int(high[0])


def _to_display_array(plane: np.ndarray, view: str, slice_axis: int) -> np.ndarray:
    # Canonical NIfTI is RAS+. For standard anatomical axes, transpose so rows
    # run inferior->superior and columns follow radiological convention.
    remaining = [axis for axis in range(3) if axis != slice_axis]
    if view == "sagittal" and set(remaining) == {1, 2}:
        plane = np.transpose(plane, (remaining.index(2), remaining.index(1)))
        return np.flipud(np.fliplr(plane))
    if view == "axial" and set(remaining) == {0, 1}:
        plane = np.transpose(plane, (remaining.index(1), remaining.index(0)))
        return np.flipud(np.fliplr(plane))
    return np.asarray(plane).T


def _save_slice_png(view: VolumeView, index: int, output_path: Path) -> None:
    plane = np.take(view.image_data, index, axis=view.slice_axis)
    display = _to_display_array(plane, view.view, view.slice_axis)
    finite = display[np.isfinite(display)]
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    scaled = np.clip((display - low) / (high - low), 0.0, 1.0)
    pixels = np.rint(scaled * 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)


def _resolve_reference(ref: str, step_outputs: dict[str, dict[str, object]], inputs: dict[str, object]) -> object:
    if ref.startswith("$input."):
        return inputs[ref]
    match = re.fullmatch(r"\$steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)", ref)
    if not match:
        raise ValueError(f"unsupported workflow reference {ref!r}")
    step_id, output_name = match.groups()
    return step_outputs[step_id][output_name]


def build_single_nifti_workflow(
    workflow_path: Path,
    input_nii: Path,
    level: str,
    output_dir: Path,
    *,
    sct_executable: str = "sct_deepseg",
    segmentation_runner: Callable[[Path, Path], tuple[Path, list[str]]] | None = None,
) -> dict[str, object]:
    """Run a validated JSON workflow on one 3D NIfTI and export its evidence."""

    workflow_path, input_nii, output_dir = workflow_path.absolute(), input_nii.absolute(), output_dir.absolute()
    document = json.loads(workflow_path.read_text(encoding="utf-8"))
    errors = validate_workflow(document)
    if errors:
        raise ValueError("invalid workflow:\n" + "\n".join(errors))
    canonical, image_data, affine = _canonical_volume(input_nii)
    sagittal, sag_angle = _make_view("sagittal_t2", "sagittal", image_data, affine, (1, 0, 0))
    axial, ax_angle = _make_view("axial_t2", "axial", image_data, affine, (0, 0, 1))
    if sagittal.slice_axis == axial.slice_axis:
        raise ValueError("NIfTI geometry does not provide distinct sagittal and axial slice axes")

    inputs: dict[str, object] = {
        "$input.exam.series.sagittal_t2": sagittal,
        "$input.exam.series.axial_t2": axial,
        "$input.query.level": level,
        "$input.query.task": document["query"]["task"],
    }
    outputs: dict[str, dict[str, object]] = {}
    segmentation_cache: dict[tuple[str, str], SegmentationVolume] = {}
    command: list[str] | None = None
    runner = segmentation_runner or TotalSpineSegBackend(sct_executable).run

    for step in document["steps"]:
        step_id, tool, operation = step["id"], step["tool"], step["operation"]
        options = step["options"]
        step_inputs = {name: _resolve_reference(ref, outputs, inputs) for name, ref in step["inputs"].items()}
        result: dict[str, object]
        if (tool, operation) == ("segmentation", "segment"):
            series = step_inputs["series"]
            if not isinstance(series, VolumeView):
                raise TypeError(f"step {step_id}: expected a volume view")
            if options["backend"] != "sct_cli" or options["model_id"] != "totalspineseg":
                raise ValueError(f"step {step_id}: single-NIfTI runner currently supports SCT TotalSpineSeg only")
            cache_key = (str(input_nii), str(options["model_id"]))
            if cache_key not in segmentation_cache:
                seg_path, command = runner(input_nii, output_dir / "segmentation")
                segmentation_cache[cache_key] = _load_segmentation(seg_path, canonical)
            result = {"mask": segmentation_cache[cache_key]}
        elif (tool, operation) == ("3d_matching", "match"):
            source, target = step_inputs["source_series"], step_inputs["target_series"]
            segmentation = step_inputs["source_mask"]
            if not isinstance(source, VolumeView) or not isinstance(target, VolumeView) or not isinstance(segmentation, SegmentationVolume):
                raise TypeError(f"step {step_id}: invalid matching input contract")
            structure = "intervertebral_discs" if "-" in _normalize_level(level) else "vertebrae"
            source_mask = _mask_for_structure(segmentation, structure, level)
            match_result = match_sag2tra(
                source_mask,
                segmentation.affine,
                target.geometry,
                level=level,
                sigma_mm=float(options.get("sigma_mm", 3.0)),
                max_distance_mm=options.get("max_distance_mm"),
                source_coordinate_system="NIFTI_RAS+",
            )
            if match_result.selected is None:
                raise ValueError(f"step {step_id}: no axial plane passed matching distance gate")
            result = {"match": match_result}
        elif (tool, operation) == ("slice_selection", "select_sagittal_reference"):
            series, segmentation = step_inputs["series"], step_inputs["mask"]
            if not isinstance(series, VolumeView) or not isinstance(segmentation, SegmentationVolume):
                raise TypeError(f"step {step_id}: invalid sagittal selection input")
            mask = _mask_for_structure(segmentation, options["structure"])
            selected = _select_max_area(series, mask)
            result = {"slice": selected}
        elif (tool, operation) == ("slice_selection", "select_matched_slice"):
            series, matches = step_inputs["series"], step_inputs["match"]
            if not isinstance(series, VolumeView) or not isinstance(matches, MatchResult) or matches.selected is None:
                raise TypeError(f"step {step_id}: invalid matched-slice selection input")
            result = {"slice": SelectedSlice(series, matches.selected.slice_index, "best_match", matches.selected.score, "gaussian_plane_score", matches)}
        elif (tool, operation) == ("roi_extraction", "construct"):
            selected, segmentation = step_inputs["image"], step_inputs["mask"]
            if not isinstance(selected, SelectedSlice) or not isinstance(segmentation, SegmentationVolume):
                raise TypeError(f"step {step_id}: invalid evidence input")
            mask = _mask_for_structure(segmentation, options["structure"], level if options["structure"] in {"vertebrae", "intervertebral_discs"} else None)
            bbox = _slice_bbox(mask, selected.index, selected.view.slice_axis)
            out_image = output_dir / "slices" / f"{selected.view.view}_level_{_normalize_level(level).replace('-', '_')}.png"
            _save_slice_png(selected.view, selected.index, out_image)
            evidence = BuiltEvidence(
                view=selected.view.view,
                image_path=str(out_image),
                slice_index=selected.index,
                slice_axis=selected.view.slice_axis,
                score=selected.score,
                score_name=selected.score_name,
                bbox_xyxy=bbox,
                match=selected.match,
            )
            result = {"evidence": evidence}
        else:
            raise NotImplementedError(f"single-NIfTI executor does not implement {tool}.{operation}")
        outputs[step_id] = result

    evidence_items = [
        _resolve_reference(ref, outputs, inputs)
        for ref in document["outputs"]["evidence"]
    ]
    if not all(isinstance(item, BuiltEvidence) for item in evidence_items):
        raise TypeError("workflow outputs must resolve to built evidence items")

    def plane_info(view: VolumeView, angle: float) -> dict[str, object]:
        normal = _plane_normal(view.affine, view.slice_axis)
        spacing = abs(float(np.dot(view.affine[:3, view.slice_axis], normal)))
        return {
            "slice_axis": view.slice_axis,
            "slice_spacing_mm": spacing,
            "plane_normal_ras": [float(x) for x in normal],
            "deviation_from_standard_view_deg": angle,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "workflow_id": document["id"],
        "input_nii": str(input_nii),
        "input_shape": list(image_data.shape),
        "coordinate_system": "NIFTI_RAS+",
        "query": {"task": document["query"]["task"], "level": level},
        "segmentation_backend": "SCT TotalSpineSeg",
        "sct_command": command,
        "views": {
            "sagittal": plane_info(sagittal, sag_angle),
            "axial": plane_info(axial, ax_angle),
        },
        "evidence": [
            {
                "view": item.view,
                "image": str(Path(item.image_path).relative_to(output_dir)),
                "slice_index": item.slice_index,
                "slice_axis": item.slice_axis,
                "selection_score": item.score,
                "selection_score_name": item.score_name,
                "mask_bbox_xyxy_in_raw_plane_axes": item.bbox_xyxy,
                "match_candidates": None
                if item.match is None
                else [
                    {
                        "slice_index": candidate.slice_index,
                        "distance_mm": candidate.distance_mm,
                        "score": candidate.score,
                    }
                    for candidate in item.match.candidates[:10]
                ],
            }
            for item in evidence_items
        ],
    }
    (output_dir / "evidence.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
