"""NIfTI adaptation of the laboratory's build_tra_map.py level selector.

The original script evaluates whether each axial DICOM plane crosses points
from each sagittal vertebral mask. This module applies the same plane-side
balance score using NIfTI voxel-to-RAS affines, and uses the selected sagittal
slice plus its neighboring slices as requested by the KEPA workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kepa.tools.matching.matching3d import ImageSeriesGeometry, match_sag2tra


def _compact_region(region: str) -> str:
    if "/" not in region:
        return region
    upper, lower = region.split("/", maxsplit=1)
    if upper and lower and upper[0] == lower[0]:
        lower = lower[1:]
    return f"{upper}-{lower}"


@dataclass(frozen=True)
class LevelSliceMatch:
    level: str
    slice_index: int | None
    method: str
    score: float | None
    source_sagittal_indices: tuple[int, ...]
    details: dict[str, Any]
    status: str = "selected"
    reason: str | None = None


def _axis_normal(affine: NDArray[np.float64], slice_axis: int) -> NDArray[np.float64]:
    in_plane = [axis for axis in range(3) if axis != slice_axis]
    normal = np.cross(affine[:3, in_plane[0]], affine[:3, in_plane[1]])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("slice plane has degenerate geometry")
    return normal / norm


def _validate_affine(value: ArrayLike, name: str) -> NDArray[np.float64]:
    affine = np.asarray(value, dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError(f"{name} must be a finite 4x4 NIfTI affine")
    if abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12:
        raise ValueError(f"{name} spatial transform is singular")
    return affine


def _crossing_ratio(points_ras: NDArray[np.float64], origin_ras: NDArray[np.float64], normal_ras: NDArray[np.float64]) -> float:
    signed = (points_ras - origin_ras) @ normal_ras
    positive = int(np.count_nonzero(signed > 1e-8))
    negative = int(np.count_nonzero(signed < -1e-8))
    if min(positive, negative) == 0:
        return 0.0
    return min(positive, negative) / max(positive, negative)


def _match_requested_regions(
    labels: NDArray,
    sagittal_affine: NDArray[np.float64],
    plane_geometries: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
    sagittal_slice_axis: int,
    sagittal_indices: tuple[int, ...],
    min_voxels_per_sagittal_slice: int,
    requested_regions: list[str],
    region_label_ids: dict[str, list[int]],
) -> dict[str, LevelSliceMatch]:
    """Match requested vertebrae directly and disc levels from adjacent labels."""
    plane_count = len(plane_geometries)
    region_groups: dict[str, tuple[str, tuple[int, ...], tuple[int, ...] | None]] = {}
    anchor_groups: dict[tuple[int, ...], NDArray[np.float64]] = {}

    for region in requested_regions:
        if "/" in region:
            parts = region.split("/")
            if len(parts) != 2:
                raise ValueError(f"disc region {region!r} must name exactly two vertebrae")
            first_name, second_name = parts
            first_ids = tuple(sorted(set(region_label_ids.get(first_name, []))))
            second_ids = tuple(sorted(set(region_label_ids.get(second_name, []))))
            region_groups[region] = ("disc", first_ids, second_ids)
            for group in (first_ids, second_ids):
                if group:
                    anchor_groups.setdefault(group, np.zeros(plane_count, dtype=np.float64))
        else:
            ids = tuple(sorted(set(region_label_ids.get(region, []))))
            region_groups[region] = ("vertebra", ids, None)
            if ids:
                anchor_groups.setdefault(ids, np.zeros(plane_count, dtype=np.float64))

    for group in anchor_groups:
        voxels = np.argwhere(np.isin(labels, group))
        ratios_by_slice = []
        for sagittal_index in sagittal_indices:
            points_voxel = voxels[voxels[:, sagittal_slice_axis] == sagittal_index]
            if len(points_voxel) < min_voxels_per_sagittal_slice:
                ratios_by_slice.append(np.zeros(plane_count, dtype=np.float64))
                continue
            points_h = np.column_stack((points_voxel, np.ones(len(points_voxel), dtype=np.float64)))
            points = (points_h @ sagittal_affine.T)[:, :3]
            ratios_by_slice.append(np.asarray([
                _crossing_ratio(points, origin, normal) for origin, normal in plane_geometries
            ], dtype=np.float64))
        if ratios_by_slice:
            anchor_groups[group] = np.mean(ratios_by_slice, axis=0)

    all_scores = np.stack(list(anchor_groups.values()), axis=0) if anchor_groups else np.zeros((0, plane_count))
    result: dict[str, LevelSliceMatch] = {}
    for region in requested_regions:
        kind, first_group, second_group = region_groups[region]
        required_groups = [first_group] if second_group is None else [first_group, second_group]
        if any(not group for group in required_groups):
            missing_names = region.split("/") if kind == "disc" else [region]
            result[region] = LevelSliceMatch(
                level=region, slice_index=None, method="ratio", score=None,
                source_sagittal_indices=sagittal_indices,
                details={"required_vertebrae": missing_names},
                status="missing", reason="one or more requested vertebral labels are not mapped",
            )
            continue
        scores = [anchor_groups[group] for group in required_groups]
        if any(not np.any(score > 0) for score in scores):
            result[region] = LevelSliceMatch(
                level=region, slice_index=None, method="ratio", score=None,
                source_sagittal_indices=sagittal_indices,
                details={"vertebral_label_ids": [list(group) for group in required_groups]},
                status="missing", reason="one or more requested vertebral segmentations are unavailable",
            )
            continue

        if kind == "vertebra":
            score_values = scores[0]
            index = int(np.argmax(score_values))
            result[region] = LevelSliceMatch(
                level=region, slice_index=index, method="ratio",
                score=float(score_values[index]), source_sagittal_indices=sagittal_indices,
                details={"vertebral_label_ids": list(first_group), "crossing_ratio": float(score_values[index])},
            )
            continue

        first_scores, second_scores = scores
        first_candidates = np.flatnonzero(first_scores > 0)
        second_candidates = np.flatnonzero(second_scores > 0)
        first_index = int(first_candidates[np.argmax(first_scores[first_candidates])])
        second_index = int(second_candidates[np.argmax(second_scores[second_candidates])])
        low, high = sorted((first_index, second_index))
        zero_candidates = [
            index for index in range(low, high + 1)
            if not np.any(all_scores[:, index] > 0)
        ]
        if zero_candidates:
            selected_index = zero_candidates[-1]
            method, score = "empty_gap_slice", None
        else:
            candidates = []
            for index in range(low, high + 1):
                first_score, second_score = float(first_scores[index]), float(second_scores[index])
                if first_score > 0 or second_score > 0:
                    joint_score = min(first_score, second_score) if first_score > 0 and second_score > 0 else max(first_score, second_score)
                    candidates.append((index, joint_score))
            if not candidates:
                result[region] = LevelSliceMatch(
                    level=region, slice_index=None, method="ratio", score=None,
                    source_sagittal_indices=sagittal_indices,
                    details={"bracketing_vertebral_indices": [first_index, second_index]},
                    status="missing", reason="no candidate axial plane lies between the requested vertebrae",
                )
                continue
            selected_index, score = min(candidates, key=lambda item: item[1])
            method = "minimum_adjacent_level_crossing"
        result[region] = LevelSliceMatch(
            level=region, slice_index=int(selected_index), method=method, score=score,
            source_sagittal_indices=sagittal_indices,
            details={
                "bracketing_vertebral_indices": [first_index, second_index],
                "between_label_ids": [list(first_group), list(second_group)],
            },
        )
    return result


def match_anatomical_levels(
    sagittal_labels: ArrayLike,
    sagittal_affine: ArrayLike,
    axial_shape: tuple[int, int, int],
    axial_affine: ArrayLike,
    *,
    sagittal_slice_axis: int,
    axial_slice_axis: int,
    reference_sagittal_index: int,
    neighborhood_radius: int = 1,
    min_voxels_per_sagittal_slice: int = 41,
    requested_regions: list[str] | None = None,
    candidate_plane_geometries: list[tuple[ArrayLike, ArrayLike]] | None = None,
    region_label_ids: dict[str, list[int]] | None = None,
) -> dict[str, LevelSliceMatch]:
    """Match vertebral and intervening disc levels to axial NIfTI slices.

    TSS vertebra labels 12--17 are translated to source-script labels 2--7.
    For each requested sagittal plane in ``reference +/- radius``, the score
    is the minority/majority point count on either side of an axial plane.
    Per vertebra, the axial plane with the best balance is selected. Disc
    planes use the gap-selection logic from ``build_tra_map.py::get_label``.
    All geometry is evaluated in NIfTI RAS+; a global RAS/LPS flip leaves the
    side-count ratio unchanged.
    """

    labels = np.asarray(sagittal_labels)
    if labels.ndim != 3:
        raise ValueError("sagittal_labels must be a 3D label volume")
    if sagittal_slice_axis not in (0, 1, 2) or axial_slice_axis not in (0, 1, 2):
        raise ValueError("slice axes must be 0, 1, or 2")
    if len(axial_shape) != 3 or any(int(size) <= 0 for size in axial_shape):
        raise ValueError("axial_shape must contain three positive dimensions")
    if not 0 <= reference_sagittal_index < labels.shape[sagittal_slice_axis]:
        raise ValueError("reference_sagittal_index is outside the sagittal volume")
    if neighborhood_radius < 0 or min_voxels_per_sagittal_slice <= 0:
        raise ValueError("neighborhood radius and minimum voxel count are invalid")
    sag_aff = _validate_affine(sagittal_affine, "sagittal_affine")
    ax_aff = _validate_affine(axial_affine, "axial_affine")
    sag_indices = tuple(
        i
        for i in range(
            max(0, reference_sagittal_index - neighborhood_radius),
            min(labels.shape[sagittal_slice_axis], reference_sagittal_index + neighborhood_radius + 1),
        )
    )

    if candidate_plane_geometries is None:
        center = (np.asarray(axial_shape, dtype=np.float64) - 1.0) / 2.0
        plane_origins = []
        axial_normal = _axis_normal(ax_aff, axial_slice_axis)
        for index in range(axial_shape[axial_slice_axis]):
            voxel = center.copy()
            voxel[axial_slice_axis] = index
            plane_origins.append((ax_aff @ np.r_[voxel, 1.0])[:3])
        plane_geometries = [(origin, axial_normal) for origin in plane_origins]
    else:
        if not candidate_plane_geometries:
            raise ValueError("candidate_plane_geometries must not be empty")
        plane_geometries = []
        for origin, normal in candidate_plane_geometries:
            origin_array = np.asarray(origin, dtype=np.float64)
            normal_array = np.asarray(normal, dtype=np.float64)
            if origin_array.shape != (3,) or normal_array.shape != (3,):
                raise ValueError("each candidate plane requires a 3D origin and normal")
            if not np.isfinite(origin_array).all() or not np.isfinite(normal_array).all():
                raise ValueError("candidate plane geometries must be finite")
            norm = float(np.linalg.norm(normal_array))
            if norm < 1e-12:
                raise ValueError("candidate plane normal must be non-zero")
            plane_geometries.append((origin_array, normal_array / norm))
    plane_origins_array = np.asarray([item[0] for item in plane_geometries], dtype=np.float64)
    if region_label_ids is not None:
        requested = requested_regions or list(region_label_ids)
        return _match_requested_regions(
            labels,
            sag_aff,
            plane_geometries,
            sagittal_slice_axis,
            sag_indices,
            min_voxels_per_sagittal_slice,
            requested,
            region_label_ids,
        )

    source_labels = range(2, 8)
    scores_by_label: dict[int, NDArray[np.float64]] = {}
    for source_label in source_labels:
        voxels = np.argwhere(labels == source_label + 10)
        group_ratios: list[NDArray[np.float64]] = []
        for sagittal_index in sag_indices:
            group = voxels[voxels[:, sagittal_slice_axis] == sagittal_index]
            if len(group) < min_voxels_per_sagittal_slice:
                group_ratios.append(np.zeros(len(plane_origins_array), dtype=np.float64))
                continue
            points = np.column_stack((group, np.ones(len(group), dtype=np.float64))) @ sag_aff.T
            points = points[:, :3] / points[:, 3, None]
            group_ratios.append(
                np.asarray(
                    [
                        _crossing_ratio(points, origin, normal)
                        for origin, normal in plane_geometries
                    ],
                    dtype=np.float64,
                )
            )
        if not group_ratios or not np.any(np.max(group_ratios, axis=1) > 0):
            # C2 is only an upstream boundary cue for the first crossing pair;
            # the cervical evidence workflow promises C3-C7, so absence of a
            # usable C2 plane must not abort otherwise valid C3-C7 matching.
            if source_label == 2:
                scores_by_label[source_label] = np.zeros(len(plane_origins_array), dtype=np.float64)
                continue
            scores_by_label[source_label] = np.zeros(len(plane_origins_array), dtype=np.float64)
            continue
        scores_by_label[source_label] = np.mean(group_ratios, axis=0)

    # Reproduce get_tra's first-crossing label pair for each axial plane.
    per_axial_slice: list[tuple[int, float, int, float]] = []
    for axial_index in range(len(plane_origins_array)):
        row = (0, 0.0, 0, 0.0)
        for source_label in source_labels:
            score = float(scores_by_label[source_label][axial_index])
            if score > 0:
                next_label = source_label + 1
                next_score = float(scores_by_label[next_label][axial_index]) if next_label <= 7 else 0.0
                row = (source_label, score, next_label if next_label <= 7 else 0, next_score)
                break
        per_axial_slice.append(row)

    vertebra_indices: dict[int, tuple[int, float]] = {}
    for source_label in source_labels:
        candidates: list[tuple[int, float]] = []
        for index, (first, first_score, second, second_score) in enumerate(per_axial_slice):
            if first == source_label:
                candidates.append((index, first_score))
            if second == source_label and second_score > 0:
                candidates.append((index, second_score))
        if candidates:
            vertebra_indices[source_label] = max(candidates, key=lambda item: item[1])

    result: dict[str, LevelSliceMatch] = {}
    for source_label in range(3, 8):
        if source_label not in vertebra_indices:
            level = f"C{source_label}"
            result[level] = LevelSliceMatch(
                level=level, slice_index=None, method="ratio", score=None,
                source_sagittal_indices=sag_indices, details={"source_label": source_label},
                status="missing", reason="no usable vertebral segmentation for this region",
            )
            continue
        index, score = vertebra_indices[source_label]
        result[f"C{source_label}"] = LevelSliceMatch(
            level=f"C{source_label}",
            slice_index=index,
            method="ratio",
            score=score,
            source_sagittal_indices=sag_indices,
            details={"source_label": source_label, "crossing_ratio": score},
        )

    for source_label in range(3, 7):
        key = f"C{source_label}-{source_label + 1}"
        if source_label not in vertebra_indices or source_label + 1 not in vertebra_indices:
            result[key] = LevelSliceMatch(
                level=key, slice_index=None, method="ratio", score=None,
                source_sagittal_indices=sag_indices, details={}, status="missing",
                reason="one or both adjacent vertebral segmentations are unavailable",
            )
            continue
        first_index = vertebra_indices[source_label][0]
        second_index = vertebra_indices[source_label + 1][0]
        low, high = sorted((first_index, second_index))
        zero_candidate = -1
        candidates: list[tuple[int, float, float]] = []
        for index in range(low, high + 1):
            row = per_axial_slice[index]
            if row == (0, 0.0, 0, 0.0):
                zero_candidate = index
                continue
            if row[0] != 0:
                candidates.append((index, row[1], row[3]))
        if zero_candidate >= 0:
            selected_index, method, score = zero_candidate, "empty_gap_slice", None
        elif candidates:
            both = [candidate for candidate in candidates if candidate[2] > 0]
            if both:
                selected = min(both, key=lambda item: min(item[1], item[2]))
                selected_index, score = selected[0], min(selected[1], selected[2])
            else:
                selected = min(candidates, key=lambda item: item[1])
                selected_index, score = selected[0], selected[1]
            method = "minimum_adjacent_level_crossing"
        else:
            raise ValueError(f"could not select axial gap slice for C{source_label}-{source_label + 1}")
        result[key] = LevelSliceMatch(
            level=key,
            slice_index=selected_index,
            method=method,
            score=score,
            source_sagittal_indices=sag_indices,
            details={"bracketing_vertebral_indices": [first_index, second_index]},
        )
    requested = requested_regions
    if requested is None:
        requested = ["C3", "C3/C4", "C4", "C4/C5", "C5", "C5/C6", "C6", "C6/C7", "C7"]
    normalized = {key.replace("-", "/"): value for key, value in result.items()}
    selected: dict[str, LevelSliceMatch] = {}
    for region in requested:
        canonical = region.replace("-", "/")
        match_key = _compact_region(canonical)
        match = result.get(match_key)
        selected[canonical] = match or LevelSliceMatch(
            level=canonical, slice_index=None, method="ratio", score=None,
            source_sagittal_indices=sag_indices, details={}, status="missing",
            reason="region is not supported by the declared segmentation labels",
        )
    return selected


def match_anatomical_levels_soft_distance(
    sagittal_labels: ArrayLike,
    sagittal_affine: ArrayLike,
    axial_shape: tuple[int, int, int],
    axial_affine: ArrayLike,
    *,
    sagittal_slice_axis: int,
    axial_slice_axis: int,
    sigma_mm: float = 3.0,
    requested_regions: list[str] | None = None,
    region_label_ids: dict[str, list[int]] | None = None,
) -> dict[str, LevelSliceMatch]:
    """Match each TSS vertebra/disc label by centroid-to-axial-plane distance.

    This is the soft-distance strategy used by the earlier RSNA geometry
    matcher: transform each anatomical label's centroid to NIfTI RAS+,
    compute its perpendicular distance to every axial plane in millimeters,
    and rank planes by a Gaussian score. TSS labels 13--17 denote C3--C7;
    labels 64--67 denote C3-4 through C6-7 discs.
    """

    labels = np.asarray(sagittal_labels)
    if labels.ndim != 3:
        raise ValueError("sagittal_labels must be a 3D label volume")
    axial_aff = _validate_affine(axial_affine, "axial_affine")
    target = ImageSeriesGeometry(
        view="axial",
        shape=tuple(int(size) for size in axial_shape),
        voxel_to_patient=tuple(tuple(float(value) for value in row) for row in axial_aff),
        slice_axis=axial_slice_axis,
        coordinate_system="NIFTI_RAS+",
    )
    result: dict[str, LevelSliceMatch] = {}
    label_to_level = {
        int(label_id): _compact_region(level)
        for level, ids in (region_label_ids or {}).items()
        for label_id in ids
    } or {
        **{label: f"C{label - 10}" for label in range(13, 18)},
        64: "C3-4", 65: "C4-5", 66: "C5-6", 67: "C6-7",
    }
    requested = requested_regions
    if requested is None:
        requested = ["C3", "C3/C4", "C4", "C4/C5", "C5", "C5/C6", "C6", "C6/C7", "C7"]
    requested_keys = {_compact_region(region): region.replace("-", "/") for region in requested}
    label_to_level = {label: level for label, level in label_to_level.items() if level in requested_keys}
    for label_id, level in label_to_level.items():
        mask = labels == label_id
        if not np.any(mask):
            result[requested_keys.get(level, level)] = LevelSliceMatch(
                level=requested_keys.get(level, level), slice_index=None, method="soft_distance", score=None,
                source_sagittal_indices=(), details={"source_label": label_id}, status="missing",
                reason="segmentation label is absent from this scan",
            )
            continue
        match = match_sag2tra(
            mask,
            sagittal_affine,
            target,
            level=level,
            strategy="centroid",
            sigma_mm=sigma_mm,
            source_coordinate_system="NIFTI_RAS+",
        )
        if match.selected is None:
            raise ValueError(f"soft-distance matching found no axial plane for {level}")
        selected = match.selected
        output_level = requested_keys.get(level, level)
        result[output_level] = LevelSliceMatch(
            level=output_level,
            slice_index=selected.slice_index,
            method="soft_distance",
            score=selected.score,
            source_sagittal_indices=tuple(range(labels.shape[sagittal_slice_axis])),
            details={
                "source_label": label_id,
                "reference_centroid_ras_mm": list(match.reference_point_mm),
                "distance_mm": selected.distance_mm,
                "sigma_mm": sigma_mm,
                "score_definition": "exp(-0.5 * (distance_mm / sigma_mm)^2)",
                "candidate_distances_mm_by_index": {
                    str(candidate.slice_index): candidate.distance_mm
                    for candidate in sorted(match.candidates, key=lambda item: item.slice_index)
                },
            },
        )
    for region in requested:
        canonical = region.replace("-", "/")
        result.setdefault(canonical, LevelSliceMatch(
            level=canonical, slice_index=None, method="soft_distance", score=None,
            source_sagittal_indices=(), details={}, status="missing",
            reason="no source label mapping is declared for this region",
        ))
    return {region.replace("-", "/"): result[region.replace("-", "/")] for region in requested}
