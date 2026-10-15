"""Geometry-only sagittal/axial slice matching.

The functions in this module do not run segmentation and do not load image
files. Callers pass a level-specific boolean mask and image geometry. The
matching result retains every target slice and its score so a later selector
can apply a workflow-specific policy.

All affines and points must use the same patient coordinate convention (for
example, DICOM LPS). A mask must already be restricted to the anatomical
level being matched. Array axis order is explicitly represented by ``shape``
and ``slice_axis``; no RAS/LPS or sagittal/axial assumptions are inferred from
array dimension order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, sqrt
from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

View = Literal["sagittal", "axial"]
Strategy = Literal["centroid", "max_support"]
Matrix4 = tuple[tuple[float, float, float, float], ...]
Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class ImageSeriesGeometry:
    """Spatial metadata needed to construct every slice plane in a series.

    ``voxel_to_patient`` maps voxel-center coordinates ``(i, j, k, 1)`` to
    patient-space millimeters. ``slice_axis`` is the array axis that indexes
    the individual 2D images in the series.
    """

    view: View
    shape: tuple[int, int, int]
    voxel_to_patient: Matrix4
    slice_axis: int
    coordinate_system: str = "DICOM_LPS"

    def __post_init__(self) -> None:
        if self.view not in {"sagittal", "axial"}:
            raise ValueError("view must be 'sagittal' or 'axial'")
        if len(self.shape) != 3 or any(int(size) <= 0 for size in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if self.slice_axis not in (0, 1, 2):
            raise ValueError("slice_axis must be 0, 1, or 2")
        matrix = np.asarray(self.voxel_to_patient, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("voxel_to_patient must be a finite 4x4 matrix")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12:
            raise ValueError("voxel_to_patient must have an invertible spatial transform")
        if not self.coordinate_system:
            raise ValueError("coordinate_system must be named explicitly")


@dataclass(frozen=True)
class MatchCandidate:
    """One target slice's geometric match to the source anatomy."""

    slice_index: int
    plane_origin_mm: Point3
    distance_mm: float
    score: float
    support_count: int | None = None


@dataclass(frozen=True)
class MatchResult:
    """Ranked target slices; ``selected`` is absent when quality gating fails."""

    direction: str
    level: str
    source_view: View
    target_view: View
    reference_point_mm: Point3
    strategy: Strategy
    candidates: tuple[MatchCandidate, ...]
    selected: MatchCandidate | None
    max_distance_mm: float | None


def _as_affine(affine: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("affine must be a finite 4x4 voxel-to-patient matrix")
    if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12:
        raise ValueError("affine spatial transform is singular")
    return matrix


def mask_voxel_centroid_patient(mask: ArrayLike, voxel_to_patient: ArrayLike) -> Point3:
    """Return the patient-space centroid of nonzero mask voxel centers.

    ``mask`` must be a 3D, level-specific binary/label mask. For a multiclass
    label volume, pass ``volume == label`` so unrelated structures do not
    influence the centroid.
    """

    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"mask must be 3D, got {array.ndim}D")
    if array.dtype.kind not in "biuf":
        raise ValueError("mask must contain numeric or boolean values")
    voxels = np.argwhere(array != 0)
    if voxels.size == 0:
        raise ValueError("cannot match an empty source mask")
    affine = _as_affine(voxel_to_patient)
    centroid_voxel = voxels.mean(axis=0)
    patient = affine @ np.array([*centroid_voxel, 1.0], dtype=np.float64)
    if abs(float(patient[3])) < 1e-12:
        raise ValueError("affine mapped centroid to an invalid homogeneous coordinate")
    point = patient[:3] / patient[3]
    return tuple(float(value) for value in point)  # type: ignore[return-value]


def _mask_points_patient(mask: ArrayLike, affine: NDArray[np.float64]) -> NDArray[np.float64]:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"mask must be 3D, got {array.ndim}D")
    voxels = np.argwhere(array != 0)
    if voxels.size == 0:
        raise ValueError("cannot match an empty source mask")
    homogeneous = np.column_stack((voxels, np.ones(len(voxels), dtype=np.float64)))
    patient = homogeneous @ affine.T
    if np.any(np.abs(patient[:, 3]) < 1e-12):
        raise ValueError("affine mapped mask voxels to invalid homogeneous coordinates")
    return patient[:, :3] / patient[:, 3, None]


def _slice_planes(geometry: ImageSeriesGeometry) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    affine = _as_affine(geometry.voxel_to_patient)
    in_plane_axes = [axis for axis in range(3) if axis != geometry.slice_axis]
    first = affine[:3, in_plane_axes[0]]
    second = affine[:3, in_plane_axes[1]]
    normal = np.cross(first, second)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("target image has degenerate in-plane geometry")
    normal /= norm
    # Put each plane origin at the center of its in-plane field of view.
    center = (np.asarray(geometry.shape, dtype=np.float64) - 1.0) / 2.0
    indices = np.repeat(center[None, :], geometry.shape[geometry.slice_axis], axis=0)
    indices[:, geometry.slice_axis] = np.arange(geometry.shape[geometry.slice_axis])
    homogeneous = np.column_stack((indices, np.ones(len(indices), dtype=np.float64)))
    origins_h = homogeneous @ affine.T
    origins = origins_h[:, :3] / origins_h[:, 3, None]
    slice_step = affine[:3, geometry.slice_axis]
    spacing_mm = abs(float(np.dot(slice_step, normal)))
    if spacing_mm < 1e-9:
        raise ValueError("slice axis does not move between distinct planes")
    return origins, normal, spacing_mm


def _rank_slices(
    *,
    mask: ArrayLike,
    source_affine: ArrayLike,
    source_view: View,
    target: ImageSeriesGeometry,
    level: str,
    direction: str,
    strategy: Strategy,
    sigma_mm: float,
    max_distance_mm: float | None,
    support_tolerance_mm: float | None,
    source_coordinate_system: str,
) -> MatchResult:
    if not level or not isinstance(level, str):
        raise ValueError("level must be a non-empty string")
    if target.view == source_view:
        raise ValueError("source and target views must differ")
    if source_coordinate_system != target.coordinate_system:
        raise ValueError(
            "source and target coordinate systems differ: "
            f"{source_coordinate_system!r} vs {target.coordinate_system!r}"
        )
    if strategy not in {"centroid", "max_support"}:
        raise ValueError("strategy must be 'centroid' or 'max_support'")
    if not isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError("sigma_mm must be finite and positive")
    if max_distance_mm is not None and (not isfinite(max_distance_mm) or max_distance_mm < 0):
        raise ValueError("max_distance_mm must be finite and non-negative")
    affine = _as_affine(source_affine)
    origins, normal, target_spacing = _slice_planes(target)

    support_counts: NDArray[np.int64] | None = None
    if strategy == "centroid":
        reference = np.asarray(mask_voxel_centroid_patient(mask, affine), dtype=np.float64)
    else:
        points = _mask_points_patient(mask, affine)
        reference = points.mean(axis=0)
        tolerance = support_tolerance_mm if support_tolerance_mm is not None else target_spacing / 2.0
        if not isfinite(tolerance) or tolerance < 0:
            raise ValueError("support_tolerance_mm must be finite and non-negative")
        # Signed coordinates along a shared normal let us evaluate all parallel
        # slice planes without an O(mask_voxels x slices) 3D distance loop.
        plane_positions = origins @ normal
        point_positions = points @ normal
        support_counts = np.asarray(
            [np.count_nonzero(np.abs(point_positions - position) <= tolerance) for position in plane_positions],
            dtype=np.int64,
        )

    distances = np.abs((origins - reference[None, :]) @ normal)
    candidates = [
        MatchCandidate(
            slice_index=index,
            plane_origin_mm=tuple(float(v) for v in origins[index]),  # type: ignore[arg-type]
            distance_mm=float(distances[index]),
            score=float(exp(-0.5 * (float(distances[index]) / sigma_mm) ** 2)),
            support_count=None if support_counts is None else int(support_counts[index]),
        )
        for index in range(len(origins))
    ]
    if strategy == "max_support":
        candidates.sort(key=lambda item: (-int(item.support_count or 0), item.distance_mm, item.slice_index))
    else:
        candidates.sort(key=lambda item: (item.distance_mm, item.slice_index))
    best = candidates[0]
    selected = best if max_distance_mm is None or best.distance_mm <= max_distance_mm else None
    return MatchResult(
        direction=direction,
        level=level,
        source_view=source_view,
        target_view=target.view,
        reference_point_mm=tuple(float(v) for v in reference),  # type: ignore[arg-type]
        strategy=strategy,
        candidates=tuple(candidates),
        selected=selected,
        max_distance_mm=max_distance_mm,
    )


def match_sag2tra(
    sagittal_level_mask: ArrayLike,
    sagittal_voxel_to_patient: ArrayLike,
    axial_series: ImageSeriesGeometry,
    *,
    level: str,
    strategy: Strategy = "centroid",
    sigma_mm: float = 3.0,
    max_distance_mm: float | None = None,
    support_tolerance_mm: float | None = None,
    source_coordinate_system: str = "DICOM_LPS",
) -> MatchResult:
    """Match a sagittal vertebra/disc level mask to candidate axial slices.

    Pass a mask for exactly one level (vertebral body or intervertebral
    region). ``centroid`` finds the target axial plane nearest its 3D centroid;
    ``max_support`` finds the plane with the greatest mask support in a slab.
    The latter is useful when a segmentation spans multiple adjacent planes.
    """

    if axial_series.view != "axial":
        raise ValueError("match_sag2tra requires an axial target series")
    return _rank_slices(
        mask=sagittal_level_mask,
        source_affine=sagittal_voxel_to_patient,
        source_view="sagittal",
        target=axial_series,
        level=level,
        direction="sag2tra",
        strategy=strategy,
        sigma_mm=sigma_mm,
        max_distance_mm=max_distance_mm,
        support_tolerance_mm=support_tolerance_mm,
        source_coordinate_system=source_coordinate_system,
    )


def match_tra2sag(
    axial_spinal_canal_mask: ArrayLike,
    axial_voxel_to_patient: ArrayLike,
    sagittal_series: ImageSeriesGeometry,
    *,
    level: str,
    strategy: Strategy = "centroid",
    sigma_mm: float = 3.0,
    max_distance_mm: float | None = None,
    support_tolerance_mm: float | None = None,
    source_coordinate_system: str = "DICOM_LPS",
) -> MatchResult:
    """Match an axial spinal-canal mask to the nearest sagittal slice plane."""

    if sagittal_series.view != "sagittal":
        raise ValueError("match_tra2sag requires a sagittal target series")
    return _rank_slices(
        mask=axial_spinal_canal_mask,
        source_affine=axial_voxel_to_patient,
        source_view="axial",
        target=sagittal_series,
        level=level,
        direction="tra2sag",
        strategy=strategy,
        sigma_mm=sigma_mm,
        max_distance_mm=max_distance_mm,
        support_tolerance_mm=support_tolerance_mm,
        source_coordinate_system=source_coordinate_system,
    )
