"""Two-dimensional ROI geometry rules evaluated on display-oriented masks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage


SIDES = ("left", "top", "right", "bottom")


def keep_anterior_components(
    label_plane: np.ndarray,
    *,
    target_label_ids: list[int],
    reference_label_ids: list[int],
    allow_missing_labels: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the largest anterior component of each target label relative to a reference mask.

    On KEPA's sagittal display, anterior is left and posterior is right. For each
    display row, target pixels at or posterior to the reference mask's left edge
    are excluded before retaining the largest 8-connected component per label.
    The source label plane is never modified.
    """
    plane = np.asarray(label_plane)
    if plane.ndim != 2:
        raise ValueError("anterior-component filtering requires a 2D display-oriented label plane")
    reference = np.isin(plane, reference_label_ids)
    reference_coords = np.argwhere(reference)
    if reference_coords.size == 0:
        raise ValueError(f"reference labels are absent: {reference_label_ids}")
    height, width = plane.shape
    boundary_by_row = np.full(height, np.nan, dtype=np.float64)
    for row in np.unique(reference_coords[:, 0]):
        boundary_by_row[row] = float(reference_coords[reference_coords[:, 0] == row, 1].min())
    known_rows = np.flatnonzero(np.isfinite(boundary_by_row))
    boundary_by_row = np.interp(np.arange(height), known_rows, boundary_by_row[known_rows])
    x_grid = np.arange(width, dtype=np.float64)[None, :]
    anterior_side = x_grid < boundary_by_row[:, None]

    filtered = plane.copy()
    per_label: dict[str, dict[str, int]] = {}
    structure = np.ones((3, 3), dtype=np.uint8)
    for label_id in target_label_ids:
        original = plane == int(label_id)
        if not np.any(original) and allow_missing_labels:
            per_label[str(label_id)] = {
                "original_pixels": 0,
                "anterior_candidate_pixels": 0,
                "retained_largest_component_pixels": 0,
                "removed_pixels": 0,
                "status": "label_absent_on_this_slice",
            }
            continue
        candidates = original & anterior_side
        components, count = ndimage.label(candidates, structure=structure)
        if count == 0:
            raise ValueError(f"label {label_id} has no component anterior to reference labels {reference_label_ids}")
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep_id = int(np.argmax(sizes))
        keep = components == keep_id
        filtered[original] = 0
        filtered[keep] = int(label_id)
        per_label[str(label_id)] = {
            "original_pixels": int(np.count_nonzero(original)),
            "anterior_candidate_pixels": int(np.count_nonzero(candidates)),
            "retained_largest_component_pixels": int(np.count_nonzero(keep)),
            "removed_pixels": int(np.count_nonzero(original) - np.count_nonzero(keep)),
        }
    return filtered, {
        "operation": "anterior_of_reference_then_largest_8_connected_component",
        "target_label_ids": [int(value) for value in target_label_ids],
        "reference_label_ids": [int(value) for value in reference_label_ids],
        "display_anterior_direction": "left",
        "reference_edge": "leftmost reference-mask pixel per image row; nearest-row interpolation where absent",
        "per_label": per_label,
    }


def _mask_extent(mask: np.ndarray) -> tuple[float, float, float, float]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("cannot extract a bounding box from an empty mask")
    y0, x0 = coordinates.min(axis=0)
    y1, x1 = coordinates.max(axis=0)
    # Half-open pixel-edge coordinates: [x0, x1), [y0, y1).
    return float(x0), float(y0), float(x1 + 1), float(y1 + 1)

def _edge_value(extent: tuple[float, float, float, float], side: str, rule: dict[str, Any]) -> float:
    x0, y0, x1, y1 = extent
    axis = "x" if side in {"left", "right"} else "y"
    lo, hi = (x0, x1) if axis == "x" else (y0, y1)
    default = "min" if side in {"left", "top"} else "max"
    anchor = rule.get("source_edge", default)
    if anchor not in {"min", "max", "center"}:
        raise ValueError(f"{side}.source_edge must be min, max, or center")
    position = float(rule.get("position", 0.0))
    span = hi - lo
    if anchor == "min":
        return lo - position * span
    if anchor == "max":
        return hi + position * span
    # For center anchors, positive follows the image coordinate axis.
    return (lo + hi) / 2.0 + position * span


def _scale_pixel_padding(
    padding_px: dict[str, int | float] | None,
    image_shape: tuple[int, int],
    reference_size: tuple[int, int] | list[int] | None,
) -> dict[str, float]:
    """Convert reference-pixel padding to the current display-image scale."""
    if not padding_px:
        return {}
    height, width = image_shape
    if reference_size is None:
        scale_x = scale_y = 1.0
    else:
        if (
            not isinstance(reference_size, (tuple, list))
            or len(reference_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in reference_size)
        ):
            raise ValueError("reference_size must be a pair of positive integers [width, height]")
        reference_width, reference_height = reference_size
        scale_x = width / reference_width
        scale_y = height / reference_height
    scaled: dict[str, float] = {}
    for side, value in padding_px.items():
        if side not in SIDES:
            raise ValueError(f"padding_px keys must be in {SIDES}, got {side!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"padding_px[{side!r}] must be a non-negative number")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"padding_px[{side!r}] must be a non-negative number")
        scaled[side] = value * (scale_x if side in {"left", "right"} else scale_y)
    return scaled


def _finalize_bbox(
    coordinates: dict[str, float],
    image_shape: tuple[int, int],
    padding_px: dict[str, int | float] | None = None,
    reference_size: tuple[int, int] | list[int] | None = None,
) -> tuple[int, int, int, int]:
    height, width = image_shape
    padding = _scale_pixel_padding(padding_px, image_shape, reference_size)
    left = math.floor(coordinates["left"] - padding.get("left", 0))
    top = math.floor(coordinates["top"] - padding.get("top", 0))
    right = math.ceil(coordinates["right"] + padding.get("right", 0))
    bottom = math.ceil(coordinates["bottom"] + padding.get("bottom", 0))
    left, right = max(0, left), min(width, right)
    top, bottom = max(0, top), min(height, bottom)
    if right <= left or bottom <= top:
        raise ValueError(f"ROI collapsed after clipping: {(left, top, right, bottom)}")
    return left, top, right, bottom


def mask_bbox(
    mask: np.ndarray,
    *,
    image_shape: tuple[int, int],
    position: dict[str, float] | None = None,
    padding_px: dict[str, int] | None = None,
    reference_size: tuple[int, int] | list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Return a half-open xyxy box around a mask, optionally shifted by span fractions."""
    extent = _mask_extent(np.asarray(mask, dtype=bool))
    position = position or {}
    coordinates = {
        "left": _edge_value(extent, "left", {"position": position.get("left", 0)}),
        "top": _edge_value(extent, "top", {"position": position.get("top", 0)}),
        "right": _edge_value(extent, "right", {"position": position.get("right", 0)}),
        "bottom": _edge_value(extent, "bottom", {"position": position.get("bottom", 0)}),
    }
    return _finalize_bbox(coordinates, image_shape, padding_px, reference_size)


def composite_bbox(
    label_plane: np.ndarray,
    *,
    image_shape: tuple[int, int],
    bounds: dict[str, dict[str, Any]],
    padding_px: dict[str, int] | None = None,
    reference_size: tuple[int, int] | list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Build each box edge from selected labels' min/max/center extents.

    A boundary may provide ordered ``candidates``. Each candidate has
    ``label_ids`` and optional ``scope``. A scope can constrain rows to the
    min/max extent of ``reference_label_ids`` (e.g. the target vertebral
    body's vertical span). The first candidate with pixels in scope wins.
    """
    if set(bounds) != set(SIDES):
        raise ValueError(f"composite ROI bounds must define exactly {SIDES}")
    plane = np.asarray(label_plane)
    coordinates: dict[str, float] = {}
    for side in SIDES:
        rule = bounds[side]
        candidates = rule.get("candidates", [rule])
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"{side}.candidates must be a non-empty list")
        failures: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            labels = candidate.get("label_ids")
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"{side}.candidate[{candidate_index}].label_ids must be non-empty")
            region = np.isin(plane, [int(value) for value in labels])
            scope = candidate.get("scope")
            if scope is not None:
                if scope.get("axis", "y") != "y":
                    raise ValueError(f"{side}.candidate[{candidate_index}].scope.axis supports only 'y'")
                reference_ids = scope.get("reference_label_ids")
                if not isinstance(reference_ids, list) or not reference_ids:
                    raise ValueError(f"{side}.candidate[{candidate_index}].scope.reference_label_ids must be non-empty")
                reference = np.isin(plane, [int(value) for value in reference_ids])
                reference_rows = np.flatnonzero(np.any(reference, axis=1))
                if reference_rows.size == 0:
                    failures.append(f"candidate {candidate_index}: scope reference labels absent")
                    continue
                padding = _scale_pixel_padding(
                    {"top": scope.get("padding_px", 0)}, plane.shape, reference_size
                )["top"]
                padding = math.ceil(padding)
                row_min = max(0, int(reference_rows.min()) - padding)
                row_max = min(plane.shape[0] - 1, int(reference_rows.max()) + padding)
                scoped = np.zeros_like(region)
                scoped[row_min : row_max + 1] = region[row_min : row_max + 1]
                region = scoped
            if not np.any(region):
                failures.append(f"candidate {candidate_index}: no source pixels in scope")
                continue
            coordinates[side] = _edge_value(_mask_extent(region), side, candidate)
            break
        else:
            raise ValueError(f"{side} has no usable boundary candidate ({'; '.join(failures)})")
    return _finalize_bbox(coordinates, image_shape, padding_px, reference_size)


def composite_bbox_with_provenance(
    label_plane: np.ndarray,
    *,
    image_shape: tuple[int, int],
    bounds: dict[str, dict[str, Any]],
    padding_px: dict[str, int] | None = None,
    reference_size: tuple[int, int] | list[int] | None = None,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Return the bbox and the selected fallback candidate per boundary."""
    if set(bounds) != set(SIDES):
        raise ValueError(f"composite ROI bounds must define exactly {SIDES}")
    plane = np.asarray(label_plane)
    coordinates: dict[str, float] = {}
    provenance: dict[str, Any] = {}
    for side in SIDES:
        candidates = bounds[side].get("candidates", [bounds[side]])
        tried: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            labels = candidate["label_ids"]
            region = np.isin(plane, [int(value) for value in labels])
            scope = candidate.get("scope")
            scope_rows: list[int] | None = None
            if scope is not None:
                reference = np.isin(plane, [int(value) for value in scope["reference_label_ids"]])
                rows = np.flatnonzero(np.any(reference, axis=1))
                if not rows.size:
                    tried.append({"candidate_index": candidate_index, "status": "scope_reference_absent"})
                    continue
                padding = _scale_pixel_padding(
                    {"top": scope.get("padding_px", 0)}, plane.shape, reference_size
                )["top"]
                padding = math.ceil(padding)
                row_min = max(0, int(rows.min()) - padding)
                row_max = min(plane.shape[0] - 1, int(rows.max()) + padding)
                scoped = np.zeros_like(region)
                scoped[row_min : row_max + 1] = region[row_min : row_max + 1]
                region = scoped
                scope_rows = [row_min, row_max]
            pixel_count = int(np.count_nonzero(region))
            if pixel_count == 0:
                tried.append({"candidate_index": candidate_index, "status": "no_pixels_in_scope"})
                continue
            coordinates[side] = _edge_value(_mask_extent(region), side, candidate)
            tried.append({
                "candidate_index": candidate_index,
                "status": "selected",
                "pixel_count": pixel_count,
                "label_ids": [int(value) for value in labels],
                "scope_rows_inclusive": scope_rows,
            })
            provenance[side] = {"selected_candidate_index": candidate_index, "attempts": tried}
            break
        else:
            raise ValueError(f"{side} has no usable boundary candidate: {tried}")
    return _finalize_bbox(coordinates, image_shape, padding_px, reference_size), provenance


def point_bbox(
    point_xy: tuple[float, float],
    *,
    image_shape: tuple[int, int],
    margin_px: dict[str, int],
    reference_size: tuple[int, int] | list[int] | None = None,
) -> tuple[int, int, int, int]:
    """Create a half-open xyxy box by expanding a display-space landmark."""
    x, y = (float(point_xy[0]), float(point_xy[1]))
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("landmark coordinates must be finite")
    if any(side not in margin_px or int(margin_px[side]) < 0 for side in SIDES):
        raise ValueError(f"margin_px must provide nonnegative values for {SIDES}")
    return _finalize_bbox(
        {"left": x, "top": y, "right": x, "bottom": y}, image_shape, margin_px, reference_size
    )
