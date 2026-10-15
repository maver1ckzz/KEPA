"""Small deterministic post-processing operations for segmentation label maps."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def keep_center_nearest_components(
    plane: np.ndarray,
    *,
    target_label_ids: list[int] | tuple[int, ...],
    minimum_area_fraction: float = 0.5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep one center-near 2-D component per target label.

    Components smaller than ``minimum_area_fraction`` times the largest
    component for that label are excluded. Among the remaining components,
    retain the one whose centroid is closest to the image center; ties prefer
    larger area and then the lower component ID.
    """
    if plane.ndim != 2:
        raise ValueError("center-nearest component filtering expects a 2-D slice")
    if not 0.0 < minimum_area_fraction <= 1.0:
        raise ValueError("minimum_area_fraction must be in (0, 1]")

    result = plane.copy()
    image_center = (np.asarray(plane.shape, dtype=np.float64) - 1.0) / 2.0
    reports: list[dict[str, object]] = []

    for label_id in target_label_ids:
        mask = plane == int(label_id)
        component_map, component_count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
        areas = np.bincount(component_map.ravel(), minlength=component_count + 1)[1:]
        if component_count == 0:
            reports.append({"label_id": int(label_id), "status": "absent", "component_count": 0})
            continue

        largest_area = int(areas.max())
        minimum_area = minimum_area_fraction * largest_area
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for component_id, area_value in enumerate(areas, start=1):
            area = int(area_value)
            if area < minimum_area:
                continue
            coords = np.argwhere(component_map == component_id)
            centroid = coords.mean(axis=0)
            distance = float(np.linalg.norm(centroid - image_center))
            candidates.append((distance, -area, component_id, coords))

        distance, negative_area, selected_id, selected_coords = min(candidates, key=lambda item: item[:3])
        result[mask] = 0
        result[selected_coords[:, 0], selected_coords[:, 1]] = int(label_id)
        reports.append(
            {
                "label_id": int(label_id),
                "status": "selected",
                "component_count": int(component_count),
                "candidate_count": len(candidates),
                "largest_component_area_pixels": largest_area,
                "minimum_candidate_area_pixels": float(minimum_area),
                "selected_component_id": int(selected_id),
                "selected_component_area_pixels": int(-negative_area),
                "selected_centroid_yx": selected_coords.mean(axis=0).tolist(),
                "image_center_yx": image_center.tolist(),
                "distance_to_image_center_pixels": distance,
                "removed_target_pixels": int(mask.sum() - len(selected_coords)),
            }
        )

    return result, {"labels": reports}
