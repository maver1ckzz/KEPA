# 3D matching tool family

## Purpose and invariant

Map an anatomical target in a source series to candidate planes in a target series. Matching returns ranked candidates, distances/scores and geometry; slice selection or the calling workflow decides which result to export. All calculations must use a named patient-space coordinate convention and the actual source/target affines.

## Port contract

| Input | Required information |
|---|---|
| `source_series` | Image reference and source voxel-to-world affine. |
| `target_series` | Target shape, affine, target slice axis, and view. |
| `source_mask` / anatomy | One clearly identified level/structure mask, or a multiclass segmentation with an explicit label mapping. |
| `reference_slice` / point | Optional policy-specific constraint; the operation descriptor must say whether it affects matching. |

Output is a `SliceMatch` or `SliceMatchSet`; every candidate should report target index/UID, target plane origin and normal, physical distance with units, score, rank/tie-break, and rejection/gating reason if any.

## Algorithms in this repository

### Geometry prototype: `matching3d.py`

`match_sag2tra` and `match_tra2sag` implement centroid-plane distance (`centroid`) and mask-support (`max_support`). Centroid distance is converted to a Gaussian score `exp(-0.5 * (d / sigma_mm)^2)`. The functions are geometry-level primitives: caller supplies a level-specific mask and `ImageSeriesGeometry`; they do not read NIfTI or run segmentation.

### Paired cervical level matcher: `tra_map.py`

- `ratio` (proportion calculation): uses the selected sagittal reference slice ± neighbors, scores how evenly each vertebral label's points fall on both sides of candidate axial planes, then applies the gap-selection rule for intervening discs.
- `soft_distance` (soft distance): computes a full 3D centroid for each TSS vertebral/disc label, measures perpendicular distance to every axial plane, and exports all candidate distances and Gaussian scores. Current mapping: vertebra labels 13–17; disc labels 64–67. Default `sigma_mm=3.0`.

The two methods share the same typed SliceMatchSet result but have different assumptions: ratio is slice-neighborhood/point-distribution based; soft_distance uses the full segmented structure centroid.
For ratio matching, requested_regions is the sole selector. A name without a slash (for example L3) is matched directly from that vertebra label. A slash name (for example L3/L4) is split and matched using the two adjacent vertebra labels already declared by the sagittal segmentation step. Therefore the segmentation label map must include each requested vertebra alias; no separate targets map is needed. The same rule works for cervical, thoracic and lumbar names.
A score is not a calibrated probability. A nearest plane is returned even when its score is low unless a future quality gate is explicitly configured.

### Region-conditioned axial-to-sagittal NFN selection: paired executor

The registered 3d_matching.tra2sag_regions operation consumes an axial image series and its cleaned canal mask, the per-region axial SliceMatchSet from the preceding Sag-to-Ax matcher, and side plus grouped point_rules (regions, alpha, beta, outward_shift).

For each requested region, the operation takes the bbox of the axial canal mask on that region's already-selected axial slice. It places a point at the configured fractional location in that bbox (for side=left, x = bbox_right - alpha * width; for right, x = bbox_left + alpha * width; vertically, y = bbox_top + beta * height). The point and bbox center are transformed through the source affine into patient space. The sagittal plane nearest the point is selected, then shifted by outward_shift planes in the direction from bbox center toward the point; when those positions coincide, the deterministic fallback is +1 plane for left and -1 for right. The output is a sagittal SliceMatchSet, with point coordinates and source axial provenance retained in its QC artifact.

The operation intentionally does not redo Sag-to-Ax selection: each requested level first gets an axial slice from the existing matcher, and that level-specific slice/mask is then used to select its sagittal slice. roi_extraction.point_bbox may consume this output with points_from_match: true to crop around the projected point. Grouped rules allow similar levels to share alpha/beta/shift values, while requested regions are injected from the workflow header and must each have a rule.

## Authoring a new matching method

Expose the strategy name and all parameters in the operation descriptor/JSON schema. State whether a source mask is binary or multi-label, how anatomy levels map to labels, whether the method consumes one reference slice or the full volume, what coordinate frame/units are used, and how ties/fallbacks are resolved. Preserve the complete candidate table in a step artifact for audit. Add tests for oblique axes, reversed directions, unequal voxel spacing and empty/missing levels.

The paired executor currently has explicit dispatch for the two strategies above; an independently packaged matcher needs the future plugin runtime or static registry/executor integration. See [tool author contract](../README.md).
