# ROI extraction tool family

ROI extraction consumes already selected image slices and segmentation output; it neither runs segmentation nor chooses the slices. Current paired-NIfTI operations are `mask_bbox`, `composite_bbox`, and `point_bbox`.

## Shared contract

The public input port is named slice_selection. It carries the output of the preceding slice selector or Sag-to-Ax matcher: either one shared slice, per-region selected slices, or matched level-indexed slices. The workflow-level save_qc option optionally writes selection/matching diagnostics, detailed ROI provenance and ROI-overlaid PNG previews under outputs/qc/. It does not change ROI coordinates or the compact machine-readable result.


- `series: MRIImageSeries` identifies the source NIfTI, view and display orientation.
- `mask: SegmentationMask` is required by mask/composite rules. Label IDs are backend-specific and are configured explicitly.
- Output `rois: ROIBundle` contains a JSON artifact with level, source slice index, display-oriented image dimensions, label IDs, rule and `bbox_xyxy`.
- ROI bundle output contains the computed coordinates in memory for downstream composition. With save_qc enabled, the full per-ROI provenance is serialized under outputs/qc/roi/.

Coordinates are 2D **display-oriented PNG pixel-edge coordinates**, `xyxy`, half-open (`left <= x < right`, `top <= y < bottom`). Display rows are oriented so superior/anterior is at the top; columns follow the declared view convention. This makes coordinates directly usable for the exported/displayed image. The manifest records the physical NIfTI plane, and the ROI JSON retains the source NIfTI reference and slice index. Pixel-space coordinates must not be interpreted as physical millimeters without the source plane geometry.

## Operations

All three ROI operations accept an optional top-level `reference_size: [width, height]`. It declares the pixel scale on which fixed pixel distances were authored; it does not resize the image or mask. When omitted, pixel distances are interpreted at the source display-image scale (the existing behavior). When present, fixed `padding_px` / `margin_px` values, including composite-candidate scope padding, are scaled independently by source width/reference width and source height/reference height. Fractional `position` values are unchanged because they are relative to the selected mask extent. Returned boxes always use the original display-image pixel coordinates.

For example, `reference_size: [512, 512]` makes a 48-pixel margin correspond to 48 pixels on a 512x512 display, 36 pixels on a 384x384 display, and 30 pixels on a 320x320 display. On a 512x256 source display, horizontal padding remains 48 pixels while vertical padding becomes 24 pixels.

### `roi_extraction.mask_bbox`

For each selected image, select the configured integer `label_ids` from the corresponding oriented mask plane and return their tight union box. Optional `padding_px` expands each edge independently. Works on a single `ImageSlice` or every member of a `SliceMatchSet`; the latter produces one box per level.

### `roi_extraction.composite_bbox`

Computes each of the four box edges independently from a JSON preset. Each edge may specify `label_ids`, an optional `source_edge` (`min`, `max`, or `center`) and a fractional `position`. For fallback behavior, an edge may instead define an ordered `candidates` list using the same fields. Each candidate is tried in order; a candidate with no source pixels in its optional `scope` is skipped. A `scope` currently supports a vertical row range derived dynamically from `reference_label_ids`' minimum and maximum rows, with optional `padding_px`. This allows a canal boundary to be measured only over the target vertebra's height, not from the full spinal-canal mask. For `left`/`top`, the default anchor is the mask extent minimum; for `right`/`bottom`, it is the maximum. Positive position moves outward and negative position inward, scaled by the selected mask's extent along that axis. For example, `right` with `position: 1` expands one full mask-width past its right edge. A center anchor uses the extent midpoint; position is applied along the increasing display coordinate direction. Coordinates are clipped to image bounds. The ROI JSON records which candidate was selected, attempts, label IDs and scoped rows for auditability.

The cervical preset defines five vertebral bodies and four discs. `composite_bbox` accepts one shared `ImageSlice` or a region-to-slice map from either sagittal region selection or Sag→Ax matching (`SliceSelectionSet` / `SliceMatchSet`). For mapped input, each requested ROI must have exactly one matching level key (for example, `C3/C4` maps to preset key `C3-4`), and each ROI is computed on that level's own selected slice; two levels may legitimately reference the same slice index. Missing levels are retained as null ROI records. The paired workflow runs sagittal `segmentation.postprocess` before selection, matching and ROI construction. A vertebral body's left edge uses its own horizontal center, and its top/bottom use its own extent. Its right edge first tries the right edge of canal-plus-cord labels restricted to that body's vertical span; if that local region is empty, it falls back to that body's own right edge expanded by one body width. For a disc, the left edge is the horizontal center of the union of the adjacent vertebral bodies; top/bottom use the adjacent bodies' inward one-third boundaries. Its right edge first tries locally scoped canal-plus-cord labels over that disc label's vertical span, then falls back to the adjacent-body union's right edge expanded by one union width. If no candidate has pixels, extraction fails explicitly rather than silently using a global canal boundary.

### `roi_extraction.point_bbox`

Creates one box per `{level, xy}` point using display-pixel coordinates and asymmetric `{left, top, right, bottom}` margins. It requires one `ImageSlice`; each margin must be a non-negative integer. The point and resulting box are clipped/validated against the displayed image bounds.

## Workflow example

`configs/workflows/cervical_scs_dual_view.json` shows the current task wiring: one selected sagittal image plus TotalSpineSeg labels produce nine sagittal ROIs; Sag→Ax matching produces nine axial indices, and the axial canal mask yields one tight canal box per matched axial image. Set save_qc to true to emit selection/matching/ROI details and overlay PNGs under outputs/qc/. outputs.results references ROI bundles and produces one consolidated outputs/roi_results.json.

## Error handling and extension

Empty masks, absent requested labels, malformed rules, incompatible views and collapsed/out-of-image boxes raise explicit errors rather than silently returning empty boxes. To add another backend or ROI rule, preserve the same typed inputs/outputs and document its coordinate system, label semantics, clipping and provenance.


## Mask-centered offset boxes

The paired operation mask_centered_bbox extracts each requested level's mask bbox center, adds center_offset_px in display coordinates, and creates a fixed half_size_px box around it. reference_size optionally defines the pixel scale for both offset and half-size, so values such as [48, 0] and [48, 48] remain stable when source planes are not 512 x 512. The operation is useful when the target is adjacent to a segmented landmark, such as a left/right neural-foramen crop relative to the spinal-canal center.
