# Slice selection tool family

## Purpose

Choose one or more image slices from a series using a declared anatomical criterion, matching result, or task policy. Matching estimates correspondence; selection applies the policy and records why a candidate was chosen.

## Intended operation contract

Inputs typically include `series: MRIImageSeries` and exactly the policy evidence needed by the selector, such as `SegmentationMask` or `SliceMatchSet`. Options declare view, selector, anatomy/label set, level, side, tie-break and fallback. Output is `ImageSlice` with source series, slice index/UID, physical plane, policy, selected score and candidate summary.

Export operations consume selected slice references and emit `EvidenceItem` or `EvidenceBundle`; they must retain the mapping from rendered image to original series and slice.

## Current implementation status

There is not yet a standalone `slice_selection` Python package. The registry and validator define `select_sagittal_reference`, `select_matched_slice`, `export_reference_slice`, and `export_matched_slices`; the paired executor contains their current implementation.

The paired SCS selector takes the sagittal series and its mask, then uses `selector: "max_label_union_area"` with a semantic `labels` name such as `"spinal_canal"`. The view is inferred from the series, so it is not repeated in options. A readable workflow may name the selected slice with `"output": "sagittal_reference_slice"` and consume it by that name in matching and ROI steps. Axial level mapping is currently produced by a `3d_matching` step; export preserves each level/index.

`select_sagittal_regions` selects one maximum-area slice independently for each named anatomical region and returns a `SliceSelectionSet`. Its `regions` option maps each output region to the label aliases whose union area should drive selection. This supports alternative policies without changing the selector implementation: a disc can use its own `C3/C4` mask, or use the union of adjacent vertebral masks `C3` and `C4`. The output artifact uses a canonical region-keyed map: every requested region has its own status and `slice_index_0based`, even when multiple regions select the same image slice. Missing regions remain present with a null slice index.

The same region-to-slice contract applies to Sag→Ax matching: its `SliceMatchSet` QC JSON exposes the selected index under `regions`, alongside match provenance. Internal artifact types retain method-specific details, but ROI tools consume either form as a per-region slice map. `select_sagittal_reference` is a separate scalar reference used by the current `ratio` matcher; it is not the per-region sagittal ROI selection.

## Requirements for a standalone selector

Keep the selector deterministic where possible. Specify behavior for empty masks, missing levels, ties, low scores and image-edge neighborhoods. Do not silently substitute a different anatomy or view. Return enough candidates and reason metadata for a reviewer to reproduce the choice. A new selector is not dynamically loadable in the current prototype; it must be wired into the validator and executor until the plugin system is implemented.
