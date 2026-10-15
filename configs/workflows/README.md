# Workflow configuration author guide

Workflow files are reusable, data-independent JSON templates. Do not create a copy per patient, place patient paths/IDs in a committed config, or embed image/mask arrays. Supply runtime input URIs and a de-identified case ID to the CLI.

## Maintained task workflows

The production workflow set is intentionally limited to the current supported tasks:

- cervical_scs_dual_view.json: cervical spinal canal stenosis.
- cervical_right_neural_foramen_axial.json: cervical right neural-foramen stenosis.
- cervical_disc_only_sagittal.json: cervical disc-space narrowing.
- rsna_lumbar_scs_soft_distance.json: RSNA lumbar spinal canal stenosis (five requested disc levels; soft-distance Sag-to-Ax matching).
- rsna_lumbar_nfn_left_dual_view.json: RSNA left neural-foramen stenosis evidence (five requested levels; per-level Sag-to-Ax selection, grouped bbox-point Ax-to-Sag selection, and both axial and sagittal ROIs).

Only the five maintained, data-independent task workflows are included here; case manifests and run outputs belong outside the repository.

## Editing checklist

- Preserve `schema_version`, declared external inputs and query-required views.
- Every step has a unique ID, known `tool.operation`, exactly the declared inputs, and supported options.
- Readable segmentation workflows can name mask results with `output` and consume them by name (`input` for the single-mask post-processing operation, or a named value in an operation's `inputs`). These aliases must refer to outputs from earlier steps. The executor translates them to typed ports internally.
- Existing workflows using `$steps.<id>.<port>` remain supported for compatibility.
- Keep each step focused. Selection and matching are distinct concepts; matching produces candidates/scores, while selection policy chooses/export results.
- Put namespaced third-party operation IDs only after the plugin catalog supports them. The current schema/registry cannot load arbitrary external tools yet.
- Validate before execution and keep patient-specific execution outputs outside this directory.

```bash
kepa workflow validate configs/workflows/cervical_scs_dual_view.json
kepa workflow validate configs/workflows/cervical_right_neural_foramen_axial.json
kepa workflow validate configs/workflows/cervical_disc_only_sagittal.json
kepa workflow validate configs/workflows/rsna_lumbar_scs_soft_distance.json
kepa workflow validate configs/workflows/rsna_lumbar_nfn_left_dual_view.json
```

See [workflow module documentation](../../src/kepa/workflow/README.md) for validation and plugin roadmap.


## Final ROI result

Use slice_selection as the ROI input port to show that it consumes the preceding selector or matcher output. The workflow-level save_qc flag optionally writes selection/matching diagnostics, detailed ROI reports and overlay previews. Give each ROI step an output alias and list those aliases under outputs.results; the paired executor combines them into outputs/roi_results.json. This is the user-facing result, organized by view with one row per requested region, including missing regions with null slice and bbox values. outputs.artifacts remains optional and is intended for intermediate provenance/debugging.
