# Workflow schema, aliases, and validation

## Configuration model

A workflow is a data-independent JSON template. It declares input aliases, requested anatomical regions, ordered tool steps, and final result aliases. Patient paths, scans, model weights, and generated outputs belong outside the configuration.

The preferred input form binds short names to view names once:

```json
"inputs": {"sag": "sagittal_t2", "axl": "axial_t2"}
```

The values (`sagittal_t2`, `axial_t2`) are the names used by later steps. A step's `id` is a unique local identifier; `input` or named `inputs` connect artifacts by alias, and `output` gives the produced artifact a readable name. The artifact's actual type is determined by the tool operation, not by its alias.

`requested_regions` lists the anatomy required in each view, using the input alias value as the key. Names such as `C4`, `C4/C5`, or `L5/S1` are interpreted by the connected operations. A requested region absent from the scan is retained as a missing result with a null slice and bbox; malformed aliases, options, or connections are validation errors.

Older workflows using `$input...` and `$steps.<id>.<port>` references remain accepted for compatibility, but new configurations should use direct aliases.

## Outputs and quality-control artifacts

Use `outputs.results` to name the final ROI bundles. The paired executor writes a compact `outputs/roi_results.json` with one entry per requested region, including view, region, zero-based slice index, bbox, and missing status. Set `save_qc: true` to retain selection/matching details, provenance, and overlay previews under `outputs/qc/`. Intermediate masks and operation manifests are debugging/provenance artifacts, not the concise user-facing result.

## Validation and supported operations

The validator checks the JSON schema, unique step IDs, known tools/operations/options, direct input/output aliases, connection order, artifact types, view compatibility, and whether requested views reach final results. A syntactically valid workflow is not proof that a particular scan contains the requested anatomy; runtime checks validate actual image geometry and mask contents.

The operation catalog and executor are static and intentionally finite today. A third-party operation cannot be activated by JSON alone: it must be declared in the registry, accepted by semantic validation, dispatched by an executor, and covered by tests. Dynamic plugin discovery remains a future design goal.

## Maintained examples

The maintained task configurations are in `configs/workflows/`:

- `cervical_scs_dual_view.json`
- `cervical_right_neural_foramen_axial.json`
- `cervical_disc_only_sagittal.json`
- `rsna_lumbar_scs_soft_distance.json`
- `rsna_lumbar_nfn_left_dual_view.json`

Validate a workflow before running it:

```bash
kepa workflow validate configs/workflows/cervical_scs_dual_view.json
kepa workflow validate configs/workflows/rsna_lumbar_scs_soft_distance.json
```

Run synthetic unit and integration tests with `python -m pytest -q`. See `configs/workflows/README.md` for authoring guidance and `../execution/README.md` for run-directory and batch behavior.
