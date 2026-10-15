# Execution layer

## Responsibilities

An executor consumes a validated workflow plus runtime inputs, resolves step references, invokes each operation, persists step artifacts, updates run provenance, and fails the run explicitly if any required step fails. It owns run-directory creation and output manifests; algorithms belong in tool implementations.

## Current executors

- `single_nifti.py`: one 3D NIfTI entry point (`kepa evidence build`). It selects/reformats views from one volume; this is not equivalent to a separately acquired axial series.
- `paired_nifti.py`: separate Sag and Ax NIfTI entry point (`kepa evidence build-paired`). It infers slice axes from NIfTI affines, segments/reuses Sag labels, selects the canal reference, matches levels, exports evidence, and writes a manifest.
- The --ax-nii argument accepts one or more axial NIfTIs. Each series stays on its native grid and is segmented independently; sagittal segmentation is reused and resident backends remain loaded. For ratio matching, every real slice plane across all files is scored in one virtual sequence using its own affine-derived origin and normal. The sequence follows the reference NIfTI slice-axis direction and is ordered by physical plane position; volumes are neither concatenated nor resampled. Adjacent vertebral matches are then computed globally, and every plane between each adjacent pair is considered for disc selection, including planes at file boundaries. Thus splitting an otherwise identical regularly sampled NIfTI into multiple files does not change the selected physical planes or scores; the final local indices are mapped back to their source files. Equal-score ties use the first plane in the stable physical sequence. Every selected axial reference is serialized as `slice_ref: {nifti_file, slice_index_0based}`; the latter is local to that NIfTI and zero-based. The `source_nifti` field retains the full path for provenance.

The paired executor currently resolves intermediate artifacts as in-memory typed Python objects within one process. It also writes durable NIfTI/JSON/PNG files into each step directory. The final manifest stores JSON-safe references/metadata rather than Python objects. Restarting at an arbitrary step by rehydrating every object from the manifest is not implemented yet.

## Run-directory contract

```text
<runs-root>/<case-id>/<run-id>/
├── manifest.json
├── steps/<step-id>/
└── outputs/{sagittal,axial}/
```

The run ID must be unique; an existing run is never overwritten. Step-owned intermediate files live under that step. The public outputs.results contract writes one compact outputs/roi_results.json with case ID, view, region, zero-based slice index and bbox; missing regions are null. The workflow-level save_qc option (default false) writes optional slice-selection/matching diagnostics, detailed ROI provenance and overlay previews under outputs/qc/. manifest.json records run provenance and is separate from the diagnostic result. Input volumes are referenced and hashed, not modified. See [runtime contract](../../../runtime_contract.md) for exact fields and the reuse-segmentation regression option.

## Adding executor support

With today's implementation, a new registered operation also needs explicit executor dispatch and a test that checks the declared outputs exist and satisfy the artifact contract. Long term, executors should dispatch by operation ID to allow-listed installed plugins via a stable `ToolContext`, not grow one central `if/elif` branch per third-party method. That plugin dispatcher and process isolation are not implemented in this prototype.
