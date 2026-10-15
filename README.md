# KEPA Evidence Workflow Framework

KEPA is a configurable framework for constructing traceable, disease-specific MRI evidence. A workflow is a JSON description of **what operations to run, in what order, with which inputs and options**. Algorithms live in tools; a workflow is not hard-coded to one model or one fixed pipeline.

The repository currently contains early single-volume and paired sagittal/axial NIfTI runners. They are research prototypes, not clinically validated software.

## The key idea: workflows compose tools through contracts

A workflow declares short aliases for its input series and requested anatomical regions, then lists tools in execution order. Each step names its tool, optional operation, direct input aliases, and output alias. For example, `"inputs": {"sag": "sagittal_t2"}` binds the MRI input to the name `sagittal_t2`; later steps refer to that name directly instead of using a `$steps...` path. See the five maintained examples in `configs/workflows/` for complete, runnable configurations.

The validator checks workflow syntax, supported operations/options, references, input/output types, view compatibility, execution order, and whether all required views reach final evidence or ROI results. A workflow is therefore configurable, but not an arbitrary list of commands: each connection must satisfy the producer/consumer contracts.

A paired workflow gives each ROI step an output alias and lists those aliases under outputs.results. The executor writes a compact outputs/roi_results.json containing only the case ID, view, requested region, zero-based slice index and bbox; unavailable regions are null. save_qc (default false) optionally exports selection/matching diagnostics, detailed ROI provenance and overlay previews under outputs/qc/. The ROI input port is named slice_selection because it consumes the result of an earlier selector or matcher. Intermediate artifacts in manifest.json support provenance/debugging and are not part of the final result.

A readable workflow can declare requested_regions per input alias (for example sagittal_t2 and axial_t2, each containing C4, C4/C5, etc.). Only requested regions are emitted by the matching/ROI chain. Missing anatomy is preserved as an explicit status=missing record with null slice/bbox values and a manifest warning; invalid configuration is rejected before inference. A step's output names its primary artifact so downstream inputs can refer to that alias directly.

## Extending KEPA with a new tool

The design goal is that researchers can publish an independent tool package and compose it with KEPA tools—or use only their own tools. A tool should not need to belong to a KEPA-owned algorithm family. To keep independently developed tools composable, implement the shared contract:

1. **Declare an operation** with a namespaced ID, such as `lab_name.my_segmenter.segment`.
2. **Declare named input and output ports**, each with a stable artifact type and schema version. Include view, coordinate-space, geometry, and label-map constraints where relevant.
3. **Declare options as a JSON Schema**, including defaults, units, valid ranges, and choices. Do not hide required parameters in code defaults.
4. **Implement one operation entry point** that receives resolved input artifacts, validated options, and a run context, then returns declared outputs. Inputs are read-only; derived outputs belong in the step's run directory.
5. **Write provenance**: tool/package version, code or checkpoint identity, parameters, input/output hashes, coordinate transforms, and warnings.
6. **Add validator and runtime support**, plus synthetic contract tests and an example workflow. A declaration without an executor is not runnable.

The intended Python-facing shape is:

```python
class ToolOperation:
    operation_id = "lab_name.my_segmenter.segment"

    def describe(self):
        return OperationDescriptor(
            inputs={"series": ArtifactPort("MRIImageSeries", schema_version="1.0")},
            outputs={"mask": ArtifactPort("SegmentationMask", schema_version="1.0")},
            options_schema={"type": "object", "properties": {}},
        )

    def execute(self, inputs, options, context):
        # Read inputs; write derived files only under context.step_dir.
        # Return outputs conforming to describe().outputs.
        ...
```

This is the **extension contract proposal**, not yet a dynamic plugin API. Today the JSON schema enumerates four tool families, the registry is static, and the paired executor dispatches a finite set of operations in code. External packages cannot yet be auto-discovered and executed by installing them alone. See [workflow module README](src/kepa/workflow/README.md) for the exact current boundary and the planned plugin-loading path.

## Maintained task workflows

| Workflow | Task |
|---|---|
| `cervical_scs_dual_view.json` | Cervical spinal canal stenosis (sagittal and axial). |
| `cervical_right_neural_foramen_axial.json` | Cervical right neural-foramen stenosis (axial output). |
| `cervical_disc_only_sagittal.json` | Cervical disc-space narrowing (sagittal output). |
| `rsna_lumbar_scs_soft_distance.json` | RSNA lumbar spinal canal stenosis (requested lumbar levels). |
| `rsna_lumbar_nfn_left_dual_view.json` | RSNA lumbar left neural-foramen evidence (sagittal and axial). |

These five maintained, data-independent task configs are in `configs/workflows/`. Runtime manifests and patient-specific outputs belong outside the source repository.

## Install, validate, and run

```bash
conda env create -f environment.yml
conda activate kepa
python -m pip install -r requirements.txt
kepa workflow list-tools
kepa workflow validate configs/workflows/cervical_scs_dual_view.json
```

For TotalSpineSeg, install its separately distributed pretrained weights once (about 2 GB for the current two-stage release) into a local, git-ignored data directory:

```bash
totalspineseg_init --data-dir runs/totalspineseg_data
```

Then pass `--totalspineseg-data runs/totalspineseg_data` when running a paired workflow. Install TotalSpineSeg, nnU-Net and PyTorch in the same `kepa` Python environment used to invoke the workflow. The resident worker defaults to that same Python. The top-level requirements are tested with Python 3.10 on Linux using CUDA 12.1 PyTorch wheels. `requirements-lock-linux-cu121.txt` is an optional platform-specific snapshot of the full tested dependency set; it is not intended for Windows/macOS or other CUDA/PyTorch combinations. Checkpoint files are external assets and are not bundled.

Run a paired Sag/Ax workflow:

```bash
kepa evidence build-paired \
  --sag-nii /data/case_sag.nii.gz \
  --ax-nii /data/case_ax.nii.gz \
  --workflow configs/workflows/cervical_scs_dual_view.json \
  --runs-root /data/kepa-runs \
  --case-id deidentified-case-001 \
  --totalspineseg-data /path/to/totalspineseg/weights \
  --axial-model-data /path/to/axial/model-root
```

For one case, `build-paired` runs the workflow immediately. For repeated cases, use the serial batch command below so each resident segmentation worker loads once and stays active for the batch. Required upstream model weights must already be installed; KEPA neither includes nor downloads checkpoints. The older single-volume `evidence build` prototype still uses the legacy SCT CLI adapter.

For serial batch inference, create a manifest by matching sagittal files named `<case-id>.nii[.gz]` with axial subfolders named `<case-id>/`. All axial NIfTIs in a matching folder are included:

```bash
kepa evidence make-paired-manifest \
  --sag-dir /data/rsna/nii/sag_t2 \
  --axial-root /data/rsna/nii/tra \
  --out /data/kepa-inputs.json
```

The manifest contains `case_id` and named `inputs.sag` / `inputs.axl` paths; `axl` is a list so split axial acquisitions are supported. Relative paths resolve from the manifest. Keep manifests containing real data paths outside the public repository. Run the cases one at a time while each model worker remains loaded:

```bash
kepa evidence build-paired-batch \
  --manifest /data/kepa-inputs.json \
  --workflow configs/workflows/rsna_lumbar_scs_soft_distance.json \
  --runs-root /data/kepa-runs \
  --gpus 3,4 \
  --totalspineseg-data /path/to/totalspineseg/weights \
  --axial-model-data /path/to/axial/model-root
```

`--gpus` is a required, ordered list of physical GPU IDs made available to the run. Each backend tries the listed GPUs in order during worker startup/device preflight; started workers remain resident across cases. Case execution is serial rather than multi-case parallel. Per-case runs live under `<runs-root>/<case-id>/<run-id>/`; batch summaries and worker logs are under `_batch_summaries/` and `_batch_workers/`. The command returns non-zero if any case fails, and the summary records each case status.

The `SegmentationRequest` / `SegmentationResult` contract and `start → segment* → close` lifecycle also supports axial models. `totalspineseg_cli` remains available as the reference sagittal adapter; `totalspineseg_python` is its resident sagittal backend. The resident `axial_nnunet_python` backend supports `canal_seg` (spinal canal) and `sc_epi` (spinal cord) from the official model releases. It loads nnU-Net v2 predictors directly in the `kepa` environment; SCT is not required at runtime. Supply `--axial-model-data /path/to/external/model-root`; the root may contain `model_seg_canal_t2w/nnUNetTrainer*/...` and `model_seg_sc_epi_nnunet/nnUNetTrainer*/...`. The combined workflow starts one worker for each backend and records the axial masks in `manifest.json` under `supporting_artifacts`. EPISeg is trained on EPI/fMRI, not anatomical axial T2; its use here is exploratory and needs independent mask-quality evaluation.

### Acknowledgements and third-party model assets

KEPA acknowledges the upstream TotalSpineSeg, model-canal-seg, EPISeg, nnU-Net, and SCT projects whose software or model interfaces inform this work. KEPA does not bundle their source or checkpoints. See [third-party notices](THIRD_PARTY_NOTICES.md) for source links, versions, attribution, and license notes; obtain each checkpoint from its official distribution and follow its asset-specific terms.

## Per-run output and privacy

Each call creates a new, non-overwriting run directory:

```text
<runs-root>/<case-id>/<run-id>/
├── manifest.json
├── steps/<step-id>/
└── outputs/{sagittal,axial}/
```

The manifest records inputs (paths, hashes, geometry), step status and parameters, and output artifacts. Use a de-identified `case-id`. Keep patient data, derived images, model weights, and run output outside the source repository. Do not commit real paths, patient identifiers, scans, or logs containing protected information.

## Module docs

- [Tool-wide contract and third-party author guide](src/kepa/tools/README.md)
- [Segmentation](src/kepa/tools/segmentation/README.md)
- [3D matching](src/kepa/tools/matching/README.md)
- [Slice selection](src/kepa/tools/slice_selection/README.md)
- [ROI extraction](src/kepa/tools/roi_extraction/README.md)
- [Workflow JSON, registry, and validation](src/kepa/workflow/README.md)
- [Execution and run directories](src/kepa/execution/README.md)
- [Workflow configuration authoring](configs/workflows/README.md)
- [Optional caption-generation client](caption_generation/README.md)
- [Runtime input/output contract](runtime_contract.md)
- [Tool API and extension guide](tool_api.md)
- [Testing and contribution notes](tests/README.md)

## License

KEPA uses a **mixed-license** arrangement:

- KEPA-authored source code, documentation, workflow configurations, and caption-generation client/prompts are licensed under Apache-2.0, except for the file listed below. See [`LICENSE`](LICENSE).
- [`src/kepa/tools/segmentation/axial_nnunet.py`](src/kepa/tools/segmentation/axial_nnunet.py) is adapted from SCT 7.1's nnU-Net inference implementation and is separately licensed under LGPL-3.0-only. The LGPL terms and incorporated GPLv3 terms are provided in [`LICENSE-LGPL-3.0-only`](LICENSE-LGPL-3.0-only) and [`LICENSE-GPL-3.0-only`](LICENSE-GPL-3.0-only); the source file contains SPDX, provenance, and modification notices.
- Third-party packages, model repositories, and model weights are not relicensed by these project licenses. Their individual terms and attribution are summarized in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); obtain model weights from their upstream sources and follow any asset-specific terms.

The Apache-2.0 license applies only to KEPA-owned materials for which the project has the rights to grant it; it does not replace or override the LGPL terms for the axial adapter or the licenses of third-party components. This notice describes the intended file-level scope of the licenses in this repository.

## Current limitations

- Plugin discovery and general-purpose external-tool dispatch are planned, not implemented.
- Slice selection is currently integrated into the paired executor; it is not yet an independent plugin package.
- Execution dispatch currently covers the curated operation set used by the maintained workflows; this is not yet a general third-party plugin runtime.
- The schemas and artifact envelopes are at prototype version 1.0; a public stable API needs versioning, migration policy, plugin isolation, and security review.
- Current examples/tests establish software behavior, not clinical validity or generalization.
