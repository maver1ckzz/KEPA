# Segmentation tool family

## Purpose

Convert a declared MRI series and segmentation specification into one or more labeled masks in a known image geometry. Segmentation tools do not choose the diagnostic slice or define the final ROI.

## Operation contract

| Port | Contract |
|---|---|
| Input `series` | `MRIImageSeries`: source URI, 3D shape, voxel-to-world affine, coordinate convention, and view. |
| Options | `view`, `backend`, `model_id`, `structures`; backend/model must advertise support for the requested view and structures. |
| Output `mask` | `SegmentationMask`: label image, source-series identity, affine/grid, label map, backend/model/version, input/output hashes. |

Masks should be returned on the input series grid unless the operation descriptor explicitly declares otherwise. If resampling is needed, record interpolation and source/target transforms. Label interpolation must preserve categorical IDs (typically nearest-neighbor); never linearly interpolate integer anatomy labels.

## Readable workflow configuration

Workflow authors select named tools rather than backend implementation IDs. The current paired workflow uses `TotalSpineSeg` for sagittal anatomy and `AxialCanalSeg` for axial canal segmentation. The executor maps these names to their registered Python backends and model IDs.

Input slots are bound once at the top level, for example `{"sag": "sagittal_t2", "axl": "axial_t2"}`. A step then refers directly to the bound name with `"input": "sagittal_t2"`; no `$inputs` prefix is needed. Use `null` for an unused slot. The validator rejects unbound names and a sagittal/axial tool connected to the wrong view.

Each named segmentation tool declares a `labels` map. Anatomical aliases use one canonical convention: `C3` denotes the C3 vertebral body, while `C3/C4` denotes the C3-C4 disc space; the same convention applies to lumbar levels such as `L5/S1`. Non-level structures keep descriptive names such as `spinal_canal`. For example: `{"spinal_canal": [1, 2], "C3": 13, "C3/C4": 64}`. These aliases map to IDs in the returned label image; they do not cause a model with fixed outputs to change its prediction classes. Tools can refer directly to an alias or a list of aliases, and group selectors such as `vertebral_bodies` and `intervertebral_discs` are derived from the naming convention for operations that need all labels of a class.

Readable workflows may name a segmentation result with `"output": "sagittal_labels"`. Later tools consume that artifact by writing the name directly, without exposing a `$steps.<id>.mask` path. A post-processing operation uses the single-mask form `"input": "sagittal_labels", "output": "clean_sagittal_labels"`. The mask artifact carries its source-series geometry and view, so post-processing does not need the original series repeated as a second input. The executor normalizes these names to typed step ports internally; manifests still record the producing step and artifact paths.

`segmentation.postprocess` is a separate declarative operation over an existing `SegmentationMask`; it does not rerun the model or mutate its source artifact. The paired executor implements sagittal `anterior_largest_component`: on every sagittal slice, clip each requested vertebra/disc label to the display-left (anterior) side of the canal/reference-label boundary, then retain that label's largest 8-connected component. It also implements axial `center_nearest_largest_component`: per axial slice and target label, retain only components whose area is at least `minimum_area_fraction` of the largest component, then choose the eligible component nearest the in-plane image center (ties prefer larger components). Both operations write a derived mask and a per-slice QC/provenance report. The original segmentation result remains available, while downstream tools explicitly choose the derived mask.

## Stable request/result and backend lifecycle

The Python contract is in `contract.py`:

- `SegmentationRequest`: one input series, one view, requested anatomy, model ID, case/step output directory, and JSON-like options.
- `SegmentationResult`: label-map path, view/model identity, backend/version, coordinate convention, label semantics, command/logs and metadata. The returned NIfTI affine remains the authoritative voxel-to-world transform.
- `SegmentationBackend`: `start()` once, `segment(request)` for one case at a time, then `close()`. A backend may keep loaded weights/model state between calls; `keeps_model_resident` declares whether it actually does.

Inputs are never modified. Each result belongs under the workflow step directory and must preserve or explicitly describe its output grid. Integer label masks are resampled only with nearest-neighbor interpolation. Each adapter declares supported views/models and rejects unsupported requests instead of silently treating one view as another.

## Current implementation

`totalspineseg_python.py` is the resident sagittal implementation. It loads the two upstream nnU-Net predictors once in `start()`, uses the installed TotalSpineSeg Python package for its two-stage pre/post-processing, and returns the final labeled NIfTI through the same `SegmentationResult` interface. It records the package version and SHA-256 of both loaded checkpoints in the result metadata. Only one stage occupies GPU memory at a time; the other stays in CPU RAM. For a single-case request, its default `serial_helpers=True` runs upstream preprocessing/postprocessing `*_mp` helpers in the worker process rather than starting a fresh process pool for each helper. nnU-Net's own prediction/export workers still use `forkserver` for Linux CUDA safety. The upstream helper mapping is restored after each request, including on exceptions. This adapter is serial: do not issue concurrent requests within one worker process. To reproduce the earlier multiprocessing path, instantiate with `serial_helpers=False`. It does not download or redistribute models.

`worker.py` and `worker_backend.py` let the lightweight KEPA environment call that in-process adapter inside a persistent Python model environment. The worker is started once per backend lifetime, accepts one JSON request at a time, and keeps its models loaded across requests. `factory.py` registers the built-in backend implementations; a future model can implement `SegmentationBackend` and add a registry entry while retaining the same request/result artifact contract. The current public workflow validator still needs a capability entry for each new backend/model pair.

`axial_nnunet.py` adds a second resident backend (`axial_nnunet_python`) with two model IDs behind the same request/result contract: `canal_seg` returns label 1 = spinal canal, and `sc_epi` returns label 1 = spinal cord. It directly initializes nnU-Net v2 predictors and implements the minimal model-specific orientation conversion in KEPA; neither SCT Python nor its CLI is required at runtime. Predictions are transformed back to the input grid, without interpolation, and the largest 3D connected canal component is retained. The worker initializes whichever external checkpoints are installed and keeps their predictors in RAM; at most one model is active on GPU. Both models accept only 3-D input through this adapter, and inference requests are serial. See `THIRD_PARTY_NOTICES.md` for SCT provenance.

Each axial result records `metadata.stage_seconds`: model activation/device transfer, source NIfTI read, orientation and array preparation, nnU-Net preprocessing, sliding-window inference, output-shape/label conversion, native-grid restoration, optional largest-component filtering, and NIfTI writing. This helps distinguish neural-network time from CPU preprocessing and output conversion.

These checkpoints are not interchangeable with TotalSpineSeg's sagittal weights. Place the official [model-canal-seg](https://github.com/ivadomed/model-canal-seg) and/or [fmri-segmentation EPISeg](https://github.com/sct-pipeline/fmri-segmentation) model release under an external root as `model_seg_canal_t2w/nnUNetTrainer*/...` and `model_seg_sc_epi_nnunet/nnUNetTrainer*/...`, respectively; pass that root with `--axial-model-data`. KEPA does not publish their weights. Both upstream repositories identify their source licenses as MIT, but model/data distribution terms should be checked at the chosen release before redistributing anything. The tested canal checkpoint release is `r20241126` (T2w); a newer contrast-agnostic release exists and has not been substituted silently.

Important domain warning: EPISeg was trained for gradient-echo EPI/fMRI, **not** anatomical axial T2 MRI. Its KEPA adapter is experimental for this dataset; a successful inference call does not establish clinical or segmentation validity. Inspect overlay quality and compare with available manual masks before using it for slice selection. Both adapters and the paired workflow have passed smoke tests on non-distributed test data. These checks establish runtime and geometry behavior only, not anatomical accuracy or clinical validity.

`totalspineseg.py` implements the first independent adapter using the upstream `totalspineseg` executable—not SCT's `sct_deepseg` command. It currently supports `sagittal_t2` only and returns the upstream `step2_output` multiclass label map. Its label map uses TotalSpineSeg IDs, including spinal cord 1, spinal canal 2, C1–C7 vertebrae 11–17, and disc levels such as C3–C4 64 through C6–C7 67. These are backend semantics, not global KEPA label IDs.

The CLI adapter preflights the executable and already-installed step-1/step-2 checkpoints. It requires `TOTALSPINESEG_DATA` or an explicit data directory and passes that root directly to upstream. KEPA does not copy, stage, or modify weights. It starts a subprocess per case (`keeps_model_resident=False`). The paired executor also accepts an already-started backend instance: a sequential multi-case driver can own one backend context, run each patient's complete JSON workflow in turn, and close it at the end.

The resident adapter keeps model startup outside the per-case loop and processes requests serially. Exact voxel-level equivalence to the upstream CLI is not guaranteed across environments. A previous Windows CLI test encountered an upstream nnU-Net issue with a non-ASCII intermediate path; that compatibility issue has not been retested.

`sct.py` remains as a legacy SCT CLI adapter for the original single-volume prototype. The paired-series workflows now select the independent TotalSpineSeg backend. They do not share an implementation with SCT.

The paired executor currently runs TotalSpineSeg on the sagittal series only. Its SCS configuration requests vertebrae, intervertebral discs and spinal canal. The current label usage is:

- spinal cord: 1; spinal canal: 2 (the sagittal selector intentionally unions 1+2);
- vertebral bodies: 13–17 for C3–C7;
- intervertebral discs: 64–67 for C3-4 through C6-7.

These IDs are TSS-specific, not universal KEPA labels. A future adapter should map backend-specific labels to a declared canonical anatomy vocabulary while preserving the original label IDs in provenance.

## Installing the upstream backend and weights

Install TotalSpineSeg and obtain its weights from the [official repository](https://github.com/neuropoly/totalspineseg) in the `kepa` environment. Set `TOTALSPINESEG_DATA` to the model-data root (containing `nnUNet/results/...`) or pass `--totalspineseg-data` to the paired executor. KEPA does not bundle, mirror, or download the executable/package or weights. KEPA method tests use the `kepa` Python for both workflow and resident worker; a different Python is used only for an explicit upstream comparison.

Example invocation:

```bash
kepa evidence build-paired \
  --sag-nii /data/case_sag.nii.gz --ax-nii /data/case_ax.nii.gz \
  --workflow configs/workflows/rsna_lumbar_scs_soft_distance.json \
  --runs-root /data/kepa-runs --totalspineseg /path/to/env/bin/totalspineseg \
  --totalspineseg-data /data/totalspineseg-models
```

Resident example: choose `cervical_scs_dual_view.json` and pass `--totalspineseg-data /path/to/weights`. The worker defaults to the current Python, which must have TotalSpineSeg, nnU-Net, PyTorch and KEPA's core dependencies installed.

`kepa evidence build-paired-batch` accepts a JSON manifest and runs cases serially while keeping each model worker alive for the entire batch. Calling `build-paired` separately starts a new worker for every run.

The canonical manifest is an object with a `cases` array. Use paths relative to the manifest or absolute paths; `axl` may name one axial NIfTI or a list of axial NIfTIs:

```json
{
  "manifest_version": "1.0",
  "cases": [
    {"case_id": "case-001", "inputs": {"sag": "sag/case-001.nii.gz", "axl": ["axl/case-001-a.nii.gz", "axl/case-001-b.nii.gz"]}}
  ]
}
```

Run with `kepa evidence build-paired-batch --manifest cases.json --workflow <workflow.json> --runs-root <output-dir> --gpus 3,4`. GPU IDs are physical device IDs in preferred order. A worker is pinned to one requested GPU; if its startup/device preflight fails (for example, due to insufficient memory), KEPA tries the next listed GPU. If only one ID is supplied, all model workers use that GPU. Cases remain sequential, and a failed case is recorded while later cases continue. A timestamped `_batch_summaries/*.json` records status, selected GPU per backend, and each case's output path or error. The old `--cases` argument and list-shaped manifest remain accepted for compatibility.

For the RSNA-style layout (`sag_dir/<case>.nii.gz` and `axial_root/<case>/*.nii.gz`), generate the manifest without typing file names:

```bash

For a combined sagittal/axial workflow, `configs/workflows/cervical_scs_dual_view.json` declares the two axial masks as supporting artifacts, in addition to sagittal and axial evidence PNGs. The batch driver creates one resident worker for each backend named in the workflow, runs a patient's steps in declared order, then moves to the next patient without unloading either worker. `outputs.artifacts` currently accepts `SegmentationMask` references; masks appear in `manifest.json` under `supporting_artifacts` and are not themselves image-evidence PNGs.

## Implementing another backend

Implement `SegmentationBackend`, declare view/model capability, preflight dependencies/checkpoints in `_start`, implement one-case inference in `_segment`, and release resources in `_close`. A batch workflow should call `segment` for each case in sequence while the backend object remains alive; it must not change the workflow into a global “segment every patient first” stage. Record upstream repository/version, checkpoint source/hash, preprocessing, postprocessing, and license. Do not download weights implicitly during a patient run.

Today a new segmentation adapter alone is not enough to make it selectable: add its capability to the static validator catalog and ensure the active executor can dispatch it. Dynamic discovery is planned; see [tool author contract](../README.md).
