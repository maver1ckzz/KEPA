"""Resident TotalSpineSeg adapter using the installed Python package and weights.

No upstream source or checkpoints are copied into KEPA. The upstream two-stage
pre/post-processing is retained; only its per-call nnU-Net predictor is replaced
with predictors initialized once for this backend's lifetime.
"""

from __future__ import annotations

import importlib
import hashlib
import multiprocessing as mp
import os
import time
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any

import nibabel as nib

from .contract import SegmentationBackend, SegmentationGeometry, SegmentationRequest, SegmentationResult
from .totalspineseg import TSS_LABELS, _installed_dataset_dirs, _nii_stem


_UPSTREAM_HELPER_LOCK = RLock()


def _serial_process_map(function: Any, *iterables: Any, **kwargs: Any) -> list[Any]:
    """Run a single-case TotalSpineSeg helper without a new process pool."""
    return [function(*items) for items in zip(*iterables)]


@contextmanager
def _serial_upstream_helpers(upstream: Any):
    """Temporarily replace only upstream *_mp helper maps, then restore them.

    The worker handles one request at a time. The lock protects other instances
    of this adapter in the same process while the upstream module is patched.
    """
    with _UPSTREAM_HELPER_LOCK:
        patched: dict[int, tuple[dict[str, Any], Any]] = {}
        try:
            for name, function in vars(upstream).items():
                if name.endswith("_mp") and callable(function) and hasattr(function, "__globals__"):
                    namespace = function.__globals__
                    if "process_map" in namespace and id(namespace) not in patched:
                        patched[id(namespace)] = (namespace, namespace["process_map"])
                        namespace["process_map"] = _serial_process_map
            if not patched:
                raise RuntimeError("no TotalSpineSeg multiprocessing helpers found; check the installed package version")
            yield len(patched)
        finally:
            for namespace, original in patched.values():
                namespace["process_map"] = original


class TotalSpineSegPythonBackend(SegmentationBackend):
    """Run sagittal TotalSpineSeg with one resident predictor per model stage.

    Calls are serial. The inactive stage is moved to CPU, so only one network
    occupies GPU memory at a time while both sets of weights stay in RAM.
    """

    backend_id = "totalspineseg_python"
    supported_views = frozenset({"sagittal_t2"})
    supported_models = frozenset({"totalspineseg"})
    keeps_model_resident = True

    def __init__(self, data_dir: Path | None = None, *, device: str = "cuda", max_workers: int = 1,
                 serial_helpers: bool = True) -> None:
        super().__init__()
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if serial_helpers and max_workers != 1:
            raise ValueError("serial_helpers requires max_workers=1")
        self.data_dir = Path(data_dir).expanduser().absolute() if data_dir else None
        self.device_name = device
        self.max_workers = max_workers
        self.serial_helpers = serial_helpers
        self._predictors: dict[Path, Any] = {}
        self._active_model: Path | None = None
        self._checkpoint_info: dict[str, dict[str, str]] = {}
        self._load_timings: dict[str, float] = {}
        self._stage_timings: dict[str, float] = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _start(self) -> None:
        backend_started = time.perf_counter()
        # Upstream preprocessing starts worker processes. Forking after torch's
        # model initialization can deadlock; use upstream's --no-stalling mode.
        if os.name == "posix" and self.device_name == "cuda":
            mp.set_start_method("forkserver", force=True)
        data_dir = self.data_dir or (
            Path(os.environ["TOTALSPINESEG_DATA"]).expanduser().absolute()
            if os.environ.get("TOTALSPINESEG_DATA") else None
        )
        if data_dir is None:
            raise RuntimeError("set TOTALSPINESEG_DATA or provide an installed TotalSpineSeg model-data directory")
        self._model_dirs = _installed_dataset_dirs(data_dir)
        try:
            import torch
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            import totalspineseg
        except ImportError as exc:
            raise RuntimeError(
                "TotalSpineSeg Python backend requires totalspineseg, nnunetv2 and torch "
                "in its model runtime"
            ) from exc
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the model runtime")
        self._torch = torch
        self._upstream = importlib.import_module("totalspineseg.inference")
        self._version = getattr(totalspineseg, "__version__", "unknown")
        self._device = torch.device(self.device_name)
        self.data_dir = data_dir
        try:
            for dataset_dir in self._model_dirs.values():
                configurations = sorted(path for path in dataset_dir.iterdir() if path.is_dir() and (path / "fold_0").is_dir())
                if len(configurations) != 1:
                    raise RuntimeError(f"expected one fold_0 configuration in {dataset_dir}, found {len(configurations)}")
                model_folder = configurations[0]
                fold_dir = model_folder / "fold_0"
                checkpoint = "checkpoint_final.pth" if (fold_dir / "checkpoint_final.pth").is_file() else "checkpoint_best.pth"
                checkpoint_path = fold_dir / checkpoint
                hash_started = time.perf_counter()
                self._checkpoint_info[dataset_dir.name] = {
                    "path": str(checkpoint_path),
                    "sha256": self._sha256(checkpoint_path),
                }
                self._load_timings[f"{dataset_dir.name}_hash_seconds"] = time.perf_counter() - hash_started
                model_started = time.perf_counter()
                predictor = nnUNetPredictor(
                    tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                    perform_everything_on_device=True, device=self._device,
                    verbose=False, verbose_preprocessing=False, allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(str(model_folder), [0], checkpoint_name=checkpoint)
                predictor._kepa_checkpoint = checkpoint
                self._predictors[model_folder.resolve()] = predictor
                self._load_timings[f"{dataset_dir.name}_load_seconds"] = time.perf_counter() - model_started
            self._load_timings["total_start_seconds"] = time.perf_counter() - backend_started
        except Exception:
            self._close()
            raise

    def _predict_nnunet(self, *, model_folder: Path, images_dir: Path, output_dir: Path,
                        folds: str, checkpoint: str, npp: int, nps: int,
                        save_probabilities: bool = False, step_size: float = 0.5,
                        disable_tta: bool = False, continue_prediction: bool = False,
                        prev_stage_predictions: Path | None = None, num_parts: int = 1,
                        part_id: int = 0, device: Any = None, verbose: bool = False,
                        disable_progress_bar: bool = False) -> None:
        key = Path(model_folder).resolve()
        predictor = self._predictors.get(key)
        if predictor is None:
            raise RuntimeError(f"upstream requested an uninitialized model: {model_folder}")
        if str(folds) != "0" or checkpoint != predictor._kepa_checkpoint:
            raise RuntimeError("upstream model fold/checkpoint differs from the preloaded predictor")
        if step_size != 0.5 or disable_tta or verbose or device != self._device:
            raise RuntimeError("upstream inference options differ from the resident predictor settings")
        if self._active_model != key:
            if self._active_model is not None:
                self._predictors[self._active_model].network.to("cpu")
                if self.device_name == "cuda":
                    self._torch.cuda.empty_cache()
            self._active_model = key
        output_dir.mkdir(parents=True, exist_ok=True)
        stage_started = time.perf_counter()
        predictor.predict_from_files(
            str(images_dir), str(output_dir), save_probabilities=save_probabilities,
            overwrite=not continue_prediction,
            num_processes_preprocessing=npp,
            num_processes_segmentation_export=nps,
            folder_with_segs_from_prev_stage=str(prev_stage_predictions) if prev_stage_predictions else None,
            num_parts=num_parts, part_id=part_id,
        )
        self._stage_timings[key.parent.name] = time.perf_counter() - stage_started

    def _segment(self, request: SegmentationRequest) -> SegmentationResult:
        assert self.data_dir is not None
        output_dir = request.output_dir.absolute()
        if any(output_dir.iterdir()):
            raise FileExistsError(f"TotalSpineSeg output directory must be empty: {output_dir}")
        started = time.perf_counter()
        self._stage_timings = {}
        original = self._upstream.predict_nnunet
        helper_context = _serial_upstream_helpers(self._upstream) if self.serial_helpers else _UPSTREAM_HELPER_LOCK
        with helper_context as helper_count:
            try:
                self._upstream.predict_nnunet = self._predict_nnunet
                self._upstream.inference(
                    input_path=request.input_image.absolute(), output_path=output_dir,
                    data_path=self.data_dir, default_release=next(iter(self._model_dirs.values())).parent.name,
                    keep_only=["step2_output"], max_workers=self.max_workers,
                    max_workers_nnunet=1, device=self._device, quiet=True,
                )
            finally:
                self._upstream.predict_nnunet = original
        final_dir = output_dir / "step2_output"
        candidates = sorted(final_dir.glob("*.nii.gz"))
        matching = [path for path in candidates if _nii_stem(path).startswith(_nii_stem(request.input_image))]
        if len(matching) != 1:
            raise RuntimeError(f"expected one TotalSpineSeg output in {final_dir}; found {matching}")
        label_map = matching[0]
        image = nib.load(str(label_map))
        if len(image.shape) != 3:
            raise RuntimeError(f"TotalSpineSeg output must be 3-D, got {image.shape}")
        return SegmentationResult(
            label_map=label_map, view=request.view, model_id=request.model_id,
            backend_id=self.backend_id, backend_version=self._version,
            geometry=SegmentationGeometry(
                shape_ijk=tuple(int(size) for size in image.shape),
                voxel_to_world=tuple(tuple(float(x) for x in row) for row in image.affine),
                coordinate_system="NIFTI_RAS+",
            ),
            label_map_semantics=TSS_LABELS,
            metadata={
                "elapsed_seconds": time.perf_counter() - started,
                "data_dir": str(self.data_dir),
                "model_release_dir": str(next(iter(self._model_dirs.values())).parent),
                "checkpoints": self._checkpoint_info,
                "model_load_timings": self._load_timings,
                "stage_prediction_seconds": self._stage_timings.copy(),
                "keeps_model_resident": True,
                "serial_upstream_helpers": self.serial_helpers,
                "patched_helper_modules": helper_count if self.serial_helpers else 0,
                "requested_structures": list(request.structures),
                "runtime": "installed TotalSpineSeg Python package with cached nnU-Net predictors and serial single-case helpers" if self.serial_helpers else "installed TotalSpineSeg Python package with cached nnU-Net predictors",
            },
        )

    def _close(self) -> None:
        for predictor in self._predictors.values():
            predictor.network.to("cpu")
        self._predictors.clear()
        self._active_model = None
        if self.device_name == "cuda" and hasattr(self, "_torch"):
            self._torch.cuda.empty_cache()
