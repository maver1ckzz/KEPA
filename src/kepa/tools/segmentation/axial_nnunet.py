# SPDX-FileCopyrightText: 2023 Polytechnique Montreal <www.neuro.polymtl.ca>
# SPDX-FileCopyrightText: 2026 KEPA contributors
# SPDX-License-Identifier: LGPL-3.0-only
# Adapted from SCT 7.1 spinalcordtoolbox/deepseg/inference.py::segment_nnunet
# (credited there to Jan Valosek and Naga Karthik) and
# spinalcordtoolbox/deepseg/nnunet.py::create_nnunet_from_plans.
# The SCT 7.1 release identifies short commit 70883e8; the local reference
# snapshot had no Git metadata, so an exact commit match is unverified.
# Modified by KEPA contributors on 2026-10-16. See THIRD_PARTY_NOTICES.md,
# LICENSE-LGPL-3.0-only, and LICENSE-GPL-3.0-only for provenance and terms.

"""Resident axial nnU-Net models behind KEPA's segmentation contract.

This is a KEPA implementation of the minimal model-loading and orientation
behavior used by SCT 7.1's ``deepseg.nnunet`` and ``deepseg.inference``. It
calls nnU-Net v2 directly and does not import SCT. The external model weights
remain outside this repository. See THIRD_PARTY_NOTICES.md for provenance.
"""

from __future__ import annotations

import hashlib
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage

from .contract import SegmentationBackend, SegmentationGeometry, SegmentationRequest, SegmentationResult


MODEL_DIRS = {
    "canal_seg": "model_seg_canal_t2w",
    "sc_epi": "model_seg_sc_epi_nnunet",
}
MODEL_STRUCTURES = {
    "canal_seg": frozenset({"spinal_canal"}),
    "sc_epi": frozenset({"spinal_cord"}),
}
MODEL_LABELS = {
    "canal_seg": {0: "background", 1: "spinal_canal"},
    "sc_epi": {0: "background", 1: "spinal_cord"},
}

# SCT names directions by the opposite of nibabel's voxel-axis convention.
# For example, SCT's "LPI" means nibabel RAS. Preserve that convention when
# reading the orientation recorded in an upstream model's dataset.json.
_SCT_TO_NIB = str.maketrans("LRAPIS", "RLPASI")


def _model_axis_codes(predictor: Any) -> tuple[str, str, str]:
    orientation = predictor.dataset_json.get("image_orientation")
    if orientation is None:
        dataset_name = predictor.plans_manager.dataset_name
        orientation = "AIL" if "RegionBasedLesionSeg" in dataset_name else "LPI"
    if not isinstance(orientation, str) or len(orientation) != 3:
        raise ValueError(f"invalid model image_orientation: {orientation!r}")
    codes = tuple(orientation.translate(_SCT_TO_NIB))
    nib.orientations.axcodes2ornt(codes)  # reject unknown/duplicate axis codes
    return codes


def _predict_single_npy_array_timed(
    image_array: np.ndarray,
    spacing: tuple[float, float, float],
    predictor: Any,
    stage_seconds: dict[str, float],
) -> np.ndarray:
    """Mirror nnU-Net's single-array path while timing its internal phases."""
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    started = time.perf_counter()
    adapter = PreprocessAdapterFromNpy(
        [image_array], [None], [{"spacing": spacing}], [None],
        predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
        num_threads_in_multithreaded=1, verbose=predictor.verbose,
    )
    item = next(adapter)
    stage_seconds["nnunet_preprocessing"] = time.perf_counter() - started

    started = time.perf_counter()
    logits = predictor.predict_logits_from_preprocessed_data(item["data"]).cpu()
    stage_seconds["sliding_window_network_inference"] = time.perf_counter() - started

    started = time.perf_counter()
    segmentation = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, predictor.plans_manager, predictor.configuration_manager,
        predictor.label_manager, item["data_properties"], return_probabilities=False,
    )
    stage_seconds["nnunet_shape_restore_and_label_conversion"] = time.perf_counter() - started
    return segmentation


def _predict_native_labels(
    source: nib.spatialimages.SpatialImage,
    predictor: Any,
    stage_seconds: dict[str, float] | None = None,
) -> np.ndarray:
    """Run nnU-Net on its model grid and return labels on the input voxel grid."""
    stage_seconds = stage_seconds if stage_seconds is not None else {}
    started = time.perf_counter()
    source_orientation = nib.orientations.io_orientation(source.affine)
    model_orientation = nib.orientations.axcodes2ornt(_model_axis_codes(predictor))
    to_model = nib.orientations.ornt_transform(source_orientation, model_orientation)
    model_image = source.as_reoriented(to_model)
    image_array = np.asarray(model_image.dataobj)
    # nnU-Net's SimpleITK reader uses reversed spatial axes relative to nibabel.
    image_array = np.ascontiguousarray(image_array.transpose(2, 1, 0)[None], dtype=np.float32)
    spacing = tuple(float(value) for value in model_image.header.get_zooms()[:3][::-1])
    stage_seconds["orientation_and_array_preparation"] = time.perf_counter() - started

    prediction = _predict_single_npy_array_timed(image_array, spacing, predictor, stage_seconds)

    started = time.perf_counter()
    model_labels = np.asarray(prediction).transpose(2, 1, 0)
    if model_labels.shape != model_image.shape:
        raise RuntimeError(f"nnU-Net returned shape {model_labels.shape}, expected {model_image.shape}")
    to_source = nib.orientations.ornt_transform(model_orientation, source_orientation)
    native_labels = nib.orientations.apply_orientation(model_labels, to_source)
    if native_labels.shape != source.shape:
        raise RuntimeError(f"native prediction shape {native_labels.shape}, expected {source.shape}")
    stage_seconds["native_grid_restore"] = time.perf_counter() - started
    return native_labels


def _checkpoint_info(model_dir: Path) -> dict[str, str]:
    trainers = sorted(path for path in model_dir.glob("nnUNetTrainer*") if path.is_dir())
    if len(trainers) != 1:
        raise FileNotFoundError(f"expected one nnUNetTrainer* folder in {model_dir}, found {len(trainers)}")
    trainer = trainers[0]
    if not (trainer / "dataset.json").is_file() or not (trainer / "plans.json").is_file():
        raise FileNotFoundError(f"dataset.json or plans.json is missing in {trainer}")
    folds = sorted(path for path in trainer.glob("fold_*") if path.is_dir())
    if not folds:
        raise FileNotFoundError(f"no fold_* checkpoint folder in {trainer}")
    checkpoint = "checkpoint_final.pth" if all((fold / "checkpoint_final.pth").is_file() for fold in folds) else "checkpoint_best.pth"
    if not all((fold / checkpoint).is_file() for fold in folds):
        raise FileNotFoundError(f"incomplete {checkpoint} across folds in {trainer}")
    digest = hashlib.sha256()
    for fold in folds:
        with (fold / checkpoint).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "trainer_dir": str(trainer),
        "checkpoint": checkpoint,
        "folds": ",".join(path.name for path in folds),
        "checkpoints_sha256": digest.hexdigest(),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return components == int(np.argmax(sizes))


class AxialNnunetBackend(SegmentationBackend):
    """Load the installed canal and/or EPISeg checkpoints once per worker."""

    backend_id = "axial_nnunet_python"
    supported_views = frozenset({"axial_t2"})
    supported_models = frozenset(MODEL_DIRS)
    keeps_model_resident = True

    def __init__(self, data_dir: Path | None = None, *, device: str = "cuda", max_workers: int = 1) -> None:
        super().__init__()
        if data_dir is None:
            raise ValueError("axial nnU-Net backend requires a model-data directory")
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if max_workers != 1:
            raise ValueError("axial nnU-Net backend accepts one serial request at a time (max_workers=1)")
        self.data_dir = Path(data_dir).expanduser().absolute()
        self.device_name = device
        self._predictors: dict[str, Any] = {}
        self._models: dict[str, Path] = {}
        self._checkpoints: dict[str, dict[str, str]] = {}
        self._load_seconds: dict[str, float] = {}
        self._active_model: str | None = None

    def _start(self) -> None:
        started = time.perf_counter()
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in the model runtime")
        self._torch = torch
        for model_id, dirname in MODEL_DIRS.items():
            model_dir = self.data_dir / dirname
            if model_dir.is_dir() and any(model_dir.glob("nnUNetTrainer*")):
                self._models[model_id] = model_dir
                self._checkpoints[model_id] = _checkpoint_info(model_dir)
        if not self._models:
            raise FileNotFoundError(
                f"no axial checkpoint found under {self.data_dir}; expected one or both of {list(MODEL_DIRS.values())}"
            )
        try:
            for model_id, model_dir in self._models.items():
                # Keep both weights in RAM; activate only the requested model on GPU.
                load_started = time.perf_counter()
                checkpoint_info = self._checkpoints[model_id]
                fold_names = checkpoint_info["folds"].split(",")
                folds = "all" if fold_names == ["fold_all"] else [
                    int(name.removeprefix("fold_")) for name in fold_names
                ]
                predictor = nnUNetPredictor(
                    tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                    perform_everything_on_device=False, device=torch.device("cpu"),
                    verbose=False, verbose_preprocessing=False, allow_tqdm=False,
                )
                predictor.initialize_from_trained_model_folder(
                    checkpoint_info["trainer_dir"], use_folds=folds,
                    checkpoint_name=checkpoint_info["checkpoint"],
                )
                self._predictors[model_id] = predictor
                self._load_seconds[model_id] = time.perf_counter() - load_started
                labels = predictor.dataset_json["labels"]
                expected = "canal" if model_id == "canal_seg" else "bold"
                if labels.get(expected) != 1:
                    raise RuntimeError(f"{model_id} checkpoint has unexpected labels: {labels}")
        except Exception:
            self._close()
            raise
        self._load_seconds["startup_total"] = time.perf_counter() - started

    def _activate(self, model_id: str) -> Any:
        predictor = self._predictors[model_id]
        if self._active_model != model_id:
            if self._active_model is not None:
                previous = self._predictors[self._active_model]
                previous.network.to("cpu")
                previous.device = self._torch.device("cpu")
            predictor.network.to(self.device_name)
            predictor.device = self._torch.device(self.device_name)
            self._active_model = model_id
        return predictor

    def preflight_device(self) -> None:
        """Check that each axial checkpoint can be placed on the selected GPU."""
        if self.device_name != "cuda":
            return
        for model_id in self._models:
            self._activate(model_id)
        if self._active_model is not None:
            active = self._predictors[self._active_model]
            active.network.to("cpu")
            active.device = self._torch.device("cpu")
        self._torch.cuda.empty_cache()

    def _segment(self, request: SegmentationRequest) -> SegmentationResult:
        if request.model_id not in self._models:
            raise FileNotFoundError(f"checkpoint for {request.model_id!r} is not installed under {self.data_dir}")
        if not set(request.structures).issubset(MODEL_STRUCTURES[request.model_id]):
            raise ValueError(f"{request.model_id} can segment only {sorted(MODEL_STRUCTURES[request.model_id])}")
        if any(request.output_dir.iterdir()):
            raise FileExistsError(f"output directory must be empty: {request.output_dir}")
        started = time.perf_counter()
        stage_seconds: dict[str, float] = {}
        phase_started = time.perf_counter()
        predictor = self._activate(request.model_id)
        stage_seconds["model_activation_and_device_transfer"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        source = nib.load(str(request.input_image))
        if len(source.shape) != 3:
            raise ValueError(f"{request.model_id} requires one 3-D NIfTI, got {source.shape}")
        stage_seconds["read_nifti"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        mask = _predict_native_labels(source, predictor, stage_seconds) > 0
        stage_seconds["prediction_and_grid_restore"] = time.perf_counter() - phase_started
        if request.model_id == "canal_seg":
            phase_started = time.perf_counter()
            mask = _largest_component(mask)
            stage_seconds["largest_component_filter"] = time.perf_counter() - phase_started
        header = source.header.copy()
        header.set_data_dtype(np.uint8)
        label_path = request.output_dir / f"{request.model_id}_labels.nii.gz"
        phase_started = time.perf_counter()
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), source.affine, header), str(label_path))
        stage_seconds["write_nifti"] = time.perf_counter() - phase_started
        return SegmentationResult(
            label_map=label_path, view=request.view, model_id=request.model_id,
            backend_id=self.backend_id, backend_version=f"KEPA-nnUNetv2-{version('nnunetv2')}",
            geometry=SegmentationGeometry(
                tuple(int(size) for size in source.shape),
                tuple(tuple(float(value) for value in row) for row in source.affine),
                "NIFTI_RAS+",
            ),
            label_map_semantics=MODEL_LABELS[request.model_id],
            metadata={
                "elapsed_seconds": time.perf_counter() - started,
                "stage_seconds": stage_seconds,
                "model_dir": str(self._models[request.model_id]),
                "checkpoint": self._checkpoints[request.model_id],
                "model_load_seconds": self._load_seconds,
                "keeps_model_resident": True,
                "predictor_api": "KEPA native-orientation adapter with nnUNetv2.predict_single_npy_array",
                "model_orientation_nibabel": "".join(_model_axis_codes(predictor)),
                "nearest_neighbor_resampled_to_native": False,
                "largest_component_only": request.model_id == "canal_seg",
                "domain_warning": "EPISeg is trained for gradient-echo EPI, not anatomical axial T2 MRI" if request.model_id == "sc_epi" else None,
            },
        )

    def _close(self) -> None:
        for predictor in self._predictors.values():
            predictor.network.to("cpu")
        self._predictors.clear()
        self._active_model = None
        if self.device_name == "cuda" and hasattr(self, "_torch"):
            self._torch.cuda.empty_cache()
