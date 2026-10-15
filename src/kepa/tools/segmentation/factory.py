"""Registry for segmentation implementations behind the shared request/result API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contract import SegmentationBackend
from .axial_nnunet import AxialNnunetBackend
from .totalspineseg import TotalSpineSegCliBackend
from .totalspineseg_python import TotalSpineSegPythonBackend


@dataclass(frozen=True)
class BackendSpec:
    constructor: Callable[..., SegmentationBackend]
    views: frozenset[str]
    models: dict[str, frozenset[str]]


_REGISTRY: dict[str, BackendSpec] = {}


def register_backend(backend_id: str, spec: BackendSpec) -> None:
    if backend_id in _REGISTRY:
        raise ValueError(f"segmentation backend {backend_id!r} is already registered")
    _REGISTRY[backend_id] = spec


def backend_specs() -> dict[str, BackendSpec]:
    return dict(_REGISTRY)


def create_backend(backend_id: str, *, data_dir: Path | None = None,
                   device: str = "cuda", max_workers: int = 1,
                   executable: str = "totalspineseg") -> SegmentationBackend:
    try:
        spec = _REGISTRY[backend_id]
    except KeyError as exc:
        raise ValueError(f"unknown segmentation backend {backend_id!r}") from exc
    if backend_id == "totalspineseg_cli":
        return spec.constructor(executable, data_dir, device=device, max_workers=max_workers)
    return spec.constructor(data_dir, device=device, max_workers=max_workers)


_TSS_STRUCTURES = frozenset({"vertebrae", "intervertebral_discs", "spinal_canal"})
register_backend("totalspineseg_cli", BackendSpec(
    TotalSpineSegCliBackend, frozenset({"sagittal_t2"}), {"totalspineseg": _TSS_STRUCTURES},
))
register_backend("totalspineseg_python", BackendSpec(
    TotalSpineSegPythonBackend, frozenset({"sagittal_t2"}), {"totalspineseg": _TSS_STRUCTURES},
))
register_backend("axial_nnunet_python", BackendSpec(
    AxialNnunetBackend, frozenset({"axial_t2"}), {
        "canal_seg": frozenset({"spinal_canal"}),
        "sc_epi": frozenset({"spinal_cord"}),
    },
))
