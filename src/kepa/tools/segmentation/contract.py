"""Stable request/result and lifecycle contracts for segmentation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Literal


View = Literal["sagittal_t2", "axial_t2"]


@dataclass(frozen=True)
class SegmentationGeometry:
    """Grid metadata for a 3-D label artifact; affine maps voxel IJK to world mm."""

    shape_ijk: tuple[int, int, int]
    voxel_to_world: tuple[tuple[float, float, float, float], ...]
    coordinate_system: str

    def __post_init__(self) -> None:
        if len(self.shape_ijk) != 3 or any(size <= 0 for size in self.shape_ijk):
            raise ValueError("segmentation geometry shape_ijk must contain three positive dimensions")
        if len(self.voxel_to_world) != 4 or any(len(row) != 4 for row in self.voxel_to_world):
            raise ValueError("voxel_to_world must be a 4x4 affine matrix")
        if not all(math.isfinite(value) for row in self.voxel_to_world for value in row):
            raise ValueError("voxel_to_world affine must contain only finite values")
        if not self.coordinate_system:
            raise ValueError("coordinate_system must be named")


@dataclass(frozen=True)
class SegmentationRequest:
    """One segmentation call for one case/series in a workflow step."""

    input_image: Path
    output_dir: Path
    view: View
    structures: tuple[str, ...]
    model_id: str
    case_id: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.structures:
            raise ValueError("segmentation request must name at least one structure")
        if len(set(self.structures)) != len(self.structures):
            raise ValueError("segmentation request structures must be unique")


@dataclass(frozen=True)
class SegmentationResult:
    """A label-map artifact plus the geometry and provenance needed downstream."""

    label_map: Path
    view: View
    model_id: str
    backend_id: str
    backend_version: str
    geometry: SegmentationGeometry
    label_map_semantics: dict[int, str]
    command: tuple[str, ...] = ()
    log_paths: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SegmentationBackend(ABC):
    """Lifecycle: start once per case/workflow, segment serially, then close.

    Backends may keep weights and a model resident between ``segment`` calls.
    A subprocess adapter can use the same contract, while explicitly reporting
    that its model is not resident so an optimized in-process backend can later
    replace it without changing workflow-facing artifacts.
    """

    backend_id: str
    supported_views: frozenset[str]
    supported_models: frozenset[str]
    keeps_model_resident: bool = False

    def __init__(self) -> None:
        self._started = False

    @abstractmethod
    def _start(self) -> None:
        """Preflight dependencies/weights and prepare reusable resources."""

    @abstractmethod
    def _segment(self, request: SegmentationRequest) -> SegmentationResult:
        """Run a single case, without taking ownership of workflow scheduling."""

    def _close(self) -> None:
        """Release resources; override for model/process cleanup."""

    def start(self) -> "SegmentationBackend":
        if not self._started:
            self._start()
            self._started = True
        return self

    @property
    def is_started(self) -> bool:
        return self._started

    def segment(self, request: SegmentationRequest) -> SegmentationResult:
        if request.view not in self.supported_views:
            raise ValueError(
                f"backend {self.backend_id!r} does not support view {request.view!r}; "
                f"supported: {sorted(self.supported_views)}"
            )
        if request.model_id not in self.supported_models:
            raise ValueError(
                f"backend {self.backend_id!r} does not provide model {request.model_id!r}"
            )
        if not self._started:
            raise RuntimeError(f"backend {self.backend_id!r} must be started before segment()")
        if not request.input_image.is_file():
            raise FileNotFoundError(f"segmentation input does not exist: {request.input_image}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        result = self._segment(request)
        if not result.label_map.is_file():
            raise RuntimeError(f"backend returned a missing label map: {result.label_map}")
        if result.view != request.view or result.model_id != request.model_id:
            raise RuntimeError("backend result view/model_id does not match the request")
        return result

    def close(self) -> None:
        if self._started:
            try:
                self._close()
            finally:
                self._started = False

    def __enter__(self) -> "SegmentationBackend":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
