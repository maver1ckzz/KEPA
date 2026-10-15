"""Use a long-lived model Python process from a lightweight KEPA runtime."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .contract import SegmentationBackend, SegmentationGeometry, SegmentationRequest, SegmentationResult
from .factory import backend_specs


class PersistentPythonBackend(SegmentationBackend):
    """Transport for any registered model backend; one worker per backend lifetime."""

    backend_id = "persistent_python"
    supported_views = frozenset()
    supported_models = frozenset()
    keeps_model_resident = True

    def __init__(self, python_executable: Path, data_dir: Path, *, model_backend: str = "totalspineseg_python",
                 device: str = "cuda", visible_gpu: str | None = None,
                 max_workers: int = 1, log_path: Path | None = None) -> None:
        super().__init__()
        self.python_executable = Path(python_executable).absolute()
        self.data_dir = Path(data_dir).absolute()
        self.model_backend = model_backend
        spec = backend_specs().get(model_backend)
        if spec is None:
            raise ValueError(f"model backend {model_backend!r} is not registered")
        self.supported_views = spec.views
        self.supported_models = frozenset(spec.models)
        self.device = device
        self.visible_gpu = str(visible_gpu) if visible_gpu is not None else None
        self.max_workers = max_workers
        self.log_path = log_path

    def _receive(self) -> dict:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if line.startswith("KEPA_RESPONSE\t"):
                value = json.loads(line.split("\t", 1)[1])
                if not value.get("ok"):
                    raise RuntimeError(f"model worker failed: {value.get('error')}\n{value.get('traceback', '')}")
                return value
        raise RuntimeError(f"model worker exited without response (code {self._process.poll()}); see {self.log_path}")

    def _send(self, value: dict) -> dict:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(value) + "\n")
        self._process.stdin.flush()
        return self._receive()

    def _start(self) -> None:
        if not self.python_executable.is_file():
            raise FileNotFoundError(f"model Python does not exist: {self.python_executable}")
        if self.log_path is None:
            raise ValueError("log_path is required for a persistent model worker")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        source_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(source_root), env.get("PYTHONPATH")]))
        if self.visible_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.visible_gpu
        try:
            self._process = subprocess.Popen(
                [str(self.python_executable), "-u", "-m", "kepa.tools.segmentation.worker",
                 "--backend", self.model_backend, "--data-dir", str(self.data_dir),
                 "--device", self.device, "--max-workers", str(self.max_workers)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                text=True, bufsize=1, env=env,
            )
            self._receive()
        except Exception:
            if hasattr(self, "_process"):
                self._process.kill()
                self._process.wait()
            self._log.close()
            raise

    def _segment(self, request: SegmentationRequest) -> SegmentationResult:
        response = self._send({"action": "segment", "request": {
            "input_image": str(request.input_image.absolute()),
            "output_dir": str(request.output_dir.absolute()),
            "view": request.view, "structures": list(request.structures),
            "model_id": request.model_id, "case_id": request.case_id,
            "options": request.options,
        }})
        value = response["result"]
        geometry = value["geometry"]
        return SegmentationResult(
            label_map=Path(value["label_map"]), view=value["view"], model_id=value["model_id"],
            backend_id=value["backend_id"], backend_version=value["backend_version"],
            geometry=SegmentationGeometry(
                tuple(geometry["shape_ijk"]), tuple(tuple(row) for row in geometry["voxel_to_world"]),
                geometry["coordinate_system"],
            ),
            label_map_semantics={int(k): v for k, v in value["label_map_semantics"].items()},
            command=tuple(value["command"]),
            log_paths=(self.log_path, *[Path(path) for path in value["log_paths"]]),
            metadata={**value["metadata"], "worker_python": str(self.python_executable)},
        )

    def _close(self) -> None:
        try:
            if self._process.poll() is None:
                self._send({"action": "close"})
                self._process.wait(timeout=15)
        finally:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()
            self._log.close()
