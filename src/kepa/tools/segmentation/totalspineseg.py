"""Independent adapter for the upstream TotalSpineSeg command-line package.

This intentionally calls ``totalspineseg`` rather than SCT's ``sct_deepseg``.
It does not copy upstream implementation code or redistribute model weights.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import nibabel as nib

from .contract import SegmentationBackend, SegmentationGeometry, SegmentationRequest, SegmentationResult


TSS_LABELS = {
    1: "spinal_cord",
    2: "spinal_canal",
    **{10 + level: f"vertebrae_C{level}" for level in range(1, 8)},
    63: "disc_C2_C3",
    **{level + 61: f"disc_C{level}_C{level + 1}" for level in range(3, 7)},
    71: "disc_C7_T1",
}
TSS_DATASETS = ("Dataset101_TotalSpineSeg_step1", "Dataset102_TotalSpineSeg_step2")


def _nii_stem(path: Path) -> str:
    name = path.name
    return name[:-7] if name.lower().endswith(".nii.gz") else path.stem


def _installed_dataset_dirs(data_dir: Path) -> dict[str, Path]:
    results_root = data_dir / "nnUNet" / "results"
    by_dataset: dict[str, list[Path]] = {}
    for dataset in TSS_DATASETS:
        by_dataset[dataset] = []
        for path in results_root.glob(f"**/{dataset}"):
            if path.is_dir() and any(
                checkpoint.is_file()
                for checkpoint in (*path.rglob("checkpoint_best.pth"), *path.rglob("checkpoint_final.pth"))
            ):
                by_dataset[dataset].append(path)
    shared_parents = set.intersection(
        *(set(path.parent for path in by_dataset[dataset]) for dataset in TSS_DATASETS)
    )
    if not shared_parents:
        missing = [name for name, paths in by_dataset.items() if not paths]
        raise FileNotFoundError(
            f"TotalSpineSeg step-1/step-2 checkpoints are incomplete under {data_dir}; "
            f"missing or unmatched datasets: {missing or list(TSS_DATASETS)}"
        )
    if results_root in shared_parents:
        selected_parent = results_root
    elif len(shared_parents) == 1:
        selected_parent = next(iter(shared_parents))
    else:
        raise RuntimeError(
            "multiple TotalSpineSeg model releases are installed under the selected data directory; "
            "use a data directory containing one compatible step-1/step-2 release"
        )
    return {dataset: selected_parent / dataset for dataset in TSS_DATASETS}


class TotalSpineSegCliBackend(SegmentationBackend):
    """One-case-at-a-time TotalSpineSeg adapter, sagittal MRI supported first."""

    backend_id = "totalspineseg_cli"
    supported_views = frozenset({"sagittal_t2"})
    supported_models = frozenset({"totalspineseg"})
    keeps_model_resident = False

    def __init__(
        self,
        executable: str = "totalspineseg",
        data_dir: Path | None = None,
        *,
        device: str = "cuda",
        max_workers: int = 1,
        no_stalling: bool = False,
    ) -> None:
        super().__init__()
        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.executable = executable
        self.data_dir = Path(data_dir).expanduser().absolute() if data_dir else None
        self.device = device
        self.max_workers = max_workers
        self.no_stalling = no_stalling
        self._resolved_executable: str | None = None

    def _start(self) -> None:
        resolved = shutil.which(self.executable)
        if resolved is None and Path(self.executable).is_file():
            resolved = str(Path(self.executable).absolute())
        if resolved is None:
            raise FileNotFoundError(
                f"cannot find TotalSpineSeg executable {self.executable!r}; "
                "install the upstream package and pass --totalspineseg"
            )
        self._resolved_executable = resolved
        data_dir = self.data_dir or (
            Path(os.environ["TOTALSPINESEG_DATA"]).expanduser().absolute()
            if os.environ.get("TOTALSPINESEG_DATA")
            else None
        )
        if data_dir is None:
            raise RuntimeError(
                "set TOTALSPINESEG_DATA or pass a data directory containing the "
                "officially installed nnUNet weights; KEPA will not download weights implicitly"
            )
        self._model_dirs = _installed_dataset_dirs(data_dir)
        self.data_dir = data_dir

    def _segment(self, request: SegmentationRequest) -> SegmentationResult:
        if request.view != "sagittal_t2":
            raise ValueError("the first TotalSpineSeg adapter currently supports sagittal_t2 only")
        assert self._resolved_executable is not None
        output_dir = request.output_dir.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # TotalSpineSeg itself supports locating checkpoints under
        # nnUNet/results/<release>/Dataset*/. Pass the user's data root as-is;
        # staging release-nested folders behind symlinks is unnecessary and
        # fails on Windows when symlink privileges are unavailable.
        command = [
            self._resolved_executable,
            str(request.input_image.absolute()),
            str(output_dir),
            "--data-dir",
            str(self.data_dir),
            "--device",
            self.device,
            "--max-workers",
            str(self.max_workers),
            "--keep-only",
            "step2_output",
            "--quiet",
        ]
        if self.no_stalling:
            command.append("--no-stalling")
        completed = subprocess.run(
            command,
            cwd=output_dir,
            check=False,
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - started
        stdout_path = output_dir / "totalspineseg.stdout.log"
        stderr_path = output_dir / "totalspineseg.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"TotalSpineSeg failed with exit code {completed.returncode}; see {stderr_path}"
            )

        final_dir = output_dir / "step2_output"
        outputs = sorted(
            path for path in final_dir.rglob("*")
            if path.is_file() and (path.name.lower().endswith(".nii") or path.name.lower().endswith(".nii.gz"))
        ) if final_dir.is_dir() else []
        input_stem = _nii_stem(request.input_image)
        matching = [path for path in outputs if _nii_stem(path).startswith(input_stem)]
        candidates = matching or outputs
        if len(candidates) != 1:
            raise RuntimeError(
                "expected exactly one final TotalSpineSeg label NIfTI in "
                f"{final_dir}, found {len(candidates)}: {[path.name for path in candidates]}"
            )
        output_image = nib.load(str(candidates[0]))
        if len(output_image.shape) != 3:
            raise RuntimeError(f"TotalSpineSeg output must be 3-D, got shape {output_image.shape}")
        geometry = SegmentationGeometry(
            shape_ijk=tuple(int(size) for size in output_image.shape),
            voxel_to_world=tuple(
                tuple(float(value) for value in row) for row in output_image.affine
            ),
            coordinate_system="NIFTI_RAS+",
        )
        return SegmentationResult(
            label_map=candidates[0].absolute(),
            view=request.view,
            model_id=request.model_id,
            backend_id=self.backend_id,
            backend_version="unknown (external CLI version not exposed by this adapter yet)",
            geometry=geometry,
            label_map_semantics=TSS_LABELS,
            command=tuple(command),
            log_paths=(stdout_path, stderr_path),
            metadata={
                "elapsed_seconds": elapsed,
                "data_dir": str(self.data_dir),
                "model_release_dir": str(next(iter(self._model_dirs.values())).parent),
                "keeps_model_resident": self.keeps_model_resident,
                "requested_structures": list(request.structures),
            },
        )
