"""Manifest parsing and GPU fallback helpers."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass(frozen=True)
class BatchCase:
    case_id: str
    sagittal_nii: Path
    axial_niis: tuple[Path, ...]

def load_batch_manifest(path: Path) -> list[BatchCase]:
    """Read and validate all paths before any model is loaded."""
    path = Path(path).expanduser().absolute()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    rows = doc if isinstance(doc, list) else doc.get("cases") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty cases array")
    result, seen = [], set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"case {i} must be an object")
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) or case_id in {".", ".."}:
            raise ValueError(f"case {i} has an invalid case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        inputs = row.get("inputs")
        sag, axl = (inputs.get("sag"), inputs.get("axl")) if isinstance(inputs, dict) else (row.get("sag_nii"), row.get("ax_nii"))
        axial = [axl] if isinstance(axl, str) else axl
        if not isinstance(sag, str) or not sag or not isinstance(axial, list) or not axial or not all(isinstance(x, str) and x for x in axial):
            raise ValueError(f"case {case_id}: expected inputs.sag and inputs.axl path(s)")
        def resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return (candidate if candidate.is_absolute() else path.parent / candidate).resolve()
        sag_path, ax_paths = resolve(sag), tuple(resolve(x) for x in axial)
        missing = [str(x) for x in (sag_path, *ax_paths) if not x.is_file()]
        if missing:
            raise FileNotFoundError(f"case {case_id}: missing input(s): {', '.join(missing)}")
        if len(set(ax_paths)) != len(ax_paths):
            raise ValueError(f"case {case_id}: duplicate axial paths")
        result.append(BatchCase(case_id, sag_path, ax_paths))
    return result

def parse_gpu_ids(value: str) -> tuple[str, ...]:
    ids = tuple(x.strip() for x in value.split(","))
    if not ids or any(not re.fullmatch(r"\d+", x) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("provide unique comma-separated GPU IDs, e.g. --gpus 3,4")
    return ids

def start_backend_with_gpu_fallback(factory: Callable[[str], object], gpu_ids: tuple[str, ...]):
    failures = []
    for gpu_id in gpu_ids:
        backend = factory(gpu_id)
        try:
            backend.start()
            return backend, gpu_id
        except Exception as exc:
            try:
                backend.close()
            except Exception:
                pass
            failures.append(f"GPU {gpu_id}: {type(exc).__name__}: {exc}")
    raise RuntimeError("backend failed on all requested GPUs: " + " | ".join(failures))

def make_paired_manifest(sag_dir: Path, axial_root: Path, output_path: Path) -> Path:
    """Generate a manifest for <sag_dir>/<case>.nii[.gz] plus <axial_root>/<case>/*.nii[.gz]."""
    sag_dir, axial_root = Path(sag_dir).expanduser().resolve(), Path(axial_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().absolute()
    if not sag_dir.is_dir() or not axial_root.is_dir():
        raise NotADirectoryError("both sagittal directory and axial root must exist")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {output_path}")

    def nii_files(folder: Path) -> list[Path]:
        return sorted(
            (item.resolve() for item in folder.iterdir()
             if item.is_file() and (item.name.endswith(".nii") or item.name.endswith(".nii.gz"))),
            key=lambda item: item.name,
        )

    sagittal_files = nii_files(sag_dir)
    if not sagittal_files:
        raise ValueError(f"no sagittal NIfTI files found in {sag_dir}")
    cases = []
    for sag in sagittal_files:
        case_id = sag.name[:-7] if sag.name.endswith(".nii.gz") else sag.stem
        axial_dir = axial_root / case_id
        axial_files = nii_files(axial_dir) if axial_dir.is_dir() else []
        if not axial_files:
            raise FileNotFoundError(f"no axial NIfTIs found under {axial_dir} for {case_id}")
        cases.append({
            "case_id": case_id,
            "inputs": {"sag": str(sag), "axl": [str(item) for item in axial_files]},
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"manifest_version": "1.0", "cases": cases}, indent=2), encoding="utf-8")
    return output_path
