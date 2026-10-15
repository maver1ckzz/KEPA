"""JSON-driven execution for paired sagittal/axial NIfTI workflows."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from nibabel.processing import resample_from_to

from kepa.tools.matching.tra_map import (
    LevelSliceMatch,
    match_anatomical_levels,
    match_anatomical_levels_soft_distance,
)
from kepa.tools.roi_extraction.geometry import (
    composite_bbox_with_provenance,
    keep_anterior_components,
    mask_bbox,
    point_bbox,
)
from kepa.tools.segmentation import (
    SegmentationBackend,
    SegmentationRequest,
    TotalSpineSegCliBackend,
    TotalSpineSegBackend,
    PersistentPythonBackend,
)
from kepa.tools.segmentation.postprocessing import keep_center_nearest_components
from kepa.workflow.validator import normalize_workflow, validate_workflow


@dataclass(frozen=True)
class SeriesVolume:
    key: str
    view: str
    path: Path
    data: np.ndarray
    affine: np.ndarray
    slice_axis: int
    view_deviation_deg: float


@dataclass(frozen=True)
class SegmentationArtifact:
    labels: np.ndarray
    affine: np.ndarray
    label_path: Path
    canal_probability_path: Path | None
    series: SeriesVolume


@dataclass(frozen=True)
class SelectedSlice:
    series: SeriesVolume
    index: int | None
    selector: str
    area: int
    area_by_index: tuple[int, ...]
    label_ids: tuple[int, ...]
    reason: str | None = None


@dataclass(frozen=True)
class SelectedSliceSet:
    series: SeriesVolume
    selections: dict[str, SelectedSlice]
    json_path: Path


@dataclass(frozen=True)
class LevelMatches:
    series: SeriesVolume
    matches: dict[str, LevelSliceMatch]
    json_path: Path | None = None


@dataclass(frozen=True)
class RoiBundleArtifact:
    rois: list[dict[str, Any]]
    json_path: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_component(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{name} must contain only letters, digits, dot, underscore, and hyphen")
    return value


def _safe_filename(value: str, name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"{name} must be a simple filename component")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plane_normal(affine: np.ndarray, slice_axis: int) -> np.ndarray:
    in_plane = [axis for axis in range(3) if axis != slice_axis]
    normal = np.cross(affine[:3, in_plane[0]], affine[:3, in_plane[1]])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("NIfTI affine produces a degenerate slice plane")
    return normal / norm


def _slice_axis(affine: np.ndarray, target_normal: tuple[float, float, float]) -> tuple[int, float]:
    target = np.asarray(target_normal, dtype=np.float64)
    target /= np.linalg.norm(target)
    alignments = [abs(float(np.dot(_plane_normal(affine, axis), target))) for axis in range(3)]
    axis = int(np.argmax(alignments))
    angle = math.degrees(math.acos(min(1.0, max(-1.0, alignments[axis]))))
    return axis, angle

def _ordered_axial_planes(sagittal_path: Path, axial_paths: list[Path]) -> list[dict[str, Any]]:
    """Order real axial planes by intersection height on the mid-sagittal plane.

    This is a virtual order: each source NIfTI keeps its native affine and local
    slice index, so differently angled volumes are never concatenated.
    """
    sagittal = nib.load(str(sagittal_path))
    sag_affine = np.asarray(sagittal.affine, dtype=np.float64)
    sag_shape = tuple(int(value) for value in sagittal.shape)
    sag_slice_axis, _ = _slice_axis(sag_affine, (1.0, 0.0, 0.0))
    in_plane = [axis for axis in range(3) if axis != sag_slice_axis]
    horizontal_target = np.asarray((0.0, -1.0, 0.0))
    horizontal_axis = max(
        in_plane,
        key=lambda axis: abs(float(np.dot(
            sag_affine[:3, axis] / np.linalg.norm(sag_affine[:3, axis]),
            horizontal_target,
        ))),
    )
    vertical_axis = next(axis for axis in in_plane if axis != horizontal_axis)
    sag_index = sag_shape[sag_slice_axis] // 2
    horizontal_index = (sag_shape[horizontal_axis] - 1.0) / 2.0
    base_voxel = np.zeros(4, dtype=np.float64)
    base_voxel[3] = 1.0
    base_voxel[sag_slice_axis] = sag_index
    base_voxel[horizontal_axis] = horizontal_index
    base_world = (sag_affine @ base_voxel)[:3]
    vertical_step = sag_affine[:3, vertical_axis]

    reference_axial = nib.load(str(axial_paths[0]))
    reference_slice_axis, _ = _slice_axis(np.asarray(reference_axial.affine, dtype=np.float64), (0.0, 0.0, 1.0))
    reference_order_direction = np.asarray(reference_axial.affine[:3, reference_slice_axis], dtype=np.float64)
    reference_order_direction /= np.linalg.norm(reference_order_direction)

    planes: list[dict[str, Any]] = []
    for path in axial_paths:
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError(f"axial NIfTI must be 3D: {path} has shape {image.shape}")
        affine = np.asarray(image.affine, dtype=np.float64)
        slice_axis, _ = _slice_axis(affine, (0.0, 0.0, 1.0))
        normal = _plane_normal(affine, slice_axis)
        for local_index in range(int(image.shape[slice_axis])):
            voxel = (np.asarray(image.shape, dtype=np.float64) - 1.0) / 2.0
            voxel[slice_axis] = local_index
            origin = (affine @ np.r_[voxel, 1.0])[:3]
            denominator = float(np.dot(normal, vertical_step))
            if abs(denominator) < 1e-10:
                superior_mm = float(origin[2])
            else:
                vertical_index = -float(np.dot(normal, base_world - origin)) / denominator
                sag_voxel = np.zeros(4, dtype=np.float64)
                sag_voxel[3] = 1.0
                sag_voxel[sag_slice_axis] = sag_index
                sag_voxel[horizontal_axis] = horizontal_index
                sag_voxel[vertical_axis] = vertical_index
                superior_mm = float((sag_affine @ sag_voxel)[2])
            planes.append({
                "source_nifti": str(path.absolute()),
                "nifti_file": path.name,
                "slice_index_0based": local_index,
                "superior_ras_mm": superior_mm,
                "slice_order_coordinate_mm": float(np.dot(origin, reference_order_direction)),
                "plane_origin_ras": origin.tolist(),
                "plane_normal_ras": normal.tolist(),
            })
    planes.sort(key=lambda item: (
        item["slice_order_coordinate_mm"],
        item["superior_ras_mm"],
        tuple(item["plane_origin_ras"]),
        tuple(item["plane_normal_ras"]),
    ))
    for rank, plane in enumerate(planes, 1):
        plane["global_order_1based"] = rank
    return planes


def _load_series(path: Path, key: str, view: str) -> SeriesVolume:
    if not path.is_file():
        raise FileNotFoundError(f"input NIfTI does not exist: {path}")
    image = nib.load(str(path))
    if len(image.shape) != 3:
        raise ValueError(f"{view} input must be a 3D NIfTI, got shape {image.shape}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).any() or float(np.nanmax(data) - np.nanmin(data)) <= 0:
        raise ValueError(f"{view} input is empty or has no intensity range")
    affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(affine).all() or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-10:
        raise ValueError(f"{view} input has invalid voxel-to-world geometry")
    target = (1.0, 0.0, 0.0) if view == "sagittal_t2" else (0.0, 0.0, 1.0)
    axis, angle = _slice_axis(affine, target)
    return SeriesVolume(key, view, path.absolute(), data, affine, axis, angle)


def _load_segmentation(path: Path, series: SeriesVolume) -> SegmentationArtifact:
    image = nib.load(str(path))
    if tuple(image.shape) != tuple(series.data.shape) or not np.allclose(image.affine, series.affine, atol=1e-4):
        image = resample_from_to(image, (series.data.shape, series.affine), order=0)
    labels = np.rint(np.asarray(image.dataobj)).astype(np.int16)
    if labels.ndim != 3 or not np.any(labels):
        raise ValueError("segmentation backend produced an empty or non-3D label map")
    canal_candidates = sorted(path.parent.glob("*_step1_canal.nii*"))
    return SegmentationArtifact(
        labels=labels,
        affine=np.asarray(image.affine, dtype=np.float64),
        label_path=path,
        canal_probability_path=canal_candidates[0] if canal_candidates else None,
        series=series,
    )


def _orient_plane(series: SeriesVolume, index: int, volume: np.ndarray) -> np.ndarray:
    plane = np.take(volume, index, axis=series.slice_axis)
    voxel_axes = [axis for axis in range(3) if axis != series.slice_axis]
    horizontal_target = np.array((0.0, -1.0, 0.0) if series.view == "sagittal_t2" else (-1.0, 0.0, 0.0))
    vertical_target = np.array((0.0, 0.0, 1.0) if series.view == "sagittal_t2" else (0.0, 1.0, 0.0))
    directions = [series.affine[:3, axis] / np.linalg.norm(series.affine[:3, axis]) for axis in voxel_axes]
    horizontal_slot = int(np.argmax([abs(float(np.dot(direction, horizontal_target))) for direction in directions]))
    vertical_slot = 1 - horizontal_slot
    if horizontal_slot == 0:
        plane = plane.T
        horizontal_axis = voxel_axes[0]
        vertical_axis = voxel_axes[1]
    else:
        horizontal_axis = voxel_axes[1]
        vertical_axis = voxel_axes[0]
    horizontal_direction = series.affine[:3, horizontal_axis]
    vertical_direction = series.affine[:3, vertical_axis]
    if float(np.dot(horizontal_direction, horizontal_target)) < 0:
        plane = np.fliplr(plane)
    # Image rows increase downward. If the voxel axis points toward the
    # requested anatomical-up direction, reverse the rows so superior/anterior
    # appears at the top of the displayed image.
    if float(np.dot(vertical_direction, vertical_target)) > 0:
        plane = np.flipud(plane)
    return np.asarray(plane)


def _restore_display_plane(series: SeriesVolume, display_plane: np.ndarray) -> np.ndarray:
    """Invert _orient_plane's display transform to native in-plane voxel order."""
    plane = np.asarray(display_plane)
    voxel_axes = [axis for axis in range(3) if axis != series.slice_axis]
    horizontal_target = np.array((0.0, -1.0, 0.0) if series.view == "sagittal_t2" else (-1.0, 0.0, 0.0))
    vertical_target = np.array((0.0, 0.0, 1.0) if series.view == "sagittal_t2" else (0.0, 1.0, 0.0))
    directions = [series.affine[:3, axis] / np.linalg.norm(series.affine[:3, axis]) for axis in voxel_axes]
    horizontal_slot = int(np.argmax([abs(float(np.dot(direction, horizontal_target))) for direction in directions]))
    horizontal_axis = voxel_axes[horizontal_slot]
    vertical_axis = voxel_axes[1 - horizontal_slot]
    if float(np.dot(series.affine[:3, vertical_axis], vertical_target)) > 0:
        plane = np.flipud(plane)
    if float(np.dot(series.affine[:3, horizontal_axis], horizontal_target)) < 0:
        plane = np.fliplr(plane)
    if horizontal_slot == 0:
        plane = plane.T
    return np.asarray(plane)


def _display_plane(series: SeriesVolume, index: int) -> np.ndarray:
    plane = _orient_plane(series, index, series.data)
    finite = plane[np.isfinite(plane)]
    low, high = np.percentile(finite, [1, 99])
    if high <= low:
        high = low + 1.0
    return np.asarray(np.clip((plane - low) / (high - low), 0, 1) * 255, dtype=np.uint8)


def _save_plane(series: SeriesVolume, index: int, path: Path, label: str | None = None) -> None:
    pixels = _display_plane(series, index)
    image = Image.fromarray(pixels).convert("RGB")
    if label:
        draw = ImageDraw.Draw(image)
        draw.rectangle((4, 4, min(image.width - 4, 190), 25), fill=(0, 0, 0))
        draw.text((8, 7), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _roi_level_name(region: str) -> str:
    """Map slash-style anatomical aliases to compact ROI-preset keys."""
    if "/" not in region:
        return region
    upper, lower = region.split("/", maxsplit=1)
    if upper and lower and upper[0] == lower[0]:
        lower = lower[1:]
    return f"{upper}-{lower}"


def _public_region_name(level: str) -> str:
    """Return the public slash form for compact preset keys such as C3-4."""
    if "-" not in level:
        return level
    upper, lower = level.split("-", maxsplit=1)
    if upper and lower and lower[0].isdigit() and upper[0] in "CTLS":
        lower = f"{upper[0]}{lower}"
    return f"{upper}/{lower}"


def _selection_indices(selection: SelectedSlice | SelectedSliceSet | LevelMatches) -> list[tuple[str | None, int]]:
    if isinstance(selection, SelectedSlice):
        return [(None, selection.index)]
    if isinstance(selection, SelectedSliceSet):
        return [(_roi_level_name(region), selected.index) for region, selected in selection.selections.items()]
    if isinstance(selection, LevelMatches):
        return [(level, match.slice_index) for level, match in selection.matches.items()]
    raise TypeError(
        "ROI operation expected ImageSlice, SliceMatchSet, or SliceSelectionSet, "
        f"got {type(selection).__name__}"
    )


def _is_region_slice_selection(selection: Any) -> bool:
    """Whether a selection contains one independently selected slice per region."""
    return isinstance(selection, (SelectedSliceSet, LevelMatches))


def _draw_roi_overlay(
    series: SeriesVolume,
    index: int,
    rois: list[dict[str, Any]],
    path: Path,
) -> None:
    image = Image.fromarray(_display_plane(series, index)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for roi in rois:
        left, top, right, bottom = roi["bbox_xyxy"]
        # The ROI geometry is half-open pixel-edge coordinates; drawing endpoints are inclusive.
        draw.rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), outline=(70, 255, 70), width=2)
        tag = str(roi["level"])
        draw.rectangle((left, top, min(image.width - 1, left + max(18, len(tag) * 6)), min(image.height - 1, top + 10)), fill=(0, 90, 0))
        draw.text((left + 2, top + 1), tag, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _roi_record(
    *, level: str, view: str, index: int, bbox: tuple[int, int, int, int], image_shape: tuple[int, int],
    rule: str, label_ids: list[int] | None = None, boundary_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "level": level,
        "view": view,
        "status": "selected",
        "reason": None,
        "slice_index_0based": index,
        "bbox_xyxy": list(bbox),
        "bbox_convention": "display-oriented image pixel edges; xyxy; half-open",
        "image_size_wh": [int(image_shape[1]), int(image_shape[0])],
        "rule": rule,
        "label_ids": label_ids or [],
    }
    if boundary_resolution is not None:
        record["boundary_resolution"] = boundary_resolution
    return record


def _missing_roi_record(level: str, view: str, reason: str, *, rule: str, index: int | None = None) -> dict[str, Any]:
    return {
        "level": level,
        "view": view,
        "status": "missing",
        "reason": reason,
        "slice_index_0based": index,
        "bbox_xyxy": None,
        "bbox_convention": "display-oriented image pixel edges; xyxy; half-open",
        "image_size_wh": None,
        "rule": rule,
        "label_ids": [],
    }


def _artifact(path: Path, run_dir: Path, artifact_type: str, **metadata: Any) -> dict[str, Any]:
    relative = path.relative_to(run_dir)
    return {
        "type": artifact_type,
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "metadata": metadata,
    }


def _plane_metadata(series: SeriesVolume, index: int) -> dict[str, Any]:
    center = (np.asarray(series.data.shape, dtype=np.float64) - 1.0) / 2.0
    center[series.slice_axis] = index
    world = (series.affine @ np.r_[center, 1.0])[:3]
    normal = _plane_normal(series.affine, series.slice_axis)
    # DICOM ImagePositionPatient uses LPS. NIfTI world coordinates use RAS+.
    ras_to_lps = np.diag((-1.0, -1.0, 1.0))
    return {
        "slice_index_0based": index,
        "slice_axis": series.slice_axis,
        "plane_center_ras_mm": [float(value) for value in world],
        "plane_center_lps_mm": [float(value) for value in ras_to_lps @ world],
        "plane_normal_ras": [float(value) for value in normal],
        "plane_normal_lps": [float(value) for value in ras_to_lps @ normal],
        "coordinate_system": "NIFTI_RAS+",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve(ref: str, inputs: dict[str, Any], outputs: dict[str, dict[str, Any]]) -> Any:
    if ref in inputs:
        return inputs[ref]
    match = re.fullmatch(r"\$steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)", ref)
    if not match:
        raise ValueError(f"unsupported reference {ref!r}")
    step_id, output_name = match.groups()
    return outputs[step_id][output_name]


def _tra2sag_matches(source, target, segmentation, source_slices, options):
    side = options.get("side", "left")
    if side not in {"left", "right"} or source.view != "axial_t2" or target.view != "sagittal_t2":
        raise ValueError("tra2sag requires axial_t2 -> sagittal_t2 and side left/right")
    rule_map = {r: (float(q["alpha"]), float(q["beta"]), int(q["outward_shift"])) for q in options["point_rules"] for r in q["regions"]}
    requested = options.get("requested_regions") or list(source_slices.matches)
    normal = _plane_normal(target.affine, target.slice_axis)
    mid = (np.asarray(target.data.shape,dtype=float)-1)/2
    origins=[]
    for i in range(target.data.shape[target.slice_axis]):
        v=mid.copy(); v[target.slice_axis]=i; origins.append((target.affine@np.r_[v,1.])[:3])
    origins=np.asarray(origins); pos=origins@normal; physical_order=np.argsort(pos)
    def point_world(series,index,x,y):
        axes=[a for a in range(3) if a!=series.slice_axis]
        ht,vt=((np.array((-1.,0.,0.)),np.array((0.,1.,0.))) if series.view=="axial_t2" else (np.array((0.,-1.,0.)),np.array((0.,0.,1.))))
        dirs=[series.affine[:3,a]/np.linalg.norm(series.affine[:3,a]) for a in axes]
        ha=axes[int(np.argmax([abs(float(np.dot(d,ht))) for d in dirs]))]; va=next(a for a in axes if a!=ha)
        if np.dot(series.affine[:3,ha],ht)<0: x=series.data.shape[ha]-1-x
        if np.dot(series.affine[:3,va],vt)>0: y=series.data.shape[va]-1-y
        v=(np.asarray(series.data.shape,dtype=float)-1)/2; v[series.slice_axis]=index; v[ha]=x; v[va]=y
        return (series.affine@np.r_[v,1.])[:3]
    def sag_xy(index,world):
        v=(np.linalg.inv(target.affine)@np.r_[world,1.])[:3]; axes=[a for a in range(3) if a!=target.slice_axis]
        ht=np.array((0.,-1.,0.)); vt=np.array((0.,0.,1.)); dirs=[target.affine[:3,a]/np.linalg.norm(target.affine[:3,a]) for a in axes]
        ha=axes[int(np.argmax([abs(float(np.dot(d,ht))) for d in dirs]))]; va=next(a for a in axes if a!=ha)
        x=target.data.shape[ha]-1-v[ha] if np.dot(target.affine[:3,ha],ht)<0 else v[ha]
        y=target.data.shape[va]-1-v[va] if np.dot(target.affine[:3,va],vt)>0 else v[va]
        return [float(x),float(y)]
    out={}
    for region in requested:
        sm=source_slices.matches.get(region); rule=rule_map.get(region)
        if sm is None or sm.status!="selected" or sm.slice_index is None or rule is None:
            out[region]=LevelSliceMatch(region,None,"tra2sag_bbox_point",None,(),{},status="missing",reason="missing source match or point rule"); continue
        yy,xx=np.nonzero(_orient_plane(source,sm.slice_index,segmentation.labels!=0))
        if not len(xx):
            out[region]=LevelSliceMatch(region,None,"tra2sag_bbox_point",None,(),{},status="missing",reason="empty canal mask"); continue
        x1,x2,y1,y2=map(float,(xx.min(),xx.max()+1,yy.min(),yy.max()+1)); a,b,shift=rule
        x=x2-a*(x2-x1) if side=="left" else x1+a*(x2-x1); y=y1+b*(y2-y1)
        p=point_world(source,sm.slice_index,x,y); c=point_world(source,sm.slice_index,(x1+x2)/2,(y1+y2)/2)
        pp,cp=float(p@normal),float(c@normal); base=int(np.argmin(np.abs(pos-pp)))
        direction=(1 if pp>=cp else -1) if abs(pp-cp)>1e-8 else (1 if side=="left" else -1)
        rank=int(np.where(physical_order==base)[0][0]); chosen=int(physical_order[min(max(rank+direction*shift,0),len(physical_order)-1)])
        dist=float(abs(pos[base]-pp))
        out[region]=LevelSliceMatch(region,chosen,"tra2sag_bbox_point",float(math.exp(-.5*(dist/float(options.get("sigma_mm",3)))**2)),(),{"source_axial_slice_index_0based":sm.slice_index,"source_bbox_xyxy":[x1,y1,x2,y2],"source_point_xy":[x,y],"source_point_ras_mm":p.tolist(),"base_sagittal_slice_index_0based":base,"outward_direction":direction,"outward_shift":shift,"target_point_xy":sag_xy(chosen,p),"point_to_base_plane_distance_mm":dist})
    return out


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_json(path, manifest)




def _build_multi_axial_nifti_workflow(
    *, workflow_path: Path, sagittal_nii: Path, axial_niis: list[Path], runs_root: Path,
    case_id: str | None, run_id: str | None, sct_executable: str,
    totalspineseg_executable: str, totalspineseg_data_dir: Path | None,
    axial_model_data_dir: Path | None, segmentation_python: Path | None,
    reuse_sagittal_segmentation: Path | None,
    segmentation_runner: Callable[[Path, Path], tuple[Path, list[str]]] | None,
    segmentation_backend: SegmentationBackend | None,
    segmentation_backends: dict[str, SegmentationBackend] | None,
    segmentation_backend_factory: Callable[[dict[str, Any]], SegmentationBackend] | None,
) -> tuple[Path, dict[str, Any]]:
    """Process axial volumes independently and select one candidate per region."""
    workflow_path, sagittal_nii = workflow_path.absolute(), Path(sagittal_nii).absolute()
    axial_niis = [Path(path).absolute() for path in axial_niis]
    if not sagittal_nii.is_file(): raise FileNotFoundError(f"sagittal NIfTI does not exist: {sagittal_nii}")
    missing = [str(p) for p in axial_niis if not p.is_file()]
    if missing: raise FileNotFoundError("axial NIfTI(s) do not exist: " + ", ".join(missing))
    if len({p.resolve() for p in axial_niis}) != len(axial_niis): raise ValueError("duplicate axial NIfTI paths are not allowed")
    document = normalize_workflow(json.loads(workflow_path.read_text(encoding="utf-8")))
    errors = validate_workflow(document)
    if errors: raise ValueError("invalid workflow:\n" + "\n".join(errors))
    matching_steps = [s for s in document["steps"] if s.get("tool") == "3d_matching" and s.get("operation") == "match_anatomical_levels"]
    if len(matching_steps) != 1: raise ValueError("multi-axial execution requires exactly one anatomical-level matching step")
    match_step_id = matching_steps[0]["id"]
    safe_case_id = _safe_component(case_id or sagittal_nii.name.split(".")[0], "case_id")
    safe_run_id = _safe_component(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8], "run_id")
    collection_dir = Path(runs_root).absolute() / safe_case_id / safe_run_id
    collection_dir.mkdir(parents=True, exist_ok=False)
    candidate_root = collection_dir / "candidates"; candidate_root.mkdir()
    (collection_dir / "workers").mkdir(parents=True, exist_ok=True)
    pool = dict(segmentation_backends or {}); owned = set()
    if segmentation_backend is not None:
        pool.setdefault(getattr(segmentation_backend, "model_backend", segmentation_backend.backend_id), segmentation_backend)
    if segmentation_runner is None and segmentation_backend_factory is None:
        for step in document["steps"]:
            if step.get("tool") != "segmentation" or step.get("operation") != "segment": continue
            options = step["options"]; backend_id = options["backend"]
            if backend_id in pool: continue
            if backend_id == "totalspineseg_cli": backend = TotalSpineSegCliBackend(totalspineseg_executable, totalspineseg_data_dir)
            elif backend_id == "totalspineseg_python":
                model_data = totalspineseg_data_dir or (Path(os.environ["TOTALSPINESEG_DATA"]) if os.environ.get("TOTALSPINESEG_DATA") else None)
                if segmentation_python is None or model_data is None: raise ValueError("totalspineseg_python requires Python and model data")
                backend = PersistentPythonBackend(segmentation_python, model_data, model_backend=backend_id, log_path=collection_dir / "workers" / "totalspineseg_worker.log")
            elif backend_id == "axial_nnunet_python":
                if segmentation_python is None or axial_model_data_dir is None: raise ValueError("axial_nnunet_python requires Python and model data")
                backend = PersistentPythonBackend(segmentation_python, axial_model_data_dir, model_backend=backend_id, log_path=collection_dir / "workers" / "axial_segmentation_worker.log")
            elif backend_id == "sct_cli": backend = TotalSpineSegBackend(sct_executable)
            else: continue
            backend.start(); pool[backend_id] = backend; owned.add(backend_id)
    entries = []; shared_sag_mask = Path(reuse_sagittal_segmentation).absolute() if reuse_sagittal_segmentation else None
    ordered_planes = _ordered_axial_planes(sagittal_nii, axial_niis)
    plane_order = {
        (str(Path(plane["source_nifti"]).resolve()), int(plane["slice_index_0based"])): plane
        for plane in ordered_planes
    }
    last_slice_by_file = {}
    for plane in ordered_planes:
        key = str(Path(plane["source_nifti"]).resolve())
        last_slice_by_file[key] = max(
            last_slice_by_file.get(key, -1), int(plane["slice_index_0based"])
        )
    try:
        for i, axial_path in enumerate(axial_niis):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", axial_path.name).strip("._-") or f"axial_{i:02d}"
            series_id = f"axial_{i:02d}_{safe_name}"
            child_run, child_manifest = build_paired_nifti_workflow(
                workflow_path=workflow_path, sagittal_nii=sagittal_nii, axial_nii=axial_path,
                runs_root=candidate_root / series_id, case_id=safe_case_id, run_id="run",
                sct_executable=sct_executable, totalspineseg_executable=totalspineseg_executable,
                totalspineseg_data_dir=totalspineseg_data_dir, axial_model_data_dir=axial_model_data_dir,
                segmentation_python=segmentation_python, reuse_sagittal_segmentation=shared_sag_mask,
                segmentation_runner=segmentation_runner, segmentation_backends=pool or None,
                segmentation_backend_factory=segmentation_backend_factory)
            if shared_sag_mask is None:
                for rec in child_manifest.get("steps", []):
                    if rec.get("tool") == "segmentation" and rec.get("operation") == "segment" and rec.get("options", {}).get("view") == "sagittal_t2":
                        mask = next((a for a in rec.get("artifacts", []) if a.get("type") == "SegmentationMask"), None)
                        if mask: shared_sag_mask = child_run / mask["path"]; break
            match_step = next((s for s in child_manifest["steps"] if s.get("id") == match_step_id), None)
            match_artifact = next((a for a in (match_step or {}).get("artifacts", []) if a.get("type") == "SliceMatchSet"), None)
            if not match_artifact: raise RuntimeError(f"no persisted match result for {series_id}")
            match_json = json.loads((child_run / match_artifact["path"]).read_text(encoding="utf-8"))
            roi_json = json.loads((child_run / child_manifest["result"]["path"]).read_text(encoding="utf-8")) if child_manifest.get("result") else None
            axial_mask_path = None
            for step_record in child_manifest.get("steps", []):
                if step_record.get("tool") == "segmentation" and step_record.get("operation") == "segment" and step_record.get("options", {}).get("view") == "axial_t2":
                    artifact = next((a for a in step_record.get("artifacts", []) if a.get("type") == "SegmentationMask"), None)
                    if artifact:
                        axial_mask_path = child_run / artifact["path"]
                        break
            entries.append({"series_id": series_id, "path": str(axial_path), "run_dir": child_run,
                "manifest": child_manifest, "matches": match_json.get("regions", {}), "roi": roi_json,
                "match_artifact": match_artifact, "axial_mask_path": axial_mask_path})
        requested = document.get("requested_regions", {}).get("axial_t2")
        if requested is None:
            requested = list(dict.fromkeys(r for e in entries for r in e["matches"]))
        requested = list(requested)
        matching_method = matching_steps[0].get("options", {}).get("method", "ratio")

        # Score every physical axial plane in one ordered candidate sequence.
        # This makes ratio matching invariant to whether the same planes are
        # stored in one NIfTI or split across several NIfTIs.
        if matching_method == "ratio":
            match_step = matching_steps[0]
            source_ref = match_step["inputs"]["source_mask"]
            source_ref_match = re.fullmatch(r"\$steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)", source_ref)
            if source_ref_match is None:
                raise ValueError("multi-NIfTI ratio matching requires a named sagittal mask input")
            source_step_id = source_ref_match.group(1)
            source_record = next((record for record in entries[0]["manifest"]["steps"] if record["id"] == source_step_id), None)
            source_artifact = next((a for a in (source_record or {}).get("artifacts", []) if a.get("type") == "SegmentationMask"), None)
            if source_artifact is None:
                raise RuntimeError("could not resolve the sagittal segmentation mask for global ratio matching")
            global_sag_mask_path = entries[0]["run_dir"] / source_artifact["path"]
            global_sag_image = nib.load(str(global_sag_mask_path))
            global_sag_labels = np.rint(np.asarray(global_sag_image.dataobj)).astype(np.int16)
            global_sag_series = _load_series(sagittal_nii, "sagittal_t2", "sagittal_t2")
            reference_ref = match_step["inputs"]["reference_slice"]
            reference_match = re.fullmatch(r"\$steps\.([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)", reference_ref)
            if reference_match is None:
                raise ValueError("multi-NIfTI ratio matching requires a named sagittal reference slice")
            reference_step = next(step for step in document["steps"] if step["id"] == reference_match.group(1))
            selector_labels = [int(value) for value in reference_step["options"].get("label_ids", ())]
            if not selector_labels:
                raise ValueError("global ratio matching requires label_ids on the sagittal reference selector")
            reduce_axes = tuple(axis for axis in range(3) if axis != global_sag_series.slice_axis)
            area_by_sag_slice = np.count_nonzero(np.isin(global_sag_labels, selector_labels), axis=reduce_axes)
            if not np.any(area_by_sag_slice):
                raise ValueError("sagittal reference anatomy is absent for global ratio matching")
            reference_sagittal_index = int(np.argmax(area_by_sag_slice))
            first_axial = _load_series(Path(entries[0]["path"]), "axial_t2", "axial_t2")
            global_matches = match_anatomical_levels(
                global_sag_labels, global_sag_image.affine, first_axial.data.shape, first_axial.affine,
                sagittal_slice_axis=global_sag_series.slice_axis,
                axial_slice_axis=first_axial.slice_axis,
                reference_sagittal_index=reference_sagittal_index,
                neighborhood_radius=int(match_step["options"].get("neighborhood_radius", 1)),
                min_voxels_per_sagittal_slice=int(match_step["options"].get("min_voxels_per_sagittal_slice", 41)),
                requested_regions=requested,
                region_label_ids=match_step["options"].get("region_label_ids"),
                candidate_plane_geometries=[
                    (plane["plane_origin_ras"], plane["plane_normal_ras"]) for plane in ordered_planes
                ],
            )
            chosen_planes = {}
            for region, match in global_matches.items():
                if match.status == "selected" and match.slice_index is not None:
                    chosen_planes[region] = (ordered_planes[match.slice_index], match)

            # Re-run the cheap workflow stages with the globally selected
            # matches, reusing masks from the first pass (no second inference).
            cached_axial_masks = {
                str(Path(entry["path"]).resolve()): entry["axial_mask_path"]
                for entry in entries if entry.get("axial_mask_path") is not None
            }
            axial_segment_steps = [
                step for step in document["steps"]
                if step.get("tool") == "segmentation" and step.get("operation") == "segment"
                and step.get("options", {}).get("view") == "axial_t2"
            ]
            if cached_axial_masks and len(axial_segment_steps) == 1:
                def cached_segmentation_runner(input_nii: Path, output_dir: Path):
                    key = str(Path(input_nii).resolve())
                    if key not in cached_axial_masks:
                        raise RuntimeError(f"no cached segmentation mask for {input_nii}")
                    return cached_axial_masks[key], ["REUSE_CACHED_SEGMENTATION", str(cached_axial_masks[key])]
            else:
                # The legacy runner seam cannot distinguish multiple segmentation
                # requests on one input path; leave those requests to their backend.
                cached_segmentation_runner = segmentation_runner
            final_root = collection_dir / "final_candidates"
            final_root.mkdir()
            for entry in entries:
                per_file_override = {}
                for region in requested:
                    selected_pair = chosen_planes.get(region)
                    if selected_pair is not None and str(Path(selected_pair[0]["source_nifti"]).resolve()) == str(Path(entry["path"]).resolve()):
                        plane, global_match = selected_pair
                        per_file_override[region] = LevelSliceMatch(
                            level=region, slice_index=int(plane["slice_index_0based"]),
                            method=global_match.method, score=global_match.score,
                            source_sagittal_indices=global_match.source_sagittal_indices,
                            details={**global_match.details, "global_order_1based": plane["global_order_1based"],
                                     "superior_ras_mm_at_mid_sagittal": plane["superior_ras_mm"]},
                        )
                    else:
                        reason = "selected plane belongs to another axial NIfTI"
                        per_file_override[region] = LevelSliceMatch(
                            level=region, slice_index=None, method="ratio", score=None,
                            source_sagittal_indices=tuple(global_matches[region].source_sagittal_indices),
                            details={}, status="missing", reason=reason,
                        )
                final_run, final_manifest = build_paired_nifti_workflow(
                    workflow_path=workflow_path, sagittal_nii=sagittal_nii, axial_nii=Path(entry["path"]),
                    runs_root=final_root / entry["series_id"], case_id=safe_case_id, run_id="globally_matched",
                    sct_executable=sct_executable, totalspineseg_executable=totalspineseg_executable,
                    totalspineseg_data_dir=totalspineseg_data_dir, axial_model_data_dir=axial_model_data_dir,
                    segmentation_python=segmentation_python, reuse_sagittal_segmentation=shared_sag_mask,
                    segmentation_runner=cached_segmentation_runner,
                    segmentation_backend=segmentation_backend, segmentation_backends=pool or None,
                    segmentation_backend_factory=segmentation_backend_factory,
                    _match_override=per_file_override,
                )
                final_match_step = next(step for step in final_manifest["steps"] if step.get("id") == match_step_id)
                final_match_artifact = next(a for a in final_match_step.get("artifacts", []) if a.get("type") == "SliceMatchSet")
                final_matches = json.loads((final_run / final_match_artifact["path"]).read_text(encoding="utf-8"))
                final_roi = json.loads((final_run / final_manifest["result"]["path"]).read_text(encoding="utf-8")) if final_manifest.get("result") else None
                entry.update({"run_dir": final_run, "manifest": final_manifest, "matches": final_matches.get("regions", {}),
                              "roi": final_roi, "match_artifact": final_match_artifact})

        def region_candidates(region: str) -> list[dict[str, Any]]:
            found = []
            for entry in entries:
                match = entry["matches"].get(region, {})
                index = match.get("slice_index_0based")
                if match.get("status") != "selected" or index is None:
                    continue
                plane = plane_order.get((str(Path(entry["path"]).resolve()), int(index)), {})
                found.append({
                    "entry": entry,
                    "match": match,
                    "index": int(index),
                    "position": float(plane.get("superior_ras_mm", 0.0)),
                    "rank": int(plane.get("global_order_1based", 10**9)),
                    "slice_ref": {
                        "nifti_file": Path(entry["path"]).name,
                        "slice_index_0based": int(index),
                    },
                    "on_source_stack_boundary": int(index) in (
                        0, last_slice_by_file.get(str(Path(entry["path"]).resolve()), -1)
                    ),
                })
            return sorted(found, key=lambda item: item["rank"])

        def choose_candidate(region: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
            if not candidates:
                return None
            is_disc = "/" in region or "-" in region
            if matching_method == "ratio" and is_disc:
                empty_gaps = [item for item in candidates if item["match"].get("method") == "empty_gap_slice"]
                if empty_gaps:
                    return empty_gaps[0]
                scored = [item for item in candidates if item["match"].get("score") is not None]
                if scored:
                    best_score = min(float(item["match"]["score"]) for item in scored)
                    return next(item for item in scored if abs(float(item["match"]["score"]) - best_score) <= 1e-8)
                return candidates[0]
            scored = [item for item in candidates if item["match"].get("score") is not None]
            if scored:
                best_score = max(float(item["match"]["score"]) for item in scored)
                return next(item for item in scored if abs(float(item["match"]["score"]) - best_score) <= 1e-8)
            return candidates[0]

        selected = {}
        report = {}
        winner_by_region = {}
        ordered_regions = sorted(requested, key=lambda region: (("/" in region or "-" in region), requested.index(region)))
        for region in ordered_regions:
            candidates = region_candidates(region)
            is_disc = "/" in region or "-" in region
            if matching_method == "ratio" and is_disc and candidates:
                parts = region.replace("-", "/").split("/")
                if len(parts) == 2:
                    upper = winner_by_region.get(parts[0]) or choose_candidate(parts[0], region_candidates(parts[0]))
                    lower = winner_by_region.get(parts[1]) or choose_candidate(parts[1], region_candidates(parts[1]))
                    if upper is not None and lower is not None:
                        lo, hi = sorted((upper["position"], lower["position"]))
                        candidates = [item for item in candidates if lo - 1e-6 <= item["position"] <= hi + 1e-6]
            winner = choose_candidate(region, candidates)
            if winner is None:
                selected[region] = None
                report[region] = {"status": "missing", "candidates": []}
                continue

            entry, match = winner["entry"], winner["match"]
            slice_ref = winner["slice_ref"]
            candidate_summary = [{
                "series_id": item["entry"]["series_id"],
                "status": item["match"].get("status"),
                "slice_ref": item["slice_ref"],
                "on_source_stack_boundary": item["on_source_stack_boundary"],
                "global_order_1based": item["rank"],
                "superior_ras_mm_at_mid_sagittal": item["position"],
                "method": item["match"].get("method"),
                "score": item["match"].get("score"),
            } for item in candidates]
            roi = ((entry["roi"] or {}).get("views", {}).get("axial_t2", {})).get(region)
            if roi is not None:
                roi = dict(
                    roi,
                    source_series_id=entry["series_id"],
                    source_nifti=entry["path"],
                    slice_ref=slice_ref,
                    match_method=match.get("method"),
                    match_score=match.get("score"),
                )
            selected[region] = roi
            winner_by_region[region] = winner
            report[region] = {
                "status": "selected" if roi is not None else "roi_missing",
                "source_series_id": entry["series_id"],
                "source_nifti": entry["path"],
                "slice_ref": slice_ref,
                "on_source_stack_boundary": winner["on_source_stack_boundary"],
                "global_order_1based": winner["rank"],
                "superior_ras_mm_at_mid_sagittal": winner["position"],
                "method": match.get("method"),
                "score": match.get("score"),
                "candidates": candidate_summary,
            }
        first_roi = entries[0]["roi"]; result_artifact = None
        views = {}
        if first_roi is not None:
            requested_by_view = document.get("requested_regions", {})
            first_views = first_roi.get("views", {})
            sagittal_requested = list(requested_by_view.get("sagittal_t2", requested))
            axial_requested = list(requested_by_view.get("axial_t2", requested))
            if "sagittal_t2" in first_views:
                sagittal_selected = {}
                for region in sagittal_requested:
                    winner = winner_by_region.get(region)
                    roi_record = None
                    if winner is not None:
                        roi_record = ((winner["entry"]["roi"] or {}).get("views", {}).get("sagittal_t2", {})).get(region)
                    if roi_record is None:
                        sagittal_selected[region] = None
                        report.setdefault(region, {})["sagittal_roi_status"] = "missing"
                    else:
                        sagittal_selected[region] = roi_record
                        report.setdefault(region, {})["sagittal_roi_status"] = "selected"
                        report[region]["sagittal_source_nifti"] = winner["entry"]["path"]
                        report[region]["sagittal_source_axial_slice_ref"] = winner["slice_ref"]
                        if report[region].get("status") != "selected":
                            report[region]["status"] = "selected"
                views["sagittal_t2"] = sagittal_selected
            if "axial_t2" in first_views:
                views["axial_t2"] = {region: selected.get(region) for region in axial_requested}
            result_path = collection_dir / "outputs" / "roi_results.json"
            _write_json(result_path, {"schema_version": "1.0", "case_id": safe_case_id, "coordinate_system": "display-oriented image pixels", "bbox_convention": "xyxy; half-open pixel-edge coordinates", "views": views})
            result_artifact = _artifact(result_path, collection_dir, "WorkflowROIResults", views=list(views), roi_count=sum(len(v) for v in views.values()))
        manifest = {"manifest_version": "1.0", "workflow_id": document["id"], "run_id": safe_run_id, "case_id": safe_case_id,
            "status": "succeeded", "started_at": _utc_now(), "completed_at": _utc_now(),
            "execution_mode": "global_physical_plane_ratio" if matching_method == "ratio" else "multi_axial_independent_candidates",
            "inputs": {"sagittal_t2": str(sagittal_nii), "axial_t2": [{"series_id": e["series_id"], "path": e["path"]} for e in entries]},
            "candidates": [{"series_id": e["series_id"], "manifest": e["run_dir"].relative_to(collection_dir).as_posix()+"/manifest.json",
                "match_artifact": e["run_dir"].relative_to(collection_dir).as_posix()+"/"+e["match_artifact"]["path"]} for e in entries],
            "region_selection": report, "requested_regions": document.get("requested_regions", {}),
            "result": ({"type": result_artifact["type"], "path": result_artifact["path"], "size_bytes": result_artifact["size_bytes"], "sha256": result_artifact["sha256"], "metadata": result_artifact["metadata"]} if result_artifact else None),
            "workflow_outputs": [], "warnings": []}
        if result_artifact: manifest["workflow_outputs"].append({"type": result_artifact["type"], "path": result_artifact["path"], "metadata": result_artifact["metadata"]})
        _write_manifest(collection_dir / "manifest.json", manifest)
        return collection_dir, manifest
    finally:
        for key in owned: pool[key].close()

def build_paired_nifti_workflow(
    workflow_path: Path,
    sagittal_nii: Path,
    axial_nii: Path | list[Path] | tuple[Path, ...],
    runs_root: Path,
    *,
    case_id: str | None = None,
    run_id: str | None = None,
    sct_executable: str = "sct_deepseg",
    totalspineseg_executable: str = "totalspineseg",
    totalspineseg_data_dir: Path | None = None,
    axial_model_data_dir: Path | None = None,
    segmentation_python: Path | None = None,
    reuse_sagittal_segmentation: Path | None = None,
    segmentation_runner: Callable[[Path, Path], tuple[Path, list[str]]] | None = None,
    segmentation_backend: SegmentationBackend | None = None,
    segmentation_backends: dict[str, SegmentationBackend] | None = None,
    segmentation_backend_factory: Callable[[dict[str, Any]], SegmentationBackend] | None = None,
    _match_override: dict[str, LevelSliceMatch] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Interpret a validated JSON workflow for separate Sag and Ax NIfTIs.

    Each step owns ``steps/<step-id>/``. Public outputs are under ``outputs/``;
    source NIfTIs are referenced by absolute path and never copied or modified.
    A new run directory is created exclusively to avoid overwriting prior runs.
    """

    if isinstance(axial_nii, (list, tuple)):
        axial_paths = [Path(path).absolute() for path in axial_nii]
        if not axial_paths:
            raise ValueError("at least one axial NIfTI is required")
        if len(axial_paths) > 1:
            return _build_multi_axial_nifti_workflow(
                workflow_path=workflow_path, sagittal_nii=sagittal_nii, axial_niis=axial_paths,
                runs_root=runs_root, case_id=case_id, run_id=run_id,
                sct_executable=sct_executable, totalspineseg_executable=totalspineseg_executable,
                totalspineseg_data_dir=totalspineseg_data_dir, axial_model_data_dir=axial_model_data_dir,
                segmentation_python=segmentation_python, reuse_sagittal_segmentation=reuse_sagittal_segmentation,
                segmentation_runner=segmentation_runner, segmentation_backend=segmentation_backend,
                segmentation_backends=segmentation_backends, segmentation_backend_factory=segmentation_backend_factory,
            )
        axial_nii = axial_paths[0]

    workflow_path = workflow_path.absolute()
    sagittal_nii, axial_nii = sagittal_nii.absolute(), axial_nii.absolute()
    if reuse_sagittal_segmentation is not None:
        reuse_sagittal_segmentation = reuse_sagittal_segmentation.absolute()
        if not reuse_sagittal_segmentation.is_file():
            raise FileNotFoundError(f"reused segmentation does not exist: {reuse_sagittal_segmentation}")
    runs_root = runs_root.absolute()
    document = normalize_workflow(json.loads(workflow_path.read_text(encoding="utf-8")))
    errors = validate_workflow(document)
    if errors:
        raise ValueError("invalid workflow:\n" + "\n".join(errors))
    case_id = _safe_component(case_id or sagittal_nii.name.split(".")[0], "case_id")
    run_id = _safe_component(run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8], "run_id")
    sagittal = _load_series(sagittal_nii, "sagittal_t2", "sagittal_t2")
    axial = _load_series(axial_nii, "axial_t2", "axial_t2")
    run_dir = runs_root / case_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs: dict[str, Any] = {
        "$input.exam.series.sagittal_t2": sagittal,
        "$input.exam.series.axial_t2": axial,
        "$input.query.task": document["query"]["task"],
    }
    step_outputs: dict[str, dict[str, Any]] = {}
    backend_pool: dict[str, SegmentationBackend] = dict(segmentation_backends or {})
    owned_backends: set[str] = set()
    if segmentation_backend is not None:
        backend_key = getattr(segmentation_backend, "model_backend", segmentation_backend.backend_id)
        if backend_key in backend_pool and backend_pool[backend_key] is not segmentation_backend:
            raise ValueError(f"multiple supplied backends for {backend_key!r}")
        backend_pool[backend_key] = segmentation_backend
    segmentation_cache: dict[tuple[Any, ...], Any] = {}
    save_qc = bool(document.get("save_qc", False))
    artifact_refs = document.get("outputs", {}).get("artifacts", [])
    retained_debug_steps = {
        match.group(1)
        for ref in artifact_refs
        if isinstance(ref, str) and (match := re.fullmatch(r"\$steps\.([A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+", ref))
    }

    def persist_debug(step_id: str) -> bool:
        return save_qc or step_id in retained_debug_steps

    def debug_output_path(kind: str, step_id: str, filename: str, step_dir: Path) -> Path:
        if save_qc:
            return run_dir / "outputs" / "qc" / kind / step_id / filename
        return step_dir / filename

    manifest: dict[str, Any] = {
        "manifest_version": "1.0",
        "workflow_id": document["id"],
        "run_id": run_id,
        "case_id": case_id,
        "status": "running",
        "started_at": _utc_now(),
        "completed_at": None,
        "coordinate_convention": {
            "input_nifti": "RAS+ as encoded by each NIfTI affine",
            "dicom": "LPS; convert RAS to LPS with diag(-1,-1,1) before comparing DICOM coordinates",
        },
        "inputs": {},
        "steps": [],
        "workflow_outputs": [],
        "supporting_artifacts": [],
        "qc": {"enabled": save_qc, "directory": "outputs/qc" if save_qc else None},
        "requested_regions": document.get("requested_regions", {}),
        "warnings": [],
        "software": {"python": sys.version.split()[0], "nibabel": nib.__version__, "numpy": np.__version__},
    }
    for series in (sagittal, axial):
        manifest["inputs"][series.view] = {
            "path": str(series.path),
            "sha256": _sha256(series.path),
            "size_bytes": series.path.stat().st_size,
            "shape": list(series.data.shape),
            "voxel_to_ras": series.affine.tolist(),
            "slice_axis": series.slice_axis,
            "deviation_from_standard_view_deg": series.view_deviation_deg,
        }
    manifest_path = run_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    try:
        for step in document["steps"]:
            step_id = step["id"]
            step_dir = run_dir / "steps" / step_id
            step_dir.mkdir(parents=True, exist_ok=False)
            record: dict[str, Any] = {
                "id": step_id,
                "tool": step["tool"],
                "operation": step["operation"],
                "status": "running",
                "started_at": _utc_now(),
                "completed_at": None,
                "inputs": step["inputs"],
                "options": step["options"],
                "outputs": {},
            }
            manifest["steps"].append(record)
            _write_manifest(manifest_path, manifest)
            key = (step["tool"], step["operation"])
            resolved = {name: _resolve(ref, inputs, step_outputs) for name, ref in step["inputs"].items()}
            options = step["options"]

            if key == ("segmentation", "segment"):
                series = resolved["series"]
                if not isinstance(series, SeriesVolume):
                    raise TypeError(f"step {step_id}: segmentation requires an MRIImageSeries")
                if reuse_sagittal_segmentation is not None and series.view == "sagittal_t2":
                    reused_dir = step_dir / "reused_segmentation"
                    reused_dir.mkdir(parents=True, exist_ok=False)
                    label_path = reused_dir / "reused_labels.nii.gz"
                    shutil.copy2(reuse_sagittal_segmentation, label_path)
                    command = ["REUSE_SEGMENTATION", str(reuse_sagittal_segmentation)]
                    record["segmentation_source"] = str(reuse_sagittal_segmentation)
                elif segmentation_runner is not None:
                    # Test/legacy injection seam; production backends use the lifecycle contract.
                    cache_key = (
                        str(series.path), options["backend"], options["model_id"],
                        tuple(options["structures"]), json.dumps(options, sort_keys=True),
                    )
                    if cache_key not in segmentation_cache:
                        segmentation_cache[cache_key] = segmentation_runner(series.path, step_dir / "segmentation")
                    label_path, command = segmentation_cache[cache_key]
                elif options["backend"] == "sct_cli":
                    if series.view != "sagittal_t2":
                        raise ValueError("legacy sct_cli adapter supports only sagittal_t2")
                    cache_key = (
                        str(series.path), options["backend"], options["model_id"],
                        tuple(options["structures"]), json.dumps(options, sort_keys=True),
                    )
                    if cache_key not in segmentation_cache:
                        segmentation_cache[cache_key] = TotalSpineSegBackend(sct_executable).run(
                            series.path, step_dir / "segmentation"
                        )
                    label_path, command = segmentation_cache[cache_key]
                else:
                    backend_key = options["backend"]
                    backend = backend_pool.get(backend_key)
                    if backend is None:
                        if segmentation_backend_factory is not None:
                            backend = segmentation_backend_factory(options)
                        elif options["backend"] == "totalspineseg_cli":
                            backend = TotalSpineSegCliBackend(
                                totalspineseg_executable,
                                totalspineseg_data_dir,
                            )
                        elif options["backend"] == "totalspineseg_python":
                            if segmentation_python is None:
                                raise ValueError("totalspineseg_python requires --segmentation-python")
                            model_data = totalspineseg_data_dir or (
                                Path(os.environ["TOTALSPINESEG_DATA"])
                                if os.environ.get("TOTALSPINESEG_DATA") else None
                            )
                            if model_data is None:
                                raise ValueError("totalspineseg_python requires --totalspineseg-data or TOTALSPINESEG_DATA")
                            backend = PersistentPythonBackend(
                                segmentation_python,
                                model_data,
                                model_backend="totalspineseg_python",
                                log_path=run_dir / "segmentation_worker.log",
                            )
                        elif options["backend"] == "axial_nnunet_python":
                            if segmentation_python is None:
                                raise ValueError("axial_nnunet_python requires --segmentation-python")
                            if axial_model_data_dir is None:
                                raise ValueError("axial_nnunet_python requires --axial-model-data")
                            backend = PersistentPythonBackend(
                                segmentation_python,
                                axial_model_data_dir,
                                model_backend="axial_nnunet_python",
                                log_path=run_dir / "axial_segmentation_worker.log",
                            )
                        else:
                            raise ValueError(f"no runtime backend configured for {options['backend']!r}")
                        backend.start()
                        backend_pool[backend_key] = backend
                        owned_backends.add(backend_key)
                    elif not backend.is_started:
                        raise RuntimeError(
                            "a caller-supplied segmentation_backend must be started by its owner "
                            "(for example, use 'with backend:' around sequential case workflows)"
                        )
                    cache_key = (
                        str(series.path), options["backend"], options["model_id"],
                        tuple(options["structures"]), json.dumps(options, sort_keys=True),
                    )
                    if cache_key not in segmentation_cache:
                        request = SegmentationRequest(
                            input_image=series.path,
                            output_dir=step_dir / "segmentation",
                            view=series.view,
                            structures=tuple(options["structures"]),
                            model_id=options["model_id"],
                            case_id=case_id,
                        )
                        segmentation_cache[cache_key] = backend.segment(request)
                    backend_result = segmentation_cache[cache_key]
                    label_path = backend_result.label_map
                    command = list(backend_result.command)
                    record["segmentation_backend"] = {
                        "id": backend_result.backend_id,
                        "version": backend_result.backend_version,
                        "view": backend_result.view,
                        "model_id": backend_result.model_id,
                        "geometry": {
                            "shape_ijk": list(backend_result.geometry.shape_ijk),
                            "voxel_to_world": [list(row) for row in backend_result.geometry.voxel_to_world],
                            "coordinate_system": backend_result.geometry.coordinate_system,
                        },
                        "label_map_semantics": {
                            str(key): value for key, value in backend_result.label_map_semantics.items()
                        },
                        "metadata": backend_result.metadata,
                    }
                    record["logs"] = [str(path) for path in backend_result.log_paths]
                try:
                    label_path.absolute().relative_to(run_dir)
                except ValueError:
                    stored_labels = step_dir / "segmentation_labels.nii.gz"
                    shutil.copy2(label_path, stored_labels)
                    label_path = stored_labels
                artifact = _load_segmentation(label_path, series)
                outputs = {"mask": artifact}
                record["command"] = command
                record["artifacts"] = [_artifact(label_path, run_dir, "SegmentationMask", labels="backend-specific labels")]
                if artifact.canal_probability_path is not None:
                    record["artifacts"].append(
                        _artifact(artifact.canal_probability_path, run_dir, "SegmentationProbability", structure="spinal_canal")
                    )
            elif key == ("segmentation", "postprocess"):
                source_mask = resolved["mask"]
                if not isinstance(source_mask, SegmentationArtifact):
                    raise TypeError("segmentation.postprocess requires a SegmentationMask")
                series = source_mask.series
                target_ids = [int(value) for value in options["target_label_ids"]]
                processed_labels = source_mask.labels.copy()
                slice_metadata: list[dict[str, Any]] = []
                method = options["method"]
                if method == "anterior_largest_component":
                    if series.view != "sagittal_t2":
                        raise ValueError("anterior_largest_component requires sagittal_t2")
                    reference_ids = [int(value) for value in options["reference_label_ids"]]
                    for index in range(processed_labels.shape[series.slice_axis]):
                        display_plane = _orient_plane(series, index, source_mask.labels)
                        if not np.isin(display_plane, reference_ids).any():
                            slice_metadata.append({
                                "slice_index_0based": index,
                                "status": "skipped_reference_labels_absent",
                                "reference_label_ids": reference_ids,
                            })
                            continue
                        filtered_plane, filter_info = keep_anterior_components(
                            display_plane,
                            target_label_ids=target_ids,
                            reference_label_ids=reference_ids,
                            allow_missing_labels=True,
                        )
                        native_plane = _restore_display_plane(series, filtered_plane)
                        plane_selector = [slice(None)] * 3
                        plane_selector[series.slice_axis] = index
                        processed_labels[tuple(plane_selector)] = native_plane
                        slice_metadata.append({
                            "slice_index_0based": index,
                            "status": "processed",
                            **filter_info,
                        })
                    operation_description = (
                        "per sagittal slice: clip each target label to the display-left/anterior side "
                        "of the canal+cord left boundary, then retain its largest 8-connected component"
                    )
                elif method == "center_nearest_largest_component":
                    if series.view != "axial_t2":
                        raise ValueError("center_nearest_largest_component requires axial_t2")
                    minimum_area_fraction = float(options.get("minimum_area_fraction", 0.5))
                    for index in range(processed_labels.shape[series.slice_axis]):
                        display_plane = _orient_plane(series, index, source_mask.labels)
                        filtered_plane, filter_info = keep_center_nearest_components(
                            display_plane,
                            target_label_ids=target_ids,
                            minimum_area_fraction=minimum_area_fraction,
                        )
                        native_plane = _restore_display_plane(series, filtered_plane)
                        plane_selector = [slice(None)] * 3
                        plane_selector[series.slice_axis] = index
                        processed_labels[tuple(plane_selector)] = native_plane
                        slice_metadata.append({
                            "slice_index_0based": index,
                            "status": "processed",
                            **filter_info,
                        })
                    operation_description = (
                        "per axial slice and target label: retain components at least "
                        f"{minimum_area_fraction:g} of the largest component, then keep the component "
                        "whose centroid is nearest the in-plane image center"
                    )
                else:
                    raise ValueError(f"unsupported segmentation postprocess method: {method}")
                processed_path = step_dir / "processed_segmentation.nii.gz"
                source_image = nib.load(str(source_mask.label_path))
                header = source_image.header.copy()
                header.set_data_dtype(np.int16)
                nib.save(nib.Nifti1Image(processed_labels.astype(np.int16), source_mask.affine, header=header), str(processed_path))
                processed_artifact = _load_segmentation(processed_path, series)
                report_path = step_dir / "postprocessing.json"
                report = {
                    "method": method,
                    "view": series.view,
                    "source_mask": str(source_mask.label_path),
                    "target_label_ids": target_ids,
                    "reference_label_ids": options.get("reference_label_ids"),
                    "minimum_area_fraction": options.get("minimum_area_fraction", 0.5)
                    if method == "center_nearest_largest_component" else None,
                    "operation": operation_description,
                    "canonical_source_unchanged": True,
                    "slices": slice_metadata,
                }
                _write_json(report_path, report)
                outputs = {"mask": processed_artifact}
                record["postprocessing"] = report
                record["artifacts"] = [
                    _artifact(processed_path, run_dir, "SegmentationMask", postprocessed=True, view=series.view),
                    _artifact(report_path, run_dir, "SegmentationPostprocessingReport", view=series.view),
                ]
            elif key == ("slice_selection", "select_sagittal_reference"):
                series, segmentation = resolved["series"], resolved["mask"]
                label_ids = tuple(int(value) for value in options.get("label_ids", ()))
                if options["selector"] == "max_label_union_area":
                    mask = np.isin(segmentation.labels, label_ids)
                else:
                    mask = segmentation.labels == 2
                reduce_axes = tuple(axis for axis in range(3) if axis != series.slice_axis)
                area_by_index = np.count_nonzero(mask, axis=reduce_axes)
                if not np.any(mask):
                    index = None
                    selected = SelectedSlice(
                        series, None, options["selector"], 0, tuple(int(value) for value in area_by_index),
                        label_ids, "requested reference anatomy is absent from the segmentation",
                    )
                else:
                    index = int(np.argmax(area_by_index))
                    selected = SelectedSlice(
                        series, index, options["selector"], int(area_by_index[index]),
                        tuple(int(value) for value in area_by_index), label_ids,
                    )
                selection_path = None
                if persist_debug(step_id):
                    selection_path = debug_output_path("slice_selection", step_id, "selection.json", step_dir)
                    _write_json(selection_path, {
                        "status": "selected" if index is not None else "missing",
                        "reason": selected.reason,
                        "slice_index_0based": index,
                        "slice_axis": series.slice_axis,
                        "selector": selected.selector,
                        "label_ids": list(label_ids),
                        "area_by_index_voxels": list(selected.area_by_index),
                        "selected_area_voxels": selected.area,
                        "plane": _plane_metadata(series, index),
                    })
                outputs = {"slice": selected}
                record["artifacts"] = (
                    [_artifact(selection_path, run_dir, "SliceSelection", view=series.view)]
                    if selection_path is not None else []
                )
            elif key == ("slice_selection", "select_sagittal_regions"):
                series, segmentation = resolved["series"], resolved["mask"]
                if series.view != "sagittal_t2":
                    raise ValueError("select_sagittal_regions requires a sagittal series")
                reduce_axes = tuple(axis for axis in range(3) if axis != series.slice_axis)
                region_label_ids = options["region_label_ids"]
                selections: dict[str, SelectedSlice] = {}
                serialized: dict[str, Any] = {}
                for region, raw_ids in region_label_ids.items():
                    label_ids = tuple(int(value) for value in raw_ids)
                    region_mask = np.isin(segmentation.labels, label_ids)
                    area_by_index = np.count_nonzero(region_mask, axis=reduce_axes)
                    if not np.any(region_mask):
                        index = None
                        selected = SelectedSlice(
                            series, None, options["selector"], 0, tuple(int(value) for value in area_by_index),
                            label_ids, f"requested anatomy is absent for {region}",
                        )
                    else:
                        index = int(np.argmax(area_by_index))
                        selected = SelectedSlice(
                            series, index, options["selector"], int(area_by_index[index]),
                            tuple(int(value) for value in area_by_index), label_ids,
                        )
                    selections[region] = selected
                    serialized[region] = {
                        "status": "selected" if index is not None else "missing",
                        "reason": selected.reason,
                        "slice_index_0based": index,
                        "slice_axis": series.slice_axis,
                        "selector": selected.selector,
                        "label_ids": list(label_ids),
                        "area_by_index_voxels": list(selected.area_by_index),
                        "selected_area_voxels": selected.area,
                        "plane": _plane_metadata(series, index),
                    }
                selection_path = None
                if persist_debug(step_id):
                    selection_path = debug_output_path("slice_selection", step_id, "selections.json", step_dir)
                    _write_json(selection_path, {
                        "selection_policy": "independent_max_label_union_area_per_region",
                        "regions": serialized,
                    })
                selected_set = SelectedSliceSet(series, selections, selection_path)
                outputs = {"slices": selected_set}
                record["artifacts"] = (
                    [_artifact(selection_path, run_dir, "SliceSelectionSet", view=series.view, count=len(selections))]
                    if selection_path is not None else []
                )
            elif key == ("3d_matching", "tra2sag_regions"):
                source, target = resolved["source_series"], resolved["target_series"]
                segmentation, source_slices = resolved["source_mask"], resolved["source_slices"]
                matched = _tra2sag_matches(source, target, segmentation, source_slices, options)
                mapping_path = debug_output_path("slice_selection", step_id, "matches.json", step_dir)
                _write_json(mapping_path, {
                    "selection_policy": "tra2sag_bbox_point_projection",
                    "direction": "axial_to_sagittal",
                    "side": options["side"],
                    "requested_regions": options.get("requested_regions"),
                    "regions": {
                        level: {
                            "status": match.status, "reason": match.reason,
                            "slice_index_0based": match.slice_index, "method": match.method,
                            "score": match.score, "details": match.details,
                            "plane": _plane_metadata(target, match.slice_index) if match.slice_index is not None else None,
                        } for level, match in matched.items()
                    },
                })
                outputs = {"matches": LevelMatches(target, matched, mapping_path)}
                record["artifacts"] = [_artifact(mapping_path, run_dir, "SliceMatchSet", view=target.view)]
                for level, match in matched.items():
                    if match.status == "missing":
                        manifest["warnings"].append({"step": step_id, "region": level, "status": "missing", "reason": match.reason})
            elif key == ("3d_matching", "match_anatomical_levels"):
                source, target = resolved["source_series"], resolved["target_series"]
                segmentation, selected = resolved["source_mask"], resolved["reference_slice"]
                method = options.get("method", "ratio")
                if _match_override is not None:
                    matched = _match_override
                elif selected.index is None and method != "soft_distance":
                    requested = options.get("requested_regions") or [
                        "C3", "C3/C4", "C4", "C4/C5", "C5", "C5/C6", "C6", "C6/C7", "C7"
                    ]
                    matched = {
                        region: LevelSliceMatch(
                            level=region, slice_index=None, method=method, score=None,
                            source_sagittal_indices=(), details={}, status="missing",
                            reason=selected.reason or "reference sagittal slice is unavailable",
                        )
                        for region in requested
                    }
                elif method == "soft_distance":
                    matched = match_anatomical_levels_soft_distance(
                        segmentation.labels,
                        segmentation.affine,
                        target.data.shape,
                        target.affine,
                        sagittal_slice_axis=source.slice_axis,
                        axial_slice_axis=target.slice_axis,
                        sigma_mm=float(options.get("sigma_mm", 3.0)),
                        requested_regions=options.get("requested_regions"),
                        region_label_ids=options.get("region_label_ids"),
                    )
                else:
                    matched = match_anatomical_levels(
                        segmentation.labels,
                        segmentation.affine,
                        target.data.shape,
                        target.affine,
                        sagittal_slice_axis=source.slice_axis,
                        axial_slice_axis=target.slice_axis,
                        reference_sagittal_index=selected.index,
                        neighborhood_radius=int(options.get("neighborhood_radius", 1)),
                        min_voxels_per_sagittal_slice=int(options.get("min_voxels_per_sagittal_slice", 41)),
                        requested_regions=options.get("requested_regions"),
                        region_label_ids=options.get("region_label_ids"),
                    )
                mapping_path = (
                    debug_output_path("slice_selection", step_id, "matches.json", step_dir)
                    if persist_debug(step_id) else step_dir / "matches.json"
                )
                _write_json(mapping_path, {
                        "selection_policy": "sagittal_to_axial_level_match",
                        "method": options.get("method", "ratio"),
                        "requested_regions": options.get("requested_regions"),
                        "source_sagittal_indices": list(next(iter(matched.values())).source_sagittal_indices) if matched else [],
                        "regions": {
                            level: {
                                "status": match.status,
                                "reason": match.reason,
                                "slice_index_0based": match.slice_index,
                                "method": match.method,
                                "score": match.score,
                                "details": match.details,
                                "plane": _plane_metadata(target, match.slice_index) if match.slice_index is not None else None,
                            }
                            for level, match in matched.items()
                        },
                })
                level_matches = LevelMatches(target, matched, mapping_path)
                outputs = {"matches": level_matches}
                for level, match in matched.items():
                    if match.status == "missing":
                        manifest["warnings"].append({
                            "step": step_id, "region": level, "status": "missing", "reason": match.reason
                        })
                record["artifacts"] = [_artifact(mapping_path, run_dir, "SliceMatchSet", view=target.view)]
            elif key == ("slice_selection", "export_reference_slice"):
                series, selected = resolved["series"], resolved["slice"]
                filename = _safe_filename(options["filename"], "filename")
                output_path = run_dir / "outputs" / "sagittal" / filename
                evidence_path = step_dir / "evidence.json"
                if selected.index is None:
                    evidence = {
                        "view": "sagittal_t2", "status": "missing", "reason": selected.reason,
                        "slice_index_0based": None, "image": None, "plane": None,
                    }
                    _write_json(evidence_path, evidence)
                    outputs = {"evidence": evidence}
                    record["artifacts"] = [_artifact(evidence_path, run_dir, "EvidenceItem", view="sagittal_t2")]
                    manifest["warnings"].append({"step": step_id, "status": "missing", "reason": selected.reason})
                else:
                    _save_plane(series, selected.index, output_path)
                    evidence = {
                        "view": "sagittal_t2", "status": "selected",
                        "slice_index_0based": selected.index,
                        "image": output_path.relative_to(run_dir).as_posix(),
                        "plane": _plane_metadata(series, selected.index),
                        "selection_area_voxels": selected.area,
                        "selection_label_ids": list(selected.label_ids),
                    }
                    _write_json(evidence_path, evidence)
                    outputs = {"evidence": evidence}
                    record["artifacts"] = [
                        _artifact(output_path, run_dir, "ImageSlice", view="sagittal_t2", slice_index=selected.index),
                        _artifact(evidence_path, run_dir, "EvidenceItem", view="sagittal_t2"),
                    ]
            elif key == ("slice_selection", "export_matched_slices"):
                series, match_set = resolved["series"], resolved["matches"]
                filename_prefix = _safe_filename(options["filename_prefix"], "filename_prefix")
                evidence_items = []
                artifacts = []
                for level, match in match_set.matches.items():
                    if match.slice_index is None:
                        evidence_items.append({
                            "level": level, "view": "axial_t2", "status": "missing",
                            "reason": match.reason, "slice_index_0based": None,
                            "image": None, "method": match.method, "score": match.score, "plane": None,
                        })
                        manifest["warnings"].append({
                            "step": step_id, "region": level, "status": "missing", "reason": match.reason
                        })
                        continue
                    filename = f"{filename_prefix}_{level.replace('/', '_').replace('-', '_')}_idx{match.slice_index:02d}.png"
                    output_path = run_dir / "outputs" / "axial" / filename
                    _save_plane(series, match.slice_index, output_path, f"{level} | index {match.slice_index}")
                    item = {
                        "level": level, "view": "axial_t2", "status": "selected",
                        "slice_index_0based": match.slice_index,
                        "image": output_path.relative_to(run_dir).as_posix(),
                        "method": match.method, "score": match.score,
                        "plane": _plane_metadata(series, match.slice_index),
                    }
                    evidence_items.append(item)
                    artifacts.append(_artifact(output_path, run_dir, "ImageSlice", view="axial_t2", level=level, slice_index=match.slice_index))
                bundle_path = step_dir / "evidence.json"
                _write_json(bundle_path, {"evidence": evidence_items})
                artifacts.append(_artifact(bundle_path, run_dir, "EvidenceBundle", view="axial_t2", count=len(evidence_items)))
                outputs = {"evidence": evidence_items}
                record["artifacts"] = artifacts
            elif key[0] == "roi_extraction" and key[1] in {"mask_bbox", "mask_centered_bbox", "composite_bbox", "point_bbox"}:
                series = resolved["series"]
                selection = resolved["selection"]
                segmentation = resolved.get("mask")
                if not isinstance(series, SeriesVolume):
                    raise TypeError("ROI extraction requires an MRIImageSeries")
                if segmentation is not None and not isinstance(segmentation, SegmentationArtifact):
                    raise TypeError("ROI extraction mask must be a SegmentationMask")
                preset: dict[str, Any] | None = None
                if key[1] == "composite_bbox":
                    preset_root = workflow_path.parent.parent / "roi_presets"
                    preset_path = (preset_root / _safe_filename(options["preset_file"], "preset_file")).resolve()
                    try:
                        preset_path.relative_to(preset_root.resolve())
                    except ValueError as exc:
                        raise ValueError("ROI preset_file must stay inside configs/roi_presets") from exc
                    preset = json.loads(preset_path.read_text(encoding="utf-8"))
                    if not isinstance(preset.get("rois"), dict) or not preset["rois"]:
                        raise ValueError(f"ROI preset has no non-empty 'rois' mapping: {preset_path}")

                selections = _selection_indices(selection)
                if key[1] == "composite_bbox" and _is_region_slice_selection(selection):
                    selected_indices = {_roi_level_name(level): index for level, index in selections}
                else:
                    selected_indices = {}
                if key[1] in {"composite_bbox", "point_bbox"} and len(selections) != 1:
                    per_region_point_matches = key[1] == "point_bbox" and isinstance(selection, LevelMatches) and options.get("points_from_match")
                    if not (key[1] == "composite_bbox" and _is_region_slice_selection(selection)) and not per_region_point_matches:
                        raise ValueError(f"{key[1]} currently requires one selected image slice")
                roi_records: list[dict[str, Any]] = []
                if key[1] in {"mask_bbox", "mask_centered_bbox"}:
                    assert segmentation is not None
                    label_ids = [int(value) for value in options["label_ids"]]
                    requested = options.get("requested_regions")
                    for level, index in selections:
                        if requested is not None and level is not None and level not in requested:
                            continue
                        rule_name = key[1]
                        if index is None:
                            roi_records.append(_missing_roi_record(
                                level or "selected_slice", series.view, "no matching slice was selected",
                                rule=rule_name,
                            ))
                            continue
                        label_plane = _orient_plane(series, index, segmentation.labels)
                        plane_mask = np.isin(label_plane, label_ids)
                        try:
                            if key[1] == "mask_bbox":
                                bbox = mask_bbox(
                                    plane_mask,
                                    image_shape=plane_mask.shape,
                                    padding_px=options.get("padding_px"),
                                    reference_size=options.get("reference_size"),
                                )
                            else:
                                yy, xx = np.nonzero(plane_mask)
                                if not len(xx):
                                    raise ValueError(f"mask labels are absent: {label_ids}")
                                height, width = plane_mask.shape
                                reference_size = options.get("reference_size")
                                if reference_size is None:
                                    scale_x = scale_y = 1.0
                                else:
                                    scale_x = width / reference_size[0]
                                    scale_y = height / reference_size[1]
                                offset_x, offset_y = map(float, options["center_offset_px"])
                                half_width, half_height = options["half_size_px"]
                                center = (
                                    (float(xx.min()) + float(xx.max()) + 1.0) / 2.0 + offset_x * scale_x,
                                    (float(yy.min()) + float(yy.max()) + 1.0) / 2.0 + offset_y * scale_y,
                                )
                                bbox = point_bbox(
                                    center,
                                    image_shape=plane_mask.shape,
                                    margin_px={
                                        "left": half_width, "right": half_width,
                                        "top": half_height, "bottom": half_height,
                                    },
                                    reference_size=reference_size,
                                )
                        except ValueError as exc:
                            roi_records.append(_missing_roi_record(
                                level or "selected_slice", series.view, str(exc),
                                rule=rule_name, index=index,
                            ))
                            continue
                        roi_records.append(_roi_record(
                            level=level or "selected_slice", view=series.view, index=index, bbox=bbox,
                            image_shape=plane_mask.shape, rule=rule_name, label_ids=label_ids,
                        ))
                elif key[1] == "composite_bbox":
                    assert segmentation is not None and preset is not None
                    requested = options.get("requested_regions")
                    if requested is None:
                        output_levels = list(preset["rois"])
                    else:
                        output_levels = [_roi_level_name(region) for region in requested]
                        unknown = sorted(set(output_levels) - set(preset["rois"]))
                        if unknown:
                            raise ValueError(f"requested regions are not supported by ROI preset: {unknown}")
                    if _is_region_slice_selection(selection):
                        selected_levels = set(selected_indices)
                        expected_levels = set(output_levels)
                        if expected_levels != selected_levels:
                            missing = sorted(expected_levels - selected_levels)
                            extra = sorted(selected_levels - expected_levels)
                            raise ValueError(
                                "per-region slice selections must exactly match the requested ROI levels; "
                                f"missing={missing}, extra={extra}"
                            )
                    shared_index = selections[0][1] if len(selections) == 1 else None
                    for level in output_levels:
                        roi_spec = preset["rois"][level]
                        index = selected_indices.get(level) if _is_region_slice_selection(selection) else shared_index
                        if index is None:
                            roi_records.append(_missing_roi_record(
                                _public_region_name(level), series.view, "no selected image slice is available for this region",
                                rule=f"composite_bbox:{preset.get('id', options['preset_file'])}",
                            ))
                            continue
                        label_plane = _orient_plane(series, index, segmentation.labels)
                        image_shape = label_plane.shape
                        try:
                            bbox, boundary_resolution = composite_bbox_with_provenance(
                                label_plane,
                                image_shape=image_shape,
                                bounds=roi_spec["bounds"],
                                padding_px=roi_spec.get("padding_px"),
                                reference_size=options.get("reference_size"),
                            )
                        except ValueError as exc:
                            roi_records.append(_missing_roi_record(
                                _public_region_name(level), series.view, str(exc),
                                rule=f"composite_bbox:{preset.get('id', options['preset_file'])}",
                                index=index,
                            ))
                            continue
                        labels_used = sorted({
                            int(label_id)
                            for rule in roi_spec["bounds"].values()
                            for candidate in rule.get("candidates", [rule])
                            for label_id in candidate["label_ids"]
                        })
                        roi_records.append(_roi_record(
                            level=_public_region_name(level), view=series.view, index=index, bbox=bbox, image_shape=image_shape,
                            rule=f"composite_bbox:{preset.get('id', options['preset_file'])}", label_ids=labels_used,
                            boundary_resolution=boundary_resolution,
                        ))
                else:
                    if options.get("points_from_match"):
                        if not isinstance(selection, LevelMatches):
                            raise TypeError("points_from_match requires a SliceMatchSet")
                        point_items = []
                        requested_points = options.get("requested_regions")
                        for level, index in selections:
                            if requested_points is not None and level not in requested_points:
                                continue
                            match = selection.matches[level]
                            xy = match.details.get("target_point_xy")
                            if index is not None and xy is not None:
                                point_items.append({"level": level, "index": index, "xy": xy})
                    else:
                        index = selections[0][1]
                        requested_points = options.get("requested_regions")
                        point_items = [
                            {"level": point.get("level"), "index": index, "xy": point.get("xy")}
                            for point in options.get("points", [])
                            if requested_points is None or point.get("level") in requested_points
                        ]
                    for point in point_items:
                        if point["index"] is None or point["xy"] is None:
                            roi_records.append(_missing_roi_record(str(point["level"]), series.view, "matched point or slice is missing", rule="point_bbox"))
                            continue
                        image_shape = _display_plane(series, point["index"]).shape
                        bbox = point_bbox(
                            tuple(point["xy"]), image_shape=image_shape, margin_px=options["margin_px"],
                            reference_size=options.get("reference_size"),
                        )
                        roi_records.append(_roi_record(
                            level=str(point["level"]), view=series.view, index=point["index"], bbox=bbox,
                            image_shape=image_shape, rule="point_bbox",
                        ))

                roi_path = None
                if persist_debug(step_id):
                    roi_path = debug_output_path("roi", step_id, "rois.json", step_dir)
                    _write_json(roi_path, {
                        "coordinate_system": "display-oriented image pixels",
                        "bbox_convention": "xyxy; half-open pixel-edge coordinates",
                        "source_series": str(series.path),
                        "preset": options.get("preset_file"),
                        "rois": roi_records,
                    })
                evidence_items: list[dict[str, Any]] = []
                legacy_evidence_requested = any(
                    isinstance(ref, str) and ref.startswith(f"$steps.{step_id}.evidence")
                    for ref in document.get("outputs", {}).get("evidence", [])
                )
                render_preview = (save_qc or legacy_evidence_requested) and options.get("render_overlay", True)
                if render_preview:
                    overlay_root = (
                        run_dir / "outputs" / "qc" / "overlays"
                        if save_qc else run_dir / "outputs" / "roi_overlays"
                    )
                    overlay_dir = overlay_root / step_id
                    by_index: dict[int, list[dict[str, Any]]] = {}
                    for roi in roi_records:
                        if roi["status"] == "selected" and roi["slice_index_0based"] is not None:
                            by_index.setdefault(roi["slice_index_0based"], []).append(roi)
                    for index, group in by_index.items():
                        level = group[0]["level"] if len(group) == 1 else "multi-level"
                        filename = f"{series.view}_{level.replace('/', '_').replace('-', '_')}_idx{index:03d}_roi.png"
                        overlay_path = overlay_dir / filename
                        _draw_roi_overlay(series, index, group, overlay_path)
                        evidence_items.append({
                            "view": series.view,
                            "level": level if len(group) == 1 else None,
                            "slice_index_0based": index,
                            "image": overlay_path.relative_to(run_dir).as_posix(),
                            "roi_count": len(group),
                            "plane": _plane_metadata(series, index),
                        })
                bundle_path = None
                if save_qc:
                    bundle_path = debug_output_path("roi", step_id, "evidence.json", step_dir)
                    _write_json(bundle_path, {"evidence": evidence_items})
                roi_bundle = RoiBundleArtifact(roi_records, roi_path)
                outputs = {"rois": roi_bundle, "evidence": evidence_items}
                for roi in roi_records:
                    if roi["status"] == "missing":
                        manifest["warnings"].append({
                            "step": step_id, "region": roi["level"], "status": "missing", "reason": roi["reason"]
                        })
                record["artifacts"] = (
                    ([_artifact(roi_path, run_dir, "ROIBundle", count=len(roi_records), view=series.view)]
                     if roi_path is not None else [])
                    + ([_artifact(bundle_path, run_dir, "EvidenceBundle", view=series.view, count=len(evidence_items))]
                       if bundle_path is not None else [])
                    + [
                        _artifact(run_dir / item["image"], run_dir, "ROIOverlay", view=series.view,
                                  slice_index=item["slice_index_0based"], roi_count=item["roi_count"])
                        for item in evidence_items
                    ]
                )
            else:
                raise NotImplementedError(f"paired-NIfTI executor does not implement {key[0]}.{key[1]}")

            step_outputs[step_id] = outputs
            record["status"] = "succeeded"
            record["completed_at"] = _utc_now()
            record["outputs"] = {
                name: {"type": type(value).__name__}
                for name, value in outputs.items()
            }
            _write_manifest(manifest_path, manifest)

        for ref in document["outputs"].get("evidence", []):
            resolved = _resolve(ref, inputs, step_outputs)
            items = resolved if isinstance(resolved, list) else [resolved]
            if not all(isinstance(item, dict) for item in items):
                raise TypeError(f"workflow output {ref!r} did not resolve to evidence objects")
            manifest["workflow_outputs"].append({"reference": ref, "artifacts": items})
        result_refs = document["outputs"].get("results", [])
        if result_refs:
            by_view: dict[str, list[dict[str, Any]]] = {}
            for ref in result_refs:
                bundle = _resolve(ref, inputs, step_outputs)
                if not isinstance(bundle, RoiBundleArtifact):
                    raise TypeError(f"workflow result {ref!r} must resolve to RoiBundleArtifact")
                for roi in bundle.rois:
                    by_view.setdefault(roi["view"], []).append(roi)
            requested_regions = document.get("requested_regions", {})
            views: dict[str, dict[str, Any]] = {}
            for view in ("sagittal_t2", "axial_t2"):
                if view not in by_view:
                    continue
                records = {roi["level"]: roi for roi in by_view[view]}
                level_order = requested_regions.get(view, list(records))
                views[view] = {}
                for level in level_order:
                    roi = records.get(level)
                    if roi is None or roi["status"] != "selected":
                        views[view][level] = None
                    else:
                        views[view][level] = {
                            "slice_index_0based": roi["slice_index_0based"],
                            "bbox_xyxy": roi["bbox_xyxy"],
                        }
            result_path = run_dir / "outputs" / "roi_results.json"
            _write_json(result_path, {
                "schema_version": "1.0",
                "case_id": case_id,
                "coordinate_system": "display-oriented image pixels",
                "bbox_convention": "xyxy; half-open pixel-edge coordinates",
                "views": views,
            })
            result = _artifact(result_path, run_dir, "WorkflowROIResults",
                               views=list(views), roi_count=sum(map(len, views.values())))
            manifest["result"] = result
            manifest["workflow_outputs"].append({
                "type": "WorkflowROIResults", "path": result["path"],
                "roi_count": result["metadata"]["roi_count"], "views": result["metadata"]["views"],
            })
        for ref in document["outputs"].get("artifacts", []):
            artifact = _resolve(ref, inputs, step_outputs)
            if isinstance(artifact, SegmentationArtifact):
                supporting = _artifact(artifact.label_path, run_dir, "SegmentationMask")
            elif isinstance(artifact, RoiBundleArtifact) and artifact.json_path is not None:
                supporting = _artifact(artifact.json_path, run_dir, "ROIBundle", count=len(artifact.rois))
            elif isinstance(artifact, SelectedSliceSet) and artifact.json_path is not None:
                supporting = _artifact(
                    artifact.json_path, run_dir, "SliceSelectionSet", view=artifact.series.view,
                    count=len(artifact.selections),
                )
            elif isinstance(artifact, LevelMatches) and artifact.json_path is not None:
                supporting = _artifact(
                    artifact.json_path, run_dir, "SliceMatchSet", view=artifact.series.view,
                    count=len(artifact.matches),
                )
            else:
                raise TypeError(
                    f"workflow artifact {ref!r} has no persisted QC file; set save_qc=true to export it"
                )
            manifest["supporting_artifacts"].append({"reference": ref, "artifact": supporting})
        manifest["status"] = "succeeded"
        manifest["completed_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
    except Exception as exc:
        if manifest["steps"] and manifest["steps"][-1]["status"] == "running":
            manifest["steps"][-1]["status"] = "failed"
            manifest["steps"][-1]["error"] = f"{type(exc).__name__}: {exc}"
            manifest["steps"][-1]["completed_at"] = _utc_now()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["completed_at"] = _utc_now()
        _write_manifest(manifest_path, manifest)
        raise
    finally:
        for backend_key in owned_backends:
            backend_pool[backend_key].close()
    return run_dir, manifest
