"""Command-line entry point for KEPA workflow tools."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from .execution.batch import load_batch_manifest, make_paired_manifest, parse_gpu_ids, start_backend_with_gpu_fallback
from .execution.single_nifti import build_single_nifti_workflow
from .execution.paired_nifti import build_paired_nifti_workflow
from .tools.segmentation import PersistentPythonBackend
from .workflow.registry import list_operations
from .workflow.validator import normalize_workflow, validate_workflow


def _workflow_validate(path: Path) -> int:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read workflow JSON: {exc}", file=sys.stderr)
        return 2
    errors = validate_workflow(document)
    if errors:
        print(f"INVALID workflow: {path}")
        for error in errors:
            print(f"- {error}")
        return 2
    print(f"VALID workflow: {document['id']} ({len(document['steps'])} steps)")
    print("Tools:", ", ".join(sorted({step["tool"] for step in document["steps"]})))
    return 0

def _evidence_make_paired_manifest(args: argparse.Namespace) -> int:
    try:
        path = make_paired_manifest(args.sag_dir, args.axial_root, args.out)
        cases = load_batch_manifest(path)
    except Exception as exc:
        print(f"ERROR: cannot make paired manifest: {exc}", file=sys.stderr)
        return 2
    print(f"Manifest: {path} ({len(cases)} cases)")
    return 0


def _evidence_build(args: argparse.Namespace) -> int:
    try:
        manifest = build_single_nifti_workflow(
            workflow_path=args.workflow,
            input_nii=args.input_nii,
            level=args.level,
            output_dir=args.out,
            sct_executable=args.sct_deepseg,
        )
    except Exception as exc:
        print(f"ERROR: evidence workflow failed: {exc}", file=sys.stderr)
        return 2
    print(f"Evidence workflow complete: {args.out.absolute() / 'evidence.json'}")
    for item in manifest["evidence"]:
        print(f"{item['view']}: {args.out.absolute() / item['image']} (slice {item['slice_index']})")
    return 0


def _evidence_build_paired(args: argparse.Namespace) -> int:
    try:
        run_dir, manifest = build_paired_nifti_workflow(
            workflow_path=args.workflow,
            sagittal_nii=args.sag_nii,
            axial_nii=args.ax_nii,
            runs_root=args.runs_root,
            case_id=args.case_id,
            run_id=args.run_id,
            totalspineseg_executable=args.totalspineseg,
            totalspineseg_data_dir=args.totalspineseg_data,
            axial_model_data_dir=args.axial_model_data,
            segmentation_python=args.segmentation_python,
            reuse_sagittal_segmentation=args.reuse_sag_segmentation,
        )
    except Exception as exc:
        print(f"ERROR: paired-NIfTI evidence workflow failed: {exc}", file=sys.stderr)
        return 2
    print(f"Workflow status: {manifest['status']}")
    print(f"Run directory: {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    if manifest.get("result"):
        print(f"Result JSON: {run_dir / manifest['result']['path']}")
    for group in manifest["workflow_outputs"]:
        for item in group.get("artifacts", []):
            print(f"{item['view']}: {run_dir / item['image']} (slice {item['slice_index_0based']})")
    return 0


def _evidence_build_paired_batch(args: argparse.Namespace) -> int:
    """Run manifest cases serially while keeping model workers resident."""
    try:
        cases = load_batch_manifest(args.manifest)
        gpu_ids = parse_gpu_ids(args.gpus)
        workflow = normalize_workflow(json.loads(args.workflow.read_text(encoding="utf-8")))
        errors = validate_workflow(workflow)
        if errors:
            raise ValueError("invalid workflow: " + "; ".join(errors))
        segmentation_steps = [
            step for step in workflow["steps"]
            if step["tool"] == "segmentation" and step["operation"] == "segment"
        ]
        backend_ids = {step["options"]["backend"] for step in segmentation_steps}
        if not backend_ids or not backend_ids.issubset({"totalspineseg_python", "axial_nnunet_python"}):
            raise ValueError("batch worker supports only resident totalspineseg_python and axial_nnunet_python steps")
        model_roots = {
            "totalspineseg_python": args.totalspineseg_data,
            "axial_nnunet_python": args.axial_model_data,
        }
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        summary_dir = args.runs_root.absolute() / "_batch_summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{batch_id}.json"
        summary = {"batch_id": batch_id, "workflow_id": workflow["id"],
                   "manifest": str(args.manifest.absolute()), "requested_gpus": list(gpu_ids),
                   "model_assignments": {}, "status": "running", "cases": []}
        def save_summary() -> None:
            temporary = summary_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            temporary.replace(summary_path)
        save_summary()
        backends, assignments = {}, {}
        with ExitStack() as stack:
            for backend_id in sorted(backend_ids, key=lambda backend: (backend != "totalspineseg_python", backend)):
                model_root = model_roots[backend_id]
                if model_root is None:
                    raise ValueError(f"model data path is required for {backend_id}")
                def factory(gpu_id: str) -> PersistentPythonBackend:
                    return PersistentPythonBackend(
                        args.segmentation_python, model_root, model_backend=backend_id,
                        visible_gpu=gpu_id,
                        log_path=args.runs_root.absolute() / "_batch_workers" / batch_id /
                                 f"{backend_id}_gpu{gpu_id}.log")
                backend, gpu_id = start_backend_with_gpu_fallback(factory, gpu_ids)
                backends[backend_id] = stack.enter_context(backend)
                assignments[backend_id] = gpu_id
                summary["model_assignments"] = assignments.copy()
                save_summary()
            for case in cases:
                item = {"case_id": case.case_id, "status": "failed"}
                try:
                    run_dir, manifest = build_paired_nifti_workflow(
                        workflow_path=args.workflow, sagittal_nii=case.sagittal_nii,
                        axial_nii=list(case.axial_niis), runs_root=args.runs_root,
                        case_id=case.case_id, segmentation_backends=backends)
                    item.update({"status": manifest["status"], "run_dir": str(run_dir)})
                    if item["status"] != "succeeded":
                        item["error"] = manifest.get("error", "workflow did not succeed")
                    print(f"{case.case_id}: {item['status']} {run_dir}", flush=True)
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"{case.case_id}: failed: {item['error']}", file=sys.stderr, flush=True)
                summary["cases"].append(item)
                save_summary()
        failures = [item for item in summary["cases"] if item["status"] != "succeeded"]
        summary["status"] = "completed_with_errors" if failures else "succeeded"
        summary["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_summary()
        print(f"Batch summary: {summary_path}", flush=True)
        return 1 if failures else 0
    except Exception as exc:
        if "summary" in locals() and "save_summary" in locals():
            summary["status"] = "failed"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            summary["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            save_summary()
        print(f"ERROR: paired batch workflow failed: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kepa", description="KEPA evidence workflow framework")
    subparsers = parser.add_subparsers(dest="command", required=True)
    workflow_parser = subparsers.add_parser("workflow", help="inspect or validate a workflow")
    workflow_subparsers = workflow_parser.add_subparsers(dest="workflow_command", required=True)
    validate_parser = workflow_subparsers.add_parser("validate", help="validate JSON schema and tool compatibility")
    validate_parser.add_argument("path", type=Path)
    workflow_subparsers.add_parser("list-tools", help="list registered tool operations")
    evidence_parser = subparsers.add_parser("evidence", help="build image evidence from MRI data")
    evidence_subparsers = evidence_parser.add_subparsers(dest="evidence_command", required=True)
    build_parser = evidence_subparsers.add_parser(
        "build", help="execute the initial two-view workflow from one 3D NIfTI"
    )
    build_parser.add_argument("--input-nii", type=Path, required=True)
    build_parser.add_argument("--level", required=True, help="e.g. C4 or C4-5")
    build_parser.add_argument("--workflow", type=Path, required=True)
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument("--sct-deepseg", default="sct_deepseg", help="SCT executable or full path")
    manifest_parser = evidence_subparsers.add_parser(
        "make-paired-manifest", help="scan sagittal files and axial per-case directories"
    )
    manifest_parser.add_argument("--sag-dir", type=Path, required=True)
    manifest_parser.add_argument("--axial-root", type=Path, required=True)
    manifest_parser.add_argument("--out", type=Path, required=True)
    paired_parser = evidence_subparsers.add_parser(
        "build-paired", help="execute a JSON workflow from one sagittal and one or more axial NIfTIs"
    )
    paired_parser.add_argument("--sag-nii", type=Path, required=True)
    paired_parser.add_argument(
        "--ax-nii", type=Path, nargs="+", required=True,
        help="one or more axial NIfTIs; each is processed independently, then candidates are combined",
    )
    paired_parser.add_argument("--workflow", type=Path, required=True)
    paired_parser.add_argument("--runs-root", type=Path, required=True)
    paired_parser.add_argument("--case-id")
    paired_parser.add_argument("--run-id")
    paired_parser.add_argument(
        "--totalspineseg", default="totalspineseg",
        help="upstream TotalSpineSeg executable (not SCT's sct_deepseg wrapper)",
    )
    paired_parser.add_argument(
        "--totalspineseg-data", type=Path,
        help="preinstalled upstream model-data directory (or set TOTALSPINESEG_DATA); KEPA does not download weights",
    )
    paired_parser.add_argument(
        "--axial-model-data", type=Path,
        help="directory containing external model_seg_canal_t2w and/or model_seg_sc_epi_nnunet checkpoints",
    )
    paired_parser.add_argument(
        "--segmentation-python", type=Path, default=Path(sys.executable),
        help="Python for the resident model worker (defaults to the current KEPA Python)",
    )
    paired_parser.add_argument(
        "--reuse-sag-segmentation", type=Path,
        help="reuse a prior sagittal label NIfTI instead of rerunning SCT (useful for workflow regression)",
    )
    batch_parser = evidence_subparsers.add_parser(
        "build-paired-batch", help="run manifest cases sequentially with resident models and GPU fallback"
    )
    batch_parser.add_argument("--manifest", "--cases", dest="manifest", type=Path, required=True,
                              help="manifest with case_id and inputs.sag/inputs.axl")
    batch_parser.add_argument("--workflow", type=Path, required=True)
    batch_parser.add_argument("--runs-root", type=Path, required=True)
    batch_parser.add_argument("--segmentation-python", type=Path, default=Path(sys.executable))
    batch_parser.add_argument("--totalspineseg-data", type=Path)
    batch_parser.add_argument("--axial-model-data", type=Path)
    batch_parser.add_argument("--gpus", required=True,
                              help="ordered physical GPUs available for model loading, e.g. 3,4,6")
    args = parser.parse_args(argv)

    if args.command == "workflow" and args.workflow_command == "validate":
        return _workflow_validate(args.path)
    if args.command == "workflow" and args.workflow_command == "list-tools":
        print("\n".join(list_operations()))
        return 0
    if args.command == "evidence" and args.evidence_command == "build":
        return _evidence_build(args)
    if args.command == "evidence" and args.evidence_command == "make-paired-manifest":
        return _evidence_make_paired_manifest(args)
    if args.command == "evidence" and args.evidence_command == "build-paired":
        return _evidence_build_paired(args)
    if args.command == "evidence" and args.evidence_command == "build-paired-batch":
        return _evidence_build_paired_batch(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
