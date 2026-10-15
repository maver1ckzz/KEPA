"""Smoke-test resident axial nnU-Net masks on one anatomical axial NIfTI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from kepa.tools.segmentation import AxialNnunetBackend, PersistentPythonBackend, SegmentationRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=["canal_seg", "sc_epi"], required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--worker-python", type=Path, help="run through KEPA's persistent JSON worker")
    args = parser.parse_args()
    structure = "spinal_canal" if args.model_id == "canal_seg" else "spinal_cord"
    backend = (
        PersistentPythonBackend(
            args.worker_python, args.data_dir, model_backend="axial_nnunet_python",
            device=args.device, log_path=args.output.parent / f"{args.output.name}_worker.log",
        ) if args.worker_python else AxialNnunetBackend(args.data_dir, device=args.device)
    )
    with backend:
        result = backend.segment(SegmentationRequest(
            input_image=args.input, output_dir=args.output,
            view="axial_t2", model_id=args.model_id, structures=(structure,),
        ))
    source = nib.load(str(args.input))
    image = nib.load(str(result.label_map))
    data = np.asarray(image.dataobj)
    summary = {
        "input": str(args.input), "model_id": args.model_id,
        "label_map": str(result.label_map),
        "shape": list(image.shape),
        "native_grid": image.shape == source.shape and bool(np.allclose(image.affine, source.affine)),
        "foreground_voxels": int(np.count_nonzero(data)),
        "unique_labels": np.unique(data).astype(int).tolist(),
        "metadata": result.metadata,
    }
    (args.output / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
