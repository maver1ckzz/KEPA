"""Exercise a resident TotalSpineSeg backend on one or more serial requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from kepa.tools.segmentation import SegmentationRequest, TotalSpineSegPythonBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 1:
        raise ValueError("repeat must be at least one")
    summaries = []
    with TotalSpineSegPythonBackend(args.data_dir, device=args.device) as backend:
        for index in range(args.repeat):
            result = backend.segment(SegmentationRequest(
                input_image=args.input, output_dir=args.output / f"request_{index + 1}",
                view="sagittal_t2", model_id="totalspineseg",
                structures=("vertebrae", "intervertebral_discs", "spinal_canal"),
                case_id=f"smoke_{index + 1}",
            ))
            image = nib.load(str(result.label_map))
            values = np.unique(np.asarray(image.dataobj)).astype(int).tolist()
            summaries.append({
                "label_map": str(result.label_map),
                "shape": list(result.geometry.shape_ijk),
                "labels": values,
                "elapsed_seconds": result.metadata["elapsed_seconds"],
                "backend_version": result.backend_version,
                "model_resident": result.metadata["keeps_model_resident"],
            })
            print(json.dumps(summaries[-1]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
