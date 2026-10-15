"""Run one real TotalSpineSeg inference through the KEPA segmentation contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from kepa.tools.segmentation.contract import SegmentationRequest
from kepa.tools.segmentation.totalspineseg import TotalSpineSegCliBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()

    request = SegmentationRequest(
        input_image=args.input,
        output_dir=args.output_dir,
        view="sagittal_t2",
        structures=("spinal_canal", "vertebrae", "discs"),
        model_id="totalspineseg",
        case_id=args.input.name.removesuffix(".nii.gz").removesuffix(".nii"),
    )
    with TotalSpineSegCliBackend(
        executable=args.executable,
        data_dir=args.data_dir,
        device=args.device,
        max_workers=args.max_workers,
    ) as backend:
        result = backend.segment(request)
    print(f"label_map={result.label_map}")
    print(f"shape_ijk={result.geometry.shape_ijk}")
    print(f"backend={result.backend_id} model={result.model_id}")
    print(f"elapsed_seconds={result.metadata['elapsed_seconds']:.3f}")
    print(f"log_paths={result.log_paths}")


if __name__ == "__main__":
    main()
