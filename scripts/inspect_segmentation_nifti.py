"""Print basic NIfTI geometry and label statistics for a segmentation smoke test."""

from __future__ import annotations

import argparse

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--reference", help="optional reference label map for voxel comparison")
    args = parser.parse_args()
    image = nib.load(args.input)
    segmentation = nib.load(args.segmentation)
    labels, counts = np.unique(np.asarray(segmentation.dataobj), return_counts=True)
    print(f"input_shape={image.shape}")
    print(f"segmentation_shape={segmentation.shape}")
    print(f"affine_equal={np.allclose(image.affine, segmentation.affine)}")
    print(f"input_affine=\n{image.affine}")
    print(f"segmentation_affine=\n{segmentation.affine}")
    print(f"label_counts={dict(zip(labels.tolist(), counts.tolist()))}")
    if args.reference:
        reference = nib.load(args.reference)
        same_grid = segmentation.shape == reference.shape and np.allclose(segmentation.affine, reference.affine)
        print(f"reference_grid_equal={same_grid}")
        if same_grid:
            print(f"reference_voxel_agreement={np.mean(np.asarray(segmentation.dataobj) == np.asarray(reference.dataobj)):.8f}")


if __name__ == "__main__":
    main()
