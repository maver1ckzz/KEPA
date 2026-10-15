"""Compare native-grid axial masks from two KEPA backend versions."""

from __future__ import annotations

import argparse
import json

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    new = nib.load(args.new)
    reference = nib.load(args.reference)
    if new.shape != reference.shape or not np.allclose(new.affine, reference.affine):
        raise ValueError("masks have different grids")
    new_mask = np.asarray(new.dataobj) > 0
    reference_mask = np.asarray(reference.dataobj) > 0
    intersection = int(np.count_nonzero(new_mask & reference_mask))
    union = int(np.count_nonzero(new_mask | reference_mask))
    print(json.dumps({
        "new_voxels": int(np.count_nonzero(new_mask)),
        "reference_voxels": int(np.count_nonzero(reference_mask)),
        "different_voxels": int(np.count_nonzero(new_mask != reference_mask)),
        "iou": intersection / union if union else 1.0,
    }))


if __name__ == "__main__":
    main()
