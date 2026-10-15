"""Exact-data axial NIfTI overlay for visual mask quality control."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _display(array: np.ndarray) -> np.ndarray:
    return np.flipud(array.T)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--canal", type=Path, required=True)
    parser.add_argument("--cord", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    image = np.asarray(nib.load(str(args.image)).dataobj)
    canal = np.asarray(nib.load(str(args.canal)).dataobj) > 0
    cord = np.asarray(nib.load(str(args.cord)).dataobj) > 0
    if image.shape != canal.shape or image.shape != cord.shape or image.ndim != 3:
        raise ValueError("the image and both masks must share one 3-D voxel grid")
    n_slices = image.shape[2]
    tile = 256
    columns = 6
    rows = int(np.ceil(n_slices / columns))
    canvas = Image.new("RGB", (columns * tile, rows * (tile + 28)), (8, 8, 8))
    pen = ImageDraw.Draw(canvas)
    for index in range(n_slices):
        frame = _display(image[:, :, index])
        lower, upper = np.percentile(frame, (1, 99))
        gray = np.uint8(np.clip((frame - lower) / max(upper - lower, 1e-6), 0, 1) * 255)
        rgb = np.repeat(gray[..., None], 3, axis=2)
        canal_mask = _display(canal[:, :, index])
        cord_mask = _display(cord[:, :, index])
        canal_border = canal_mask & ~ndimage.binary_erosion(canal_mask)
        cord_border = cord_mask & ~ndimage.binary_erosion(cord_mask)
        rgb[canal_border] = (0, 255, 70)
        rgb[cord_border] = (255, 40, 230)
        panel = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.BICUBIC)
        x, y = (index % columns) * tile, (index // columns) * (tile + 28)
        canvas.paste(panel, (x, y))
        pen.text((x + 5, y + tile + 5), f"slice {index}: canal {canal_mask.sum()} / cord {cord_mask.sum()}", fill="white")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
