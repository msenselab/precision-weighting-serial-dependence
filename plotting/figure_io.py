"""Install rendered figure pairs without metadata-only repository churn."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


def png_pixels_equal(first: Path, second: Path) -> bool:
    """Return whether two PNG files have identical dimensions and pixels."""
    if not first.exists() or not second.exists():
        return False
    try:
        first_pixels = mpimg.imread(first)
        second_pixels = mpimg.imread(second)
    except OSError:
        return False
    return first_pixels.shape == second_pixels.shape and np.array_equal(
        first_pixels, second_pixels
    )


def install_figure_pair(staging: Path, destination: Path, stem: str) -> bool:
    """Copy a staged PNG/PDF pair only when its rendered PNG content changed."""
    staged_png = staging / f"{stem}.png"
    destination_png = destination / f"{stem}.png"
    if not staged_png.exists():
        raise FileNotFoundError(f"Missing staged figure: {staged_png}")

    if png_pixels_equal(staged_png, destination_png):
        staged_pdf = staging / f"{stem}.pdf"
        destination_pdf = destination / f"{stem}.pdf"
        if staged_pdf.exists() and not destination_pdf.exists():
            shutil.copyfile(staged_pdf, destination_pdf)
            return True
        return False

    copied = False
    for ext in ("png", "pdf"):
        src = staging / f"{stem}.{ext}"
        if src.exists():
            shutil.copyfile(src, destination / f"{stem}.{ext}")
            copied = True
    return copied
