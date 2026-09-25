"""Lossless uint16 depth <-> 3-channel uint8 image, for LeRobot ``image`` features.

LeRobot only stores 3-channel 8-bit images, so each depth value ``d`` is split across two
channels: R = ``d >> 8``, G = ``d & 0xFF``, B = 0. Written as PNG this is exact.
"""
from __future__ import annotations

import numpy as np

DEPTH_ENCODING = "uint16_packed_rg8"
"""Identifier recorded in the dataset sidecar for this packing scheme."""


def pack_depth(depth: np.ndarray) -> np.ndarray:
    """``(H, W)`` uint16 depth -> ``(H, W, 3)`` uint8 image (R = high byte, G = low byte)."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected (H, W) depth, got shape {depth.shape}.")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth, got dtype {depth.dtype}.")
    packed = np.zeros((*depth.shape, 3), dtype=np.uint8)
    packed[..., 0] = depth >> 8
    packed[..., 1] = depth & 0xFF
    return packed


def unpack_depth(image) -> np.ndarray:
    """Inverse of :func:`pack_depth`, back to ``(H, W)`` uint16.

    Accepts either the packed ``(H, W, 3)`` uint8 array, or what ``LeRobotDataset`` returns
    when loading an ``image`` feature: a float ``(3, H, W)`` tensor/array in ``[0, 1]``.
    """
    if hasattr(image, "detach"):  # torch.Tensor, without importing torch here
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]):
        raise ValueError(f"Expected a 3-channel image, got shape {image.shape}.")
    if image.shape[-1] != 3:  # (3, H, W) -> (H, W, 3)
        image = np.moveaxis(image, 0, -1)
    if image.dtype == np.uint8:
        hi, lo = image[..., 0].astype(np.uint16), image[..., 1].astype(np.uint16)
    else:
        hi = np.rint(image[..., 0] * 255.0).astype(np.uint16)
        lo = np.rint(image[..., 1] * 255.0).astype(np.uint16)
    return (hi << 8) | lo
