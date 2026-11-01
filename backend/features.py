"""Feature extraction for the RandomForest classifier.

The forest expects a 2276-dimensional vector. That length, and the structure of
the trained model, pin down the layout:

    [0:1764]     HOG  - 9 orientations, 16x16 px/cell, 2x2 cells/block, on the
                        128x128 grayscale image (7*7 blocks * 2*2 cells * 9 = 1764)
    [1764:2276]  8x8x8 joint RGB colour histogram, L2-normalised (8**3 = 512)

Evidence for that split rather than the reverse: every one of the model's 258
zero-importance features falls inside [1764:2276], which is what empty colour
histogram bins look like. The HOG block has no zero-importance features at all.

CHANNEL ORDER is the one genuinely ambiguous choice. OpenCV's `imread` yields
BGR, so a `cv2.calcHist([img], [0,1,2], ...)` recipe bins B,G,R in that order,
while PIL/skimage yield RGB. It only affects the histogram block (the grayscale
conversion applies the matching luma weights either way), but getting it wrong
corrupts 512 of the 2276 features. It is configurable via `channel_order` in
labels.json so it can be flipped without touching code.
"""

import io

import numpy as np
from PIL import Image
from skimage.feature import hog

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = "L2-Hys"
HIST_BINS = 8

HOG_LENGTH = 1764
HIST_LENGTH = HIST_BINS**3
EXPECTED_LENGTH = HOG_LENGTH + HIST_LENGTH  # 2276

# cv2's BGR2GRAY / skimage's rgb2gray both use the ITU-R 601-2 luma weights.
_LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def load_rgb(image_bytes: bytes, size: int) -> np.ndarray:
    """Decode to a uint8 RGB array of (size, size, 3)."""
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _hog_block(rgb: np.ndarray) -> np.ndarray:
    # Weighted sum on RGB; equivalent to cv2.cvtColor(bgr, COLOR_BGR2GRAY).
    gray = (rgb.astype(np.float64) * _LUMA_RGB).sum(axis=2) / 255.0
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm=HOG_BLOCK_NORM,
        feature_vector=True,
    )


def _hist_block(rgb: np.ndarray, channel_order: str) -> np.ndarray:
    """8x8x8 joint colour histogram, L2-normalised (cv2.normalize's default)."""
    img = rgb[..., ::-1] if channel_order == "bgr" else rgb
    pixels = img.reshape(-1, 3)
    hist, _ = np.histogramdd(
        pixels,
        bins=(HIST_BINS, HIST_BINS, HIST_BINS),
        range=((0, 256), (0, 256), (0, 256)),
    )
    hist = hist.astype(np.float32).ravel()
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def extract(image_bytes: bytes, size: int = 128, channel_order: str = "bgr") -> np.ndarray:
    """bytes -> (1, 2276) float32 feature matrix ready for RandomForest.predict."""
    rgb = load_rgb(image_bytes, size)
    vector = np.hstack([_hog_block(rgb), _hist_block(rgb, channel_order)])

    if vector.size != EXPECTED_LENGTH:
        raise ValueError(
            f"feature vector is {vector.size}, expected {EXPECTED_LENGTH}; "
            "the extraction settings do not match the trained model"
        )
    return vector.astype(np.float32)[None, :]
