"""
Asynchronous Point Spread Function deconvolution.

Thermal cameras spread energy across neighboring pixels (Gaussian PSF).
Instead of deconvolving a whole frame, we apply the inverse PSF kernel
to each event as it arrives — zero frame-buffer delay.

The 7×7 Gaussian kernel is the empirical thermal PSF for uncooled
microbolometer arrays (typical σ ≈ 1.2 pixels at f/1.0).
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def make_psf_kernel(sigma: float = 1.2, size: int = 7) -> np.ndarray:
    """Normalized 2D Gaussian kernel representing the thermal sensor PSF."""
    center = size // 2
    y, x = np.mgrid[-center:center + 1, -center:center + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


class PSFDeconvolver:
    """
    Sparse PSF accumulator — applies the kernel footprint on each event.

    Maintains a float32 event surface (same resolution as sensor).
    Each incoming event adds the PSF kernel centered at its (x, y).
    This is the spatial deconvolution step: instead of measuring a blurred
    point, we reconstruct where the energy actually came from.
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        sigma: float = 1.2,
        kernel_size: int = 7,
        decay: float = 0.99,
    ):
        self.width = width
        self.height = height
        self.decay = decay
        self._kernel = make_psf_kernel(sigma, kernel_size)
        self._half = kernel_size // 2
        self._surface = np.zeros((height, width), dtype=np.float32)

    def accumulate(self, x: int, y: int, polarity: float) -> None:
        """
        Stamp the PSF kernel at (x, y) with the event's polarity weight.
        ON events (+1) add energy; OFF events (-1) subtract.
        """
        x0 = max(0, x - self._half)
        x1 = min(self.width, x + self._half + 1)
        y0 = max(0, y - self._half)
        y1 = min(self.height, y + self._half + 1)

        kx0 = x0 - (x - self._half)
        ky0 = y0 - (y - self._half)
        kx1 = kx0 + (x1 - x0)
        ky1 = ky0 + (y1 - y0)

        self._surface[y0:y1, x0:x1] += polarity * self._kernel[ky0:ky1, kx0:kx1]

    def decay_surface(self) -> None:
        """Exponential temporal decay — keeps surface fresh."""
        self._surface *= self.decay

    def get_surface(self) -> np.ndarray:
        return self._surface

    def reset(self) -> None:
        self._surface[:] = 0.0
