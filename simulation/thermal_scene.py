"""
Synthetic thermal scene generator.

Renders a Gaussian thermal blob (the target) moving across a 2D background
with realistic thermal noise. Output is a sequence of intensity frames that
the EventCamera converts to a DVS-style event stream.

Models:
  - Target: 2D Gaussian intensity anomaly (3-8 K above background)
  - Background: uniform + spatially correlated Gaussian noise (FPN + NETD)
  - Motion: configurable trajectory (linear, sinusoidal, ballistic)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TargetMotion:
    x0: float = 173.0          # starting x (pixels)
    y0: float = 130.0          # starting y
    vx: float = 80.0           # pixels/second
    vy: float = 40.0           # pixels/second
    ax: float = 0.0            # acceleration px/s^2
    ay: float = 0.0
    sigma: float = 6.0         # thermal blob radius (pixels)
    amplitude: float = 5.0     # temperature anomaly (arbitrary units)


class ThermalScene:
    """
    Generates sequential thermal intensity frames.

    Typical usage:
        scene = ThermalScene(width=346, height=260)
        for t in np.arange(0, 1.0, 1/1000):
            frame = scene.render(t)
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        fps: float = 1000.0,
        target: TargetMotion = None,
        noise_std: float = 0.3,
        seed: int = 42,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.target = target or TargetMotion()
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        # Fixed-pattern noise (FPN) — spatially correlated, constant per pixel
        self._fpn = self._rng.normal(0, noise_std * 0.5, (height, width)).astype(np.float32)

        # Pixel coordinate grids for vectorized rendering
        self._yy, self._xx = np.mgrid[0:height, 0:width].astype(np.float32)

    def _target_position(self, t: float) -> tuple[float, float]:
        m = self.target
        x = m.x0 + m.vx * t + 0.5 * m.ax * t ** 2
        y = m.y0 + m.vy * t + 0.5 * m.ay * t ** 2
        return x, y

    def render(self, t: float) -> np.ndarray:
        """
        Render one thermal frame at time t (seconds).
        Returns float32 array of shape (height, width).
        """
        x_t, y_t = self._target_position(t)

        # Gaussian blob
        r2 = (self._xx - x_t) ** 2 + (self._yy - y_t) ** 2
        blob = self.target.amplitude * np.exp(-r2 / (2 * self.target.sigma ** 2))

        # Temporal noise (NETD — noise equivalent temperature difference)
        temporal_noise = self._rng.normal(0, self.noise_std, (self.height, self.width)).astype(np.float32)

        return blob + self._fpn + temporal_noise

    def target_gt(self, t: float) -> tuple[float, float]:
        """Ground-truth target position at time t."""
        return self._target_position(t)
