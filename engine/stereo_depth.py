"""
Stereoscopic temporal coincidence depth estimation.

Unlike traditional stereo (which matches texture), VIPER uses spike TIMING.
When the left and right cameras both spike at homologous pixel locations
within a time window Δt_max, their timing difference encodes depth.

Physics:
    disparity d = x_L - x_R  (pixels)
    depth Z = f * B / d       (meters)
    where f = focal length (px), B = stereo baseline (m)

The Δt coincidence window replaces block matching — no texture needed.
This works even on featureless thermal scenes.
"""
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


@dataclass
class Spike:
    t: float   # timestamp (seconds)
    x: int
    y: int
    cam: int   # 0 = left, 1 = right


@dataclass
class DepthHit:
    x: float
    y: float
    z: float          # depth in meters
    confidence: float
    t: float


class StereoCoincidence:
    """
    Matches spikes from left and right cameras within a time window.

    For each incoming spike, searches the opposite camera's recent spike
    buffer for a spatially homologous spike (same epipolar row ±ε pixels)
    within Δt_max seconds.
    """

    def __init__(
        self,
        focal_length: float = 200.0,    # pixels
        baseline: float = 0.10,          # meters (10 cm stereo baseline)
        dt_max: float = 1e-3,            # 1 ms coincidence window
        epipolar_tol: int = 2,           # pixels vertical tolerance
        buffer_size: int = 500,
    ):
        self.f = focal_length
        self.B = baseline
        self.dt_max = dt_max
        self.epipolar_tol = epipolar_tol
        # Ring buffers — one per camera
        self._buffers: list[deque[Spike]] = [deque(maxlen=buffer_size), deque(maxlen=buffer_size)]

    def push_spike(self, spike: Spike) -> Optional[DepthHit]:
        """
        Add a spike and check for coincidence with the opposite camera.
        Returns a DepthHit if a valid stereo pair is found, else None.
        """
        opposite = 1 - spike.cam
        opp_buf = self._buffers[opposite]

        best: Optional[DepthHit] = None
        best_dt = self.dt_max

        for candidate in opp_buf:
            dt = abs(spike.t - candidate.t)
            if dt > self.dt_max:
                continue
            if abs(spike.y - candidate.y) > self.epipolar_tol:
                continue

            # Assign left/right correctly regardless of which cam fires first
            if spike.cam == 0:
                x_L, x_R = spike.x, candidate.x
            else:
                x_L, x_R = candidate.x, spike.x

            disparity = float(x_L - x_R)
            if abs(disparity) < 1e-3:
                continue  # degenerate: same column, depth → ∞

            z = self.f * self.B / abs(disparity)
            # x_world: center between the two detections in image coords
            x_world = (x_L + x_R) / 2.0
            y_world = (spike.y + candidate.y) / 2.0

            # Confidence inversely proportional to Δt and y epipolar error
            conf = 1.0 - (dt / self.dt_max)

            if dt < best_dt:
                best_dt = dt
                best = DepthHit(x=x_world, y=y_world, z=z, confidence=conf, t=spike.t)

        self._buffers[spike.cam].append(spike)
        return best
