"""
SNNViperEngine — the full VIPER pipeline in one class.

Pipeline per event:
    1. IMU feed-forward warp  → stabilized (x', y')
    2. PSF deconvolution      → energy accumulated into event surface
    3. LIF neuron update      → spike or no spike
    4. Stereo coincidence     → if spike, check opposite camera
    5. Depth hit              → emit TrackState(x, y, z, conf, latency_ms)

Designed to run on CPU for simulation, maps to Loihi 2 via Lava for production.
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .imu_warp import IMUWarp
from .psf_deconv import PSFDeconvolver
from .lif_neurons import LIFGrid
from .stereo_depth import StereoCoincidence, Spike, DepthHit


@dataclass
class TrackState:
    x: float          # image x (stabilized, pixels)
    y: float          # image y (stabilized, pixels)
    z: float          # depth (meters); NaN if monocular
    confidence: float
    latency_us: float # microseconds from event arrival to output


class SNNViperEngine:
    """
    VIPER: feed-forward IMU compensation + async PSF deconv + LIF + stereo depth.

    Instantiate once, then call:
        update_imu(omega, dt)          — at 1000 Hz from gyroscope
        process_event(t, x, y, p, cam) — per event from each camera

    Returns a TrackState on every spike that produces a stereo depth hit,
    or None when the event does not trigger a spike / no stereo match.
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        focal_length: float = 200.0,
        baseline: float = 0.10,
        lif_beta: float = 0.95,
        lif_threshold: float = 0.5,
        psf_sigma: float = 1.2,
        dt_max: float = 1e-3,
    ):
        # Two independent warpers (left + right camera may differ in mounting)
        self._warp = [
            IMUWarp(focal_length, width, height),
            IMUWarp(focal_length, width, height),
        ]
        # Two independent PSF accumulators
        self._psf = [
            PSFDeconvolver(width, height, psf_sigma),
            PSFDeconvolver(width, height, psf_sigma),
        ]
        # Two independent LIF grids
        self._lif = [
            LIFGrid(width, height, lif_beta, lif_threshold),
            LIFGrid(width, height, lif_beta, lif_threshold),
        ]
        self._stereo = StereoCoincidence(focal_length, baseline, dt_max)

        self._n_events = 0
        self._n_spikes = 0
        self._n_hits = 0

    def update_imu(self, omega: np.ndarray, dt: float, cam: int = -1) -> None:
        """
        Integrate one IMU reading into the warp matrix.
        cam=-1 updates both cameras (shared IMU is common).
        """
        if cam == -1:
            self._warp[0].update(omega, dt)
            self._warp[1].update(omega, dt)
        else:
            self._warp[cam].update(omega, dt)

    def process_event(
        self,
        t: float,
        x: int,
        y: int,
        polarity: float,
        cam: int,
    ) -> Optional[TrackState]:
        """
        Process one event from camera `cam` (0=left, 1=right).
        Returns TrackState if this event triggers a stereo depth hit.
        """
        t0 = time.perf_counter()
        self._n_events += 1

        # Stage 1: IMU feed-forward stabilization
        x_s, y_s = self._warp[cam].warp_event(x, y)
        xi, yi = int(round(x_s)), int(round(y_s))

        # Stage 2: PSF deconvolution accumulation
        self._psf[cam].accumulate(xi, yi, polarity)
        current = self._psf[cam].get_surface()[yi, xi]

        # Stage 3: LIF neuron update
        fired = self._lif[cam].step_event(xi, yi, current)

        if not fired:
            return None

        self._n_spikes += 1

        # Stage 4: Stereo coincidence
        spike = Spike(t=t, x=xi, y=yi, cam=cam)
        hit: Optional[DepthHit] = self._stereo.push_spike(spike)

        if hit is None:
            # Monocular spike — no depth yet
            return None

        self._n_hits += 1
        latency_us = (time.perf_counter() - t0) * 1e6

        return TrackState(
            x=hit.x,
            y=hit.y,
            z=hit.z,
            confidence=hit.confidence,
            latency_us=latency_us,
        )

    @property
    def stats(self) -> dict:
        return {
            "events": self._n_events,
            "spikes": self._n_spikes,
            "depth_hits": self._n_hits,
            "spike_rate": self._n_spikes / max(1, self._n_events),
        }
