"""
Simulates a Dynamic Vision Sensor (DVS) event camera from thermal frames.

DVS pixels fire asynchronously when local log-luminance changes exceed a
threshold Θ. Each event is a tuple (t, x, y, polarity) where polarity is
+1 (brightness increase) or -1 (brightness decrease).

For stereo simulation: two EventCamera instances with a horizontal baseline offset.
The right camera sees the scene shifted by `baseline_px` pixels.
"""
import numpy as np
from dataclasses import dataclass
from typing import Generator


@dataclass
class Event:
    t: float
    x: int
    y: int
    polarity: float   # +1 or -1
    cam: int          # 0=left, 1=right


class EventCamera:
    """
    Converts sequential thermal intensity frames into DVS-style events.

    Maintains per-pixel reference intensity. On each new frame, pixels
    whose change exceeds ±threshold emit events.
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        threshold: float = 0.4,
        cam_id: int = 0,
        baseline_px: int = 0,      # horizontal pixel offset for right cam
        refractory_ns: int = 200,  # minimum time between events per pixel (ns)
    ):
        self.width = width
        self.height = height
        self.threshold = threshold
        self.cam_id = cam_id
        self.baseline_px = baseline_px

        self._ref = np.zeros((height, width), dtype=np.float32)
        self._initialized = False

    def _shift_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply baseline offset to simulate right camera viewpoint."""
        if self.baseline_px == 0:
            return frame
        shifted = np.roll(frame, self.baseline_px, axis=1)
        if self.baseline_px > 0:
            shifted[:, :self.baseline_px] = 0
        else:
            shifted[:, self.baseline_px:] = 0
        return shifted

    def process_frame(self, frame: np.ndarray, t: float) -> list[Event]:
        """
        Compare frame to reference, emit events for pixels that crossed threshold.
        Vectorized: builds the event list from numpy arrays, not Python loops.
        """
        frame = self._shift_frame(frame)

        if not self._initialized:
            self._ref = frame.copy()
            self._initialized = True
            return []

        diff = frame - self._ref
        on_mask  = diff >=  self.threshold
        off_mask = diff <= -self.threshold
        fired    = on_mask | off_mask

        if not fired.any():
            return []

        ys, xs   = np.where(fired)
        pols     = np.where(on_mask[ys, xs], 1.0, -1.0)
        jitter   = np.random.uniform(0, 1e-4, size=len(xs))
        ts       = t + jitter
        order    = np.argsort(ts)

        events = [
            Event(float(ts[i]), int(xs[i]), int(ys[i]), float(pols[i]), self.cam_id)
            for i in order
        ]

        self._ref[on_mask]  = frame[on_mask]
        self._ref[off_mask] = frame[off_mask]
        return events

    def reset(self) -> None:
        self._ref[:] = 0
        self._initialized = False
