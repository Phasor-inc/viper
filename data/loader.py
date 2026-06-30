"""
Real event camera data loader — VIPER real-data validation layer.

Supports any H5 event dataset with (t, x, y, p) arrays.
Tested against:
  - Town03 CARLA event dataset (640x480, ~20M events, Unix µs timestamps)
  - Standard DVS346 aedat4 exports

Ground truth loader parses stamped_groundtruth.txt (TUM-RGBD format):
  # timestamp[seconds] px py pz qx qy qz qw
  1.732750949e+09 ...

IMU derivation: angular velocity ω is computed from successive quaternion poses
via finite-difference approximation:  ω ≈ 2·Im(q2·conj(q1)) / Δt
"""
from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional
import warnings

try:
    import hdf5plugin   # registers compression codecs (bitshuffle, blosc, lz4, etc.)
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py / hdf5plugin not installed. pip install h5py hdf5plugin")


@dataclass
class EventBatch:
    """Slice of events from a real event camera file."""
    t: np.ndarray        # timestamps (seconds, float64, normalized to start at 0)
    x: np.ndarray        # x coords (uint16)
    y: np.ndarray        # y coords (uint16)
    p: np.ndarray        # polarity: +1 (ON) or −1 (OFF)
    width: int
    height: int
    source: str


@dataclass
class IMUFromGT:
    """Angular velocity derived from camera pose ground truth."""
    t: np.ndarray        # timestamps (seconds, normalized)
    omega: np.ndarray    # (N, 3) rad/s


@dataclass
class GroundTruthPoses:
    t: np.ndarray        # seconds, normalized
    pos: np.ndarray      # (N, 3) xyz world position (meters)
    quat: np.ndarray     # (N, 4) quaternions (qx, qy, qz, qw)
    imu: IMUFromGT       # derived angular velocities


def _detect_schema(f) -> dict[str, str]:
    """
    Auto-detect the key names for (t, x, y, p) in an H5 file.
    Handles both flat and nested layouts.
    """
    candidates = {
        't': ['t', 'timestamps', 'time', 'ts', 'events/t'],
        'x': ['x', 'x_pos', 'col',  'events/x'],
        'y': ['y', 'y_pos', 'row',  'events/y'],
        'p': ['p', 'pol', 'polarity', 'polarities', 'events/p'],
    }
    schema = {}
    for field, keys in candidates.items():
        for k in keys:
            if k in f:
                schema[field] = k
                break
        if field not in schema:
            raise KeyError(
                f"Cannot find '{field}' in H5 file. "
                f"Keys present: {list(f.keys())}"
            )
    return schema


class EventH5Dataset:
    """
    Lazy loader for event camera H5 files.

    Usage:
        ds = EventH5Dataset('/path/to/events.h5')
        for batch in ds.iter_windows(window_s=0.01):
            # batch.t, batch.x, batch.y, batch.p

    Or load all at once (memory-permitting):
        batch = ds.load_all(max_events=5_000_000)
    """

    def __init__(self, path: str):
        if not H5PY_AVAILABLE:
            raise RuntimeError("h5py required: pip install h5py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Event file not found: {path}")

        self.path = path
        self._schema: Optional[dict] = None
        self._n_events: Optional[int] = None
        self._t0: Optional[float] = None
        self.width:  Optional[int] = None
        self.height: Optional[int] = None

        self._probe()

    def _probe(self):
        with h5py.File(self.path, 'r') as f:
            self._schema = _detect_schema(f)
            self._n_events = len(f[self._schema['t']])
            # Sample 2 timestamps to get t0 and detect time unit
            t_sample = f[self._schema['t']][:2]
            self._t0 = float(t_sample[0])
            # Detect unit: if t0 > 1e12, timestamps are in microseconds
            self._t_scale = 1e-6 if self._t0 > 1e12 else 1.0
            # Resolution from max x/y in a sample
            sample = min(100_000, self._n_events)
            x_s = np.array(f[self._schema['x']][:sample])
            y_s = np.array(f[self._schema['y']][:sample])
            self.width  = int(x_s.max()) + 1
            self.height = int(y_s.max()) + 1

    @property
    def n_events(self) -> int:
        return self._n_events

    @property
    def duration_s(self) -> float:
        with h5py.File(self.path, 'r') as f:
            t_end = float(np.array(f[self._schema['t']][-1]))
        return (t_end - self._t0) * self._t_scale

    def load_slice(self, start: int, end: int) -> EventBatch:
        """Load events by index range [start, end)."""
        end = min(end, self._n_events)
        with h5py.File(self.path, 'r') as f:
            t_raw = np.array(f[self._schema['t']][start:end], dtype=np.float64)
            x     = np.array(f[self._schema['x']][start:end])
            y     = np.array(f[self._schema['y']][start:end])
            p_raw = np.array(f[self._schema['p']][start:end])

        t = (t_raw - self._t0) * self._t_scale  # normalized seconds
        p = np.where(p_raw == 1, 1.0, -1.0)     # 0/1 → -1/+1

        return EventBatch(t=t, x=x, y=y, p=p,
                          width=self.width, height=self.height,
                          source=os.path.basename(self.path))

    def load_all(self, max_events: int = 5_000_000) -> EventBatch:
        n = min(max_events, self._n_events)
        return self.load_slice(0, n)

    def iter_windows(
        self,
        window_s: float = 0.01,
        max_events: int = 5_000_000,
        chunk_size: int = 500_000,
    ) -> Iterator[EventBatch]:
        """
        Yield EventBatch objects in fixed time windows.
        Loads data in chunks to stay memory-efficient on large files.
        """
        idx = 0
        total = min(max_events, self._n_events)
        buf_t = buf_x = buf_y = buf_p = None

        with h5py.File(self.path, 'r') as f:
            t_key = self._schema['t']
            while idx < total:
                # Fill buffer
                end = min(idx + chunk_size, total)
                t_chunk = (np.array(f[t_key][idx:end], dtype=np.float64) - self._t0) * self._t_scale
                x_chunk = np.array(f[self._schema['x']][idx:end])
                y_chunk = np.array(f[self._schema['y']][idx:end])
                p_chunk = np.where(np.array(f[self._schema['p']][idx:end]) == 1, 1.0, -1.0)

                idx = end

                # Emit time windows from chunk
                i = 0
                while i < len(t_chunk):
                    t_start = t_chunk[i]
                    t_end   = t_start + window_s
                    mask    = t_chunk[i:] < t_end
                    n       = int(mask.sum())
                    if n == 0:
                        break
                    yield EventBatch(
                        t=t_chunk[i:i+n], x=x_chunk[i:i+n],
                        y=y_chunk[i:i+n], p=p_chunk[i:i+n],
                        width=self.width, height=self.height,
                        source=os.path.basename(self.path),
                    )
                    i += n


def _load_gt_poses(gt_path: str) -> GroundTruthPoses:
    """Parse TUM-format groundtruth file and derive IMU angular velocity."""
    raw = np.loadtxt(gt_path, comments='#')
    t_raw = raw[:, 0]                    # seconds (high precision)
    t     = t_raw - t_raw[0]            # normalize to 0
    pos   = raw[:, 1:4]
    quat  = raw[:, 4:8]                  # (qx, qy, qz, qw)

    # Angular velocity: ω ≈ 2 · Im(q2 · conj(q1)) / Δt
    omegas = []
    ts_omega = []
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        if dt < 1e-9:
            continue
        qx1, qy1, qz1, qw1 = quat[i-1]
        qx2, qy2, qz2, qw2 = quat[i]
        # q2 * conj(q1): conj(q1) = (-qx1, -qy1, -qz1, qw1)
        rx = qw1*qx2 - (-qx1)*qw2 - (-qy1)*qz2 + (-qz1)*qy2  # wrong sign fix below
        # Correct Hamilton product: (a+bi+cj+dk)(e+fi+gj+hk)
        # q2 * conj(q1):
        # w = qw2*qw1 + qx2*qx1 + qy2*qy1 + qz2*qz1
        # x = qw2*(-qx1) + qx2*qw1 + qy2*(-qz1) - qz2*(-qy1)  ... simplified:
        # Using direct formula for relative rotation imaginary part:
        rx = qw1*qx2 - qw2*qx1 + qy1*qz2 - qz1*qy2
        ry = qw1*qy2 - qw2*qy1 + qz1*qx2 - qx1*qz2
        rz = qw1*qz2 - qw2*qz1 + qx1*qy2 - qy1*qx2
        omegas.append([2*rx/dt, 2*ry/dt, 2*rz/dt])
        ts_omega.append((t[i] + t[i-1]) / 2.0)

    return GroundTruthPoses(
        t=t, pos=pos, quat=quat,
        imu=IMUFromGT(
            t=np.array(ts_omega),
            omega=np.array(omegas),
        )
    )


def load_dataset(events_h5: str, gt_txt: Optional[str] = None):
    """
    Load an event dataset and optionally its ground truth poses.
    Returns (EventH5Dataset, GroundTruthPoses | None).
    """
    ds = EventH5Dataset(events_h5)
    gt = _load_gt_poses(gt_txt) if gt_txt and os.path.exists(gt_txt) else None
    return ds, gt
