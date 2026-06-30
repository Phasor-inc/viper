"""
Track manager with Kalman filter for VIPER depth hits.

Each stereo depth hit (x, y, z) from SNNViperEngine is fed here.
The manager maintains a pool of active tracks, associates new hits
to existing tracks via nearest-neighbour gating, and runs a 6-DOF
Kalman filter (position + velocity) on each track.

State vector: [x, y, z, vx, vy, vz]
Observation:  [x, y, z]

Outputs a list of confirmed TrackOutput objects every call — one per
live track — with smoothed position, velocity, and track ID.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import itertools


# ── Kalman filter (constant-velocity model) ───────────────────────

class KalmanTrack:
    """
    6-DOF constant-velocity Kalman filter for a single target.

    State:       x = [px, py, pz, vx, vy, vz]
    Transition:  F = I + dt * [[0,I],[0,0]]   (constant velocity)
    Observation: H = [I | 0]                  (observe position only)
    """
    _id_counter = itertools.count(1)

    def __init__(
        self,
        x0: np.ndarray,   # initial (px, py, pz)
        dt: float = 0.001,
        process_noise: float = 0.5,
        obs_noise: float = 0.1,
    ):
        self.id = next(KalmanTrack._id_counter)
        self.dt = dt

        # State
        self._x = np.array([x0[0], x0[1], x0[2], 0.0, 0.0, 0.0])

        # Covariance
        self._P = np.eye(6) * 1.0

        # Transition matrix
        self._F = np.eye(6)
        self._F[0, 3] = dt
        self._F[1, 4] = dt
        self._F[2, 5] = dt

        # Observation matrix
        self._H = np.zeros((3, 6))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = 1.0

        # Noise matrices
        q = process_noise
        self._Q = np.eye(6) * q * q
        self._R = np.eye(3) * obs_noise * obs_noise

        self.hits = 1
        self.misses = 0
        self.last_t: float = 0.0

    def predict(self, dt: Optional[float] = None) -> np.ndarray:
        """Predict state forward by dt (defaults to self.dt)."""
        if dt is not None and dt > 0:
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
        else:
            F = self._F
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q
        return self._x[:3]

    def update(self, z: np.ndarray) -> np.ndarray:
        """Correct with observation z = [px, py, pz]. Returns smoothed position."""
        H, R = self._H, self._R
        S = H @ self._P @ H.T + R              # innovation covariance
        K = self._P @ H.T @ np.linalg.inv(S)  # Kalman gain
        y = z - H @ self._x                    # innovation
        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ H) @ self._P
        self.hits += 1
        self.misses = 0
        return self._x[:3]

    @property
    def position(self) -> np.ndarray:
        return self._x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._x[3:].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self._x[3:]))


# ── Track output ──────────────────────────────────────────────────

@dataclass
class TrackOutput:
    id: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    speed: float
    confidence: float
    hits: int
    age_s: float


# ── Track manager ─────────────────────────────────────────────────

class TrackManager:
    """
    Associates SNNViperEngine depth hits to persistent tracks.

    Lifecycle:
        TENTATIVE  — track created, waiting for N_INIT confirmations
        CONFIRMED  — stable track, output to downstream
        DELETED    — missed MAX_MISSES predictions, removed

    Usage:
        mgr = TrackManager()
        # in your event loop:
        track_state = engine.process_event(...)
        if track_state:
            outputs = mgr.update(track_state, t=ev.t)
        # outputs is a list of TrackOutput (one per confirmed track)
    """

    N_INIT     = 3    # hits required before a track is confirmed
    MAX_MISSES = 5    # missed predictions before deletion
    GATE_M     = 2.0  # Mahalanobis gate distance (pixels/meters)

    def __init__(
        self,
        dt: float = 0.001,
        process_noise: float = 0.5,
        obs_noise: float = 0.1,
        gate: float = 50.0,   # pixel/meter distance gate for association
    ):
        self._dt = dt
        self._pn = process_noise
        self._on = obs_noise
        self._gate = gate
        self._tracks: list[KalmanTrack] = []
        self._status: dict[int, str] = {}   # track_id → 'tentative'|'confirmed'
        self._t_created: dict[int, float] = {}
        self._last_t: float = 0.0

    def update(
        self,
        hit,         # TrackState from SNNViperEngine (has .x, .y, .z, .confidence)
        t: float,
    ) -> list[TrackOutput]:
        """
        Feed one depth hit, return all confirmed tracks after association + filter.
        """
        z = np.array([hit.x, hit.y, hit.z])
        dt = max(1e-6, t - self._last_t)
        self._last_t = t

        # ── Predict all existing tracks ───────────────────────────
        for tr in self._tracks:
            tr.predict(dt)

        # ── Associate hit to nearest track within gate ────────────
        best_track = None
        best_dist  = self._gate

        for tr in self._tracks:
            dist = float(np.linalg.norm(z - tr.position))
            if dist < best_dist:
                best_dist  = dist
                best_track = tr

        if best_track is not None:
            best_track.update(z)
            best_track.last_t = t
            if (self._status[best_track.id] == 'tentative'
                    and best_track.hits >= self.N_INIT):
                self._status[best_track.id] = 'confirmed'
        else:
            # New track
            tr = KalmanTrack(z, self._dt, self._pn, self._on)
            tr.last_t = t
            self._tracks.append(tr)
            self._status[tr.id] = 'tentative'
            self._t_created[tr.id] = t
            # If N_INIT=1, confirm immediately on creation
            if tr.hits >= self.N_INIT:
                self._status[tr.id] = 'confirmed'

        # ── Mark tracks that weren't updated ─────────────────────
        updated_ids = {best_track.id} if best_track else set()
        for tr in self._tracks:
            if tr.id not in updated_ids:
                tr.misses += 1

        # ── Remove dead tracks ────────────────────────────────────
        self._tracks = [
            tr for tr in self._tracks
            if tr.misses < self.MAX_MISSES
        ]
        for tid in list(self._status):
            if not any(tr.id == tid for tr in self._tracks):
                del self._status[tid]
                self._t_created.pop(tid, None)

        # ── Emit confirmed tracks ─────────────────────────────────
        outputs = []
        for tr in self._tracks:
            if self._status.get(tr.id) != 'confirmed':
                continue
            p = tr.position
            v = tr.velocity
            age = t - self._t_created.get(tr.id, t)
            outputs.append(TrackOutput(
                id=tr.id,
                x=float(p[0]), y=float(p[1]), z=float(p[2]),
                vx=float(v[0]), vy=float(v[1]), vz=float(v[2]),
                speed=tr.speed,
                confidence=float(hit.confidence),
                hits=tr.hits,
                age_s=age,
            ))
        return outputs

    @property
    def n_confirmed(self) -> int:
        return sum(1 for s in self._status.values() if s == 'confirmed')

    @property
    def n_tentative(self) -> int:
        return sum(1 for s in self._status.values() if s == 'tentative')
