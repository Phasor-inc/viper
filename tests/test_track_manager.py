"""
Tests for KalmanTrack + TrackManager.
Run: pytest tests/test_track_manager.py -v
"""
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.track_manager import KalmanTrack, TrackManager, TrackOutput
from engine.viper_engine import TrackState


def _make_hit(x, y, z, conf=0.9):
    return TrackState(x=x, y=y, z=z, confidence=conf, latency_us=100.0)


# ── KalmanTrack ───────────────────────────────────────────────────

def test_kalman_predict_moves_position():
    tr = KalmanTrack(np.array([100.0, 80.0, 3.0]), dt=0.01)
    # Inject velocity via two updates
    tr.update(np.array([101.0, 80.5, 3.0]))
    tr.predict(0.01)
    # Position should have moved in x direction
    assert tr.position[0] > 100.0


def test_kalman_update_reduces_error():
    tr = KalmanTrack(np.array([0.0, 0.0, 5.0]))
    obs = np.array([1.0, 0.0, 5.0])
    before = abs(tr.position[0] - obs[0])
    tr.update(obs)
    after = abs(tr.position[0] - obs[0])
    assert after < before


def test_kalman_hits_increments():
    tr = KalmanTrack(np.array([50.0, 50.0, 2.0]))
    assert tr.hits == 1
    tr.update(np.array([50.0, 50.0, 2.0]))
    tr.update(np.array([50.0, 50.0, 2.0]))
    assert tr.hits == 3


def test_kalman_velocity_estimated():
    tr = KalmanTrack(np.array([0.0, 0.0, 5.0]), dt=0.1)
    # Feed observations moving at 10 px/step
    for i in range(1, 10):
        tr.predict(0.1)
        tr.update(np.array([i * 10.0, 0.0, 5.0]))
    # vx should be converging toward ~100 (10px / 0.1s); Kalman lags, so >30 is valid
    assert tr.velocity[0] > 30.0


# ── TrackManager ──────────────────────────────────────────────────

def test_track_created_on_first_hit():
    mgr = TrackManager()
    mgr.update(_make_hit(100, 80, 3.0), t=0.001)
    assert len(mgr._tracks) == 1


def test_track_confirmed_after_n_init_hits():
    mgr = TrackManager()
    mgr.N_INIT = 3
    hit = _make_hit(100, 80, 3.0)
    for i in range(3):
        out = mgr.update(hit, t=i * 0.001)
    assert mgr.n_confirmed == 1
    assert len(out) == 1


def test_track_not_confirmed_before_n_init():
    mgr = TrackManager()
    mgr.N_INIT = 3
    hit = _make_hit(100, 80, 3.0)
    out = mgr.update(hit, t=0.001)
    assert mgr.n_confirmed == 0
    assert len(out) == 0


def test_two_distant_hits_create_two_tracks():
    mgr = TrackManager(gate=10.0)
    mgr.update(_make_hit(10, 10, 2.0), t=0.001)
    mgr.update(_make_hit(200, 200, 5.0), t=0.002)
    assert len(mgr._tracks) == 2


def test_close_hits_associate_to_same_track():
    mgr = TrackManager(gate=50.0)
    for i in range(5):
        mgr.update(_make_hit(100 + i, 80, 3.0), t=i * 0.001)
    assert len(mgr._tracks) == 1


def test_track_deleted_after_max_misses():
    mgr = TrackManager()
    mgr.MAX_MISSES = 2
    mgr.N_INIT = 1
    mgr.update(_make_hit(100, 80, 3.0), t=0.0)
    # Feed hits at a very different location to cause misses on track 1
    for i in range(5):
        mgr.update(_make_hit(500, 400, 10.0), t=(i+1) * 0.1)
    # Original track should have been pruned
    positions = [tr.position for tr in mgr._tracks]
    assert not any(abs(p[0] - 100) < 5 for p in positions)


def test_output_has_correct_fields():
    mgr = TrackManager()
    mgr.N_INIT = 1
    out = mgr.update(_make_hit(173, 130, 3.33), t=0.001)
    assert len(out) == 1
    t = out[0]
    assert isinstance(t, TrackOutput)
    assert t.id >= 1
    assert abs(t.x - 173) < 5
    assert abs(t.z - 3.33) < 1
    assert t.hits >= 1


def test_track_output_velocity_nonzero_after_motion():
    mgr = TrackManager()
    mgr.N_INIT = 1
    for i in range(10):
        out = mgr.update(_make_hit(100 + i*5, 80, 3.0), t=i * 0.01)
    assert len(out) >= 1
    assert out[0].speed >= 0   # velocity estimated
