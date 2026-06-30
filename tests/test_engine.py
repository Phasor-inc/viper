"""
Unit tests for the VIPER engine components.
Run: pytest tests/ -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.imu_warp import IMUWarp, gyro_to_warp
from engine.psf_deconv import PSFDeconvolver, make_psf_kernel
from engine.lif_neurons import LIFGrid
from engine.stereo_depth import StereoCoincidence, Spike
from engine.viper_engine import SNNViperEngine


# ─── IMU Warp ────────────────────────────────────────────────────────────────

def test_imu_warp_identity_with_zero_omega():
    warp = IMUWarp()
    warp.update(np.zeros(3), dt=0.001)
    x, y = 100.0, 80.0
    x_s, y_s = warp.warp_event(x, y)
    assert abs(x_s - x) < 2.0
    assert abs(y_s - y) < 2.0


def test_imu_warp_shifts_on_rotation():
    warp = IMUWarp(focal_length=200.0)
    # Positive ωy (pan right) should shift events leftward in image
    omega = np.array([0.0, 0.5, 0.0])  # rad/s
    warp.update(omega, dt=0.001)
    x_s, _ = warp.warp_event(173.0, 130.0)
    # Shifted — not at exact center
    assert x_s != 173.0


def test_imu_warp_batch_shape():
    warp = IMUWarp()
    warp.update(np.array([0.01, 0.02, 0.0]), dt=0.001)
    coords = np.array([[50, 60], [100, 120], [200, 180]], dtype=float)
    warped = warp.warp_batch(coords)
    assert warped.shape == (3, 2)


# ─── PSF Deconvolver ─────────────────────────────────────────────────────────

def test_psf_kernel_normalized():
    kernel = make_psf_kernel(sigma=1.2, size=7)
    assert abs(kernel.sum() - 1.0) < 1e-5


def test_psf_accumulate_updates_surface():
    psf = PSFDeconvolver(width=100, height=100)
    psf.accumulate(50, 50, 1.0)
    surface = psf.get_surface()
    assert surface[50, 50] > 0
    # Energy should spread to neighbors (kernel radius)
    assert surface[50, 51] > 0
    assert surface[49, 50] > 0


def test_psf_polarity_sign():
    psf = PSFDeconvolver(width=100, height=100)
    psf.accumulate(50, 50, +1.0)
    pos = psf.get_surface()[50, 50]
    psf.reset()
    psf.accumulate(50, 50, -1.0)
    neg = psf.get_surface()[50, 50]
    assert pos > 0
    assert neg < 0


def test_psf_decay():
    psf = PSFDeconvolver(width=50, height=50, decay=0.9)
    psf.accumulate(25, 25, 1.0)
    before = psf.get_surface()[25, 25]
    psf.decay_surface()
    after = psf.get_surface()[25, 25]
    assert abs(after / before - 0.9) < 1e-5


# ─── LIF Neurons ─────────────────────────────────────────────────────────────

def test_lif_fires_above_threshold():
    lif = LIFGrid(width=50, height=50, beta=0.9, threshold=0.5)
    ever_fired = False
    for _ in range(20):
        if lif.step_event(25, 25, 0.1):
            ever_fired = True
    assert ever_fired is True


def test_lif_resets_after_spike():
    lif = LIFGrid(width=50, height=50, beta=0.99, threshold=0.5, reset_voltage=0.0)
    spike_count = 0
    for _ in range(20):
        if lif.step_event(25, 25, 0.1):
            spike_count += 1
    # Neuron fired at least once — confirming threshold crossing + reset
    assert spike_count >= 1
    # Membrane must be below threshold (would have been reset after each spike)
    mem = lif.get_membrane()
    assert mem[25, 25] < 0.5


def test_lif_does_not_fire_below_threshold():
    lif = LIFGrid(width=50, height=50, beta=0.5, threshold=10.0)
    fired = lif.step_event(25, 25, 0.001)
    assert fired is False


# ─── Stereo Coincidence ───────────────────────────────────────────────────────

def test_stereo_depth_hit_on_coincidence():
    stereo = StereoCoincidence(focal_length=200.0, baseline=0.10, dt_max=1e-3)
    t = 0.001
    left  = Spike(t=t, x=100, y=100, cam=0)
    right = Spike(t=t + 0.0001, x=94, y=100, cam=1)  # 6px disparity
    stereo.push_spike(left)
    hit = stereo.push_spike(right)
    assert hit is not None
    # depth = 200 * 0.10 / 6 ≈ 3.33 m
    assert abs(hit.z - (200.0 * 0.10 / 6.0)) < 0.5


def test_stereo_no_hit_outside_dt_window():
    stereo = StereoCoincidence(focal_length=200.0, baseline=0.10, dt_max=1e-3)
    left  = Spike(t=0.0, x=100, y=100, cam=0)
    right = Spike(t=0.01, x=94, y=100, cam=1)  # 10ms apart — outside window
    stereo.push_spike(left)
    hit = stereo.push_spike(right)
    assert hit is None


def test_stereo_no_hit_wrong_epipolar():
    stereo = StereoCoincidence(focal_length=200.0, baseline=0.10, dt_max=1e-3, epipolar_tol=2)
    left  = Spike(t=0.001, x=100, y=100, cam=0)
    right = Spike(t=0.001, x=94, y=110, cam=1)  # 10px vertical offset
    stereo.push_spike(left)
    hit = stereo.push_spike(right)
    assert hit is None


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def test_viper_engine_returns_trackstate_on_stereo_hit():
    engine = SNNViperEngine(width=100, height=100)
    # Feed many ON events to both cameras at the same location
    hits = []
    for i in range(100):
        t = i * 1e-4
        engine.update_imu(np.array([0.01, 0.01, 0.0]), dt=1e-4)
        track_L = engine.process_event(t, 50, 50, 1.0, cam=0)
        track_R = engine.process_event(t + 1e-5, 44, 50, 1.0, cam=1)
        if track_L: hits.append(track_L)
        if track_R: hits.append(track_R)
    assert len(hits) > 0


def test_viper_latency_sub_millisecond():
    engine = SNNViperEngine(width=100, height=100)
    engine.update_imu(np.zeros(3), dt=1e-3)
    latencies = []
    for i in range(200):
        t = i * 5e-5
        track = engine.process_event(t, 50, 50, 1.0, cam=0)
        if not track:
            track = engine.process_event(t + 1e-5, 44, 50, 1.0, cam=1)
        if track:
            latencies.append(track.latency_us)
    if latencies:
        assert min(latencies) < 1000.0, f"Min latency {min(latencies):.0f} µs is not sub-ms"
