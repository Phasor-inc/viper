# VIPER

**Sub-millisecond neuromorphic thermal target tracking.**

> The first algorithm to combine feed-forward IMU motion compensation with spiking neural networks and event-based processing to achieve sub-millisecond thermal target tracking on neuromorphic hardware.

---

## Core Innovation

| Problem | Traditional approach | VIPER |
|---|---|---|
| Platform motion blur | Post-hoc optical flow (adds latency) | IMU feed-forward warp *before* processing |
| Thermal PSF blur | Frame-level deconvolution | Per-event async kernel stamp |
| Depth estimation | Texture block-matching | Spike timing coincidence (Δt) |
| Latency | 20-50ms (frame buffer) | **<1ms (event-driven)** |

Inspired by pit viper infrared sensing — biological neural architecture, not ML training.

---

## Pipeline

```
Thermal event (t, x, y, p)
         │
         ▼
[1] IMU Feed-Forward Warp          ← gyroscope at 1000 Hz, pre-stabilizes coords
         │  (x', y') = W @ (x, y)
         ▼
[2] PSF Deconvolution              ← 7×7 Gaussian kernel stamped per-event
         │  surface[y', x'] += p * kernel
         ▼
[3] LIF Neuron Update              ← membrane accumulates, fires at threshold
         │  V_mem = β·V_mem + I; fire if V > θ
         ▼
[4] Stereo Coincidence (Δt)        ← left + right spike within 1ms → depth
         │  Z = f·B / disparity
         ▼
TrackState(x, y, z, confidence, latency_µs)
```

---

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                    # 15/15 tests
python benchmark/latency_bench.py             # latency proof
```

---

## Project Structure

```
viper/
├── engine/
│   ├── imu_warp.py          # Feed-forward IMU stabilization
│   ├── psf_deconv.py        # Async PSF deconvolution (7×7 Gaussian)
│   ├── lif_neurons.py       # Leaky Integrate-and-Fire neuron grid
│   ├── stereo_depth.py      # Temporal coincidence depth estimation
│   └── viper_engine.py      # SNNViperEngine — full pipeline
├── simulation/
│   ├── thermal_scene.py     # Synthetic thermal scene + target motion
│   ├── event_camera.py      # DVS event stream from thermal frames
│   └── imu_simulator.py     # Gyroscope / IMU data simulator
├── loihi/
│   └── loihi_bridge.py      # Intel Loihi 2 deployment via Lava
├── benchmark/
│   └── latency_bench.py     # VIPER vs frame-based latency benchmark
└── tests/
    └── test_engine.py       # 15 unit tests
```

---

## Usage

```python
from engine import SNNViperEngine
import numpy as np

engine = SNNViperEngine(
    width=346, height=260,
    focal_length=200.0,
    baseline=0.10,          # 10cm stereo baseline
)

# Update IMU at 1000 Hz
engine.update_imu(omega=np.array([0.01, 0.02, 0.0]), dt=0.001)

# Process one event from left camera
track = engine.process_event(t=0.001, x=100, y=80, polarity=1.0, cam=0)
# Process matching event from right camera
track = engine.process_event(t=0.0011, x=94, y=80, polarity=1.0, cam=1)

if track:
    print(f"Target at ({track.x:.1f}, {track.y:.1f}, {track.z:.2f}m) "
          f"conf={track.confidence:.2f} latency={track.latency_us:.0f}µs")
```

---

## Loihi 2 Deployment

```python
from loihi import ViperLoihiNetwork

net = ViperLoihiNetwork(width=346, height=260, use_hardware=True)
net.deploy()
# Then use net.process_event() / net.update_imu() identically to SNNViperEngine
```

Requires `lava-nc`: https://github.com/lava-nc/lava

---

## Hardware Integration (Phase 3)

- **Prophesee Metavision**: `prophesee-openeb` SDK → `hardware/prophesee_adapter.py`
- **DVS346**: `libcaer` / DVSOT21 format → `hardware/dvs_adapter.py`
- **IMU**: serial / ROS2 topic → `hardware/imu_reader.py`

---

## IP Notes

Novel aspects for patent continuation:
1. Feed-forward (not post-hoc) IMU compensation applied at event level
2. Async per-event PSF deconvolution (no frame accumulation)
3. Spike timing disparity (Δt) for stereo depth without texture matching
4. Architecture maps natively to Loihi 2 process model
