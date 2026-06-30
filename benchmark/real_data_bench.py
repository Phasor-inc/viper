"""
VIPER Real-Data Benchmark — Town03 event camera dataset.

Runs the full VIPER pipeline on real DVS event streams (not synthetic).
Uses GT camera poses to derive real IMU angular velocity.

Runs in monocular mode (stages 1–3 only: IMU warp + PSF deconv + LIF).
Stereo depth (stage 4) requires a second camera — see hardware/ adapters.

Extracts and prints:
  - Per-event latency through all active stages (µs)
  - LIF spike rate on real vs synthetic data
  - Event throughput (kEvents/s)
  - Stage-by-stage timing breakdown
  - Comparison table: real vs synthetic numbers
"""
import sys, os, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.imu_warp import IMUWarp
from engine.psf_deconv import PSFDeconvolver
from engine.lif_neurons import LIFGrid
from data import load_dataset

# ── Paths ─────────────────────────────────────────────────────────
BASE = "/Users/beffiong/Downloads/DL_Dataset_Fall_2024/Town03/rgb/rgb_aligned_town03_day"
EVENTS_H5 = os.path.join(BASE, "events_corrected.h5")
GT_TXT    = os.path.join(BASE, "stamped_groundtruth.txt")

# ── Config ─────────────────────────────────────────────────────────
MAX_EVENTS  = 2_000_000   # ~5s of real data
WINDOW_S    = 0.005        # 5ms processing windows (matches real-time at 200 Hz)
LIF_BETA      = 0.95
LIF_THRESHOLD = 0.5
PSF_SIGMA     = 1.2

# Reference numbers from synthetic Phase 1 benchmark
SYNTHETIC_MEAN_US   = 187.9
SYNTHETIC_SPIKE_RATE = 0.108


def interpolate_imu(gt_imu, t_query: float) -> np.ndarray:
    """Linear interpolation of IMU reading at arbitrary time."""
    if t_query <= gt_imu.t[0]:
        return gt_imu.omega[0]
    if t_query >= gt_imu.t[-1]:
        return gt_imu.omega[-1]
    idx = np.searchsorted(gt_imu.t, t_query) - 1
    idx = max(0, min(idx, len(gt_imu.t) - 2))
    alpha = (t_query - gt_imu.t[idx]) / (gt_imu.t[idx+1] - gt_imu.t[idx] + 1e-12)
    return gt_imu.omega[idx] * (1 - alpha) + gt_imu.omega[idx+1] * alpha


def run_real_data_bench():
    print("\n" + "=" * 64)
    print("  VIPER Real-Data Benchmark — Town03 Event Camera Dataset")
    print("=" * 64)

    # ── Load data ─────────────────────────────────────────────────
    print(f"\n[1/4] Loading dataset...")
    if not os.path.exists(EVENTS_H5):
        print(f"  ERROR: {EVENTS_H5} not found.")
        print("  Set EVENTS_H5 to your event camera H5 file path.")
        return None

    ds, gt = load_dataset(EVENTS_H5, GT_TXT)
    print(f"  Events    : {ds.n_events:,}  (loading {MAX_EVENTS:,})")
    print(f"  Duration  : {ds.duration_s:.2f}s")
    print(f"  Resolution: {ds.width}×{ds.height}")
    if gt:
        print(f"  GT poses  : {len(gt.t)} samples  (ω max={np.abs(gt.imu.omega).max():.4f} rad/s)")

    # ── Initialize engine (monocular: stages 1–3 only) ────────────
    print(f"\n[2/4] Initializing VIPER engine ({ds.width}×{ds.height})...")
    warp = IMUWarp(focal_length=400.0, width=ds.width, height=ds.height)
    psf  = PSFDeconvolver(ds.width, ds.height, PSF_SIGMA)
    lif  = LIFGrid(ds.width, ds.height, LIF_BETA, LIF_THRESHOLD)

    # ── Process events ────────────────────────────────────────────
    print(f"\n[3/4] Processing {MAX_EVENTS:,} real events...")

    latencies_us    = []
    stage_t         = {"warp": [], "psf": [], "lif": []}
    spike_xs, spike_ys = [], []
    n_events = 0
    n_spikes = 0
    t_wall_0 = time.perf_counter()
    last_imu_t = -1.0
    imu_dt = 0.001   # 1ms IMU update interval

    batch = ds.load_all(MAX_EVENTS)
    n_total = len(batch.t)

    for i in range(n_total):
        ev_t = float(batch.t[i])
        x    = int(batch.x[i])
        y    = int(batch.y[i])
        pol  = float(batch.p[i])

        # Update IMU at ~1000 Hz from GT poses
        if gt and (ev_t - last_imu_t) >= imu_dt:
            omega = interpolate_imu(gt.imu, ev_t)
            warp.update(omega, imu_dt)
            last_imu_t = ev_t

        t0 = time.perf_counter()

        # Stage 1: IMU warp
        t1 = time.perf_counter()
        xs, ys = warp.warp_event(x, y)
        xi, yi = int(round(xs)), int(round(ys))
        stage_t["warp"].append((time.perf_counter() - t1) * 1e6)

        # Stage 2: PSF deconvolution
        t2 = time.perf_counter()
        psf.accumulate(xi, yi, pol)
        current = psf.get_surface()[yi, xi]
        stage_t["psf"].append((time.perf_counter() - t2) * 1e6)

        # Stage 3: LIF update
        t3 = time.perf_counter()
        fired = lif.step_event(xi, yi, current)
        stage_t["lif"].append((time.perf_counter() - t3) * 1e6)

        total_us = (time.perf_counter() - t0) * 1e6
        latencies_us.append(total_us)
        n_events += 1

        if fired:
            n_spikes += 1
            spike_xs.append(xi)
            spike_ys.append(yi)

    t_wall = time.perf_counter() - t_wall_0

    # ── Results ───────────────────────────────────────────────────
    lats = np.array(latencies_us)
    for k in stage_t:
        stage_t[k] = np.array(stage_t[k])

    spike_rate = n_spikes / max(1, n_events)
    throughput = n_events / t_wall / 1000   # kEv/s

    print(f"\n[4/4] Results\n")

    print(f"{'─'*64}")
    print(f"  {'Metric':<38} {'Value':>12}")
    print(f"{'─'*64}")

    # Latency
    print(f"  {'LATENCY (stages 1–3, monocular)':<38}")
    print(f"  {'  Mean':<38} {np.mean(lats):>10.2f} µs")
    print(f"  {'  Median (p50)':<38} {np.percentile(lats,50):>10.2f} µs")
    print(f"  {'  p95':<38} {np.percentile(lats,95):>10.2f} µs")
    print(f"  {'  p99':<38} {np.percentile(lats,99):>10.2f} µs")
    print(f"  {'  Min':<38} {np.min(lats):>10.2f} µs")
    print(f"  {'  Sub-millisecond rate':<38} {np.mean(lats<1000)*100:>9.1f}%")

    # Stage breakdown
    print(f"\n  {'STAGE BREAKDOWN':<38}")
    for stage, label in [("warp","  Stage 1 — IMU warp"),
                          ("psf", "  Stage 2 — PSF deconv"),
                          ("lif", "  Stage 3 — LIF neurons")]:
        arr = stage_t[stage]
        print(f"  {label:<38} {np.mean(arr):>8.3f} µs  ±{np.std(arr):.3f}")

    # Throughput
    print(f"\n  {'THROUGHPUT & RATES':<38}")
    print(f"  {'  Events processed':<38} {n_events:>12,}")
    print(f"  {'  LIF spikes fired':<38} {n_spikes:>12,}")
    print(f"  {'  Spike rate':<38} {spike_rate*100:>11.2f}%")
    print(f"  {'  Event throughput':<38} {throughput:>9.1f} kEv/s")
    print(f"  {'  Wall time':<38} {t_wall:>11.2f}s")

    # vs synthetic
    print(f"\n  {'VS SYNTHETIC BENCHMARK':<38}")
    print(f"  {'  Latency (mean)':<38} {'real':>6} {np.mean(lats):>7.1f} µs  |  synth {SYNTHETIC_MEAN_US:.1f} µs")
    print(f"  {'  Spike rate':<38} {'real':>6} {spike_rate*100:>7.2f}%   |  synth {SYNTHETIC_SPIKE_RATE*100:.2f}%")
    print(f"{'─'*64}")

    return {
        "n_events": n_events,
        "n_spikes": n_spikes,
        "spike_rate": spike_rate,
        "latencies_us": lats,
        "stage_timings": stage_t,
        "throughput_kev_s": throughput,
        "spike_xs": np.array(spike_xs),
        "spike_ys": np.array(spike_ys),
        "width": ds.width,
        "height": ds.height,
    }


def plot_results(result: dict, save_path: str = "viper_real_data_results.png"):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plt.rcParams.update({
        "figure.facecolor": "#0a0a0f", "axes.facecolor": "#0f0f1a",
        "axes.edgecolor": "#2a2a3e",   "axes.labelcolor": "#aaa",
        "xtick.color": "#666",          "ytick.color": "#666",
        "text.color": "#ddd",           "grid.color": "#1a1a2e",
        "font.family": "monospace",
    })

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#0a0a0f")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ORANGE = "#ff6b35"
    BLUE   = "#4fc3f7"
    GREEN  = "#3ddc84"

    # 1: Latency histogram
    ax1 = fig.add_subplot(gs[0, :2])
    lats = result["latencies_us"]
    clip = np.percentile(lats, 99.5)
    bins = np.linspace(0, min(clip, 1200), 80)
    ax1.hist(lats[lats < clip] / 1000, bins=bins / 1000,
             color=ORANGE, alpha=0.85, edgecolor="none")
    ax1.axvline(np.mean(lats)/1000,   color="white", lw=1.5, ls="--",
                label=f"Mean {np.mean(lats):.1f} µs")
    ax1.axvline(np.median(lats)/1000, color=BLUE,    lw=1.5, ls=":",
                label=f"p50  {np.median(lats):.1f} µs")
    ax1.axvline(1.0, color=GREEN, lw=1, ls="-", alpha=0.5, label="1 ms")
    ax1.set_xlabel("Latency (ms)"); ax1.set_ylabel("Count")
    ax1.set_title("Real Event Data — Per-Event Pipeline Latency", color=ORANGE, fontsize=11)
    ax1.legend(framealpha=0.2, fontsize=9)
    ax1.grid(True, axis="y", alpha=0.3)
    sub_ms = np.mean(lats < 1000) * 100
    ax1.text(0.02, 0.88, f"{sub_ms:.1f}% sub-ms", transform=ax1.transAxes, color=GREEN, fontsize=10)

    # 2: Stage breakdown
    ax2 = fig.add_subplot(gs[0, 2])
    stages = ["warp", "psf", "lif"]
    labels = ["IMU\nWarp", "PSF\nDeconv", "LIF\nNeurons"]
    means  = [np.mean(result["stage_timings"][s]) for s in stages]
    stds   = [np.std(result["stage_timings"][s])  for s in stages]
    colors = [ORANGE, "#ffa726", "#ffeb3b"]
    ax2.bar(labels, means, yerr=stds, color=colors, alpha=0.85,
            edgecolor="none", capsize=5,
            error_kw={"ecolor": "#888", "elinewidth": 1})
    ax2.set_ylabel("Time (µs)")
    ax2.set_title("Stage Timing (Real Data)", color=ORANGE)
    ax2.grid(True, axis="y", alpha=0.3)

    # 3: Spike heatmap
    ax3 = fig.add_subplot(gs[1, :2])
    W, H = result["width"], result["height"]
    spike_map = np.zeros((H, W), dtype=np.float32)
    for sx, sy in zip(result["spike_xs"], result["spike_ys"]):
        if 0 <= sy < H and 0 <= sx < W:
            spike_map[sy, sx] += 1
    if spike_map.max() > 0:
        spike_map = np.log1p(spike_map)
    ax3.imshow(spike_map, cmap="inferno", aspect="auto")
    ax3.set_title("LIF Spike Density Map (Real Events)", color=ORANGE)
    ax3.set_xlabel("x (pixels)"); ax3.set_ylabel("y (pixels)")

    # 4: Real vs synthetic comparison
    ax4 = fig.add_subplot(gs[1, 2])
    metrics = ["Mean\nlatency (µs)", "Spike\nrate (%)"]
    real_vals  = [np.mean(lats), result["spike_rate"] * 100]
    synth_vals = [SYNTHETIC_MEAN_US, SYNTHETIC_SPIKE_RATE * 100]
    x = np.arange(len(metrics))
    w = 0.35
    ax4.bar(x - w/2, real_vals,  w, label="Real data",  color=ORANGE, alpha=0.85)
    ax4.bar(x + w/2, synth_vals, w, label="Synthetic",  color=BLUE,   alpha=0.85)
    ax4.set_xticks(x); ax4.set_xticklabels(metrics, fontsize=8)
    ax4.set_title("Real vs Synthetic", color=ORANGE)
    ax4.legend(framealpha=0.2, fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    plt.suptitle("VIPER — Real Event Camera Validation (Town03 Dataset)",
                 fontsize=13, color="white", y=0.98, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print(f"\n✓ Saved {save_path}")


if __name__ == "__main__":
    result = run_real_data_bench()
    if result:
        plot_results(result)
