"""
VIPER demo server.
Streams synthetic thermal events + track output over WebSocket.
Run: python demo/server.py
Open: http://localhost:8765
"""
import asyncio
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from engine import SNNViperEngine, TrackManager
from simulation import ThermalScene, EventCamera, IMUSimulator
from simulation.thermal_scene import TargetMotion

app = FastAPI()

HTML = open(os.path.join(os.path.dirname(__file__), "index.html")).read()


@app.get("/")
async def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()

    motion = TargetMotion(x0=20, y0=130, vx=250, vy=30, ax=0, ay=-10,
                          sigma=5.0, amplitude=5.0)
    scene  = ThermalScene(346, 260, fps=200.0, target=motion, noise_std=0.18, seed=3)
    cam_L  = EventCamera(346, 260, threshold=0.75, cam_id=0, baseline_px=0)
    cam_R  = EventCamera(346, 260, threshold=0.75, cam_id=1, baseline_px=30)
    imu    = IMUSimulator(rate_hz=1000.0)
    engine  = SNNViperEngine(width=346, height=260, focal_length=200.0, baseline=0.10)
    tracker = TrackManager(gate=40.0)

    dt = 1.0 / 200.0
    t = 0.0
    imu_t = 0.0
    imu_dt = 0.001
    latencies = []

    try:
        while True:
            # Advance IMU
            while imu_t <= t:
                r = imu.reading_at(imu_t)
                engine.update_imu(r.omega, r.dt)
                imu_t += imu_dt

            frame = scene.render(t)
            evL = cam_L.process_frame(frame, t)
            evR = cam_R.process_frame(frame, t)

            event_data = []
            track_data = None
            confirmed_tracks = []

            for ev in sorted(evL + evR, key=lambda e: e.t):
                event_data.append({
                    "x": ev.x, "y": ev.y,
                    "p": int(ev.polarity),
                    "c": ev.cam,
                })
                raw = engine.process_event(ev.t, ev.x, ev.y, ev.polarity, ev.cam)
                if raw is not None:
                    latencies.append(raw.latency_us)
                    confirmed_tracks = tracker.update(raw, t=ev.t)
                    if confirmed_tracks:
                        tr = confirmed_tracks[0]
                        track_data = {
                            "x": round(tr.x, 1),
                            "y": round(tr.y, 1),
                            "z": round(tr.z, 2),
                            "vx": round(tr.vx, 2),
                            "vy": round(tr.vy, 2),
                            "speed": round(tr.speed, 2),
                            "conf": round(tr.confidence, 3),
                            "track_id": tr.id,
                            "hits": tr.hits,
                            "lat_us": round(raw.latency_us, 1),
                            "mean_us": round(float(np.mean(latencies[-50:])), 1),
                        }

            gt_x, gt_y = scene.target_gt(t)

            msg = {
                "t": round(t, 4),
                "events": event_data[:300],
                "track": track_data,
                "n_confirmed": tracker.n_confirmed,
                "n_tentative": tracker.n_tentative,
                "gt": {"x": round(gt_x, 1), "y": round(gt_y, 1)},
                "stats": engine.stats,
            }
            await ws.send_text(json.dumps(msg))

            t += dt
            if t > 8.0:   # loop forever
                t = 0.0
                cam_L.reset(); cam_R.reset()
                latencies.clear()

            await asyncio.sleep(dt)   # real-time pacing

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="warning")
