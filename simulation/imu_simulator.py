"""
IMU (gyroscope) simulator.

Simulates angular velocity from a platform undergoing typical
defense platform vibration: slow drift + high-frequency mechanical noise.

Output format: (ωx, ωy, ωz) in rad/s at 1000 Hz.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class IMUReading:
    t: float
    omega: np.ndarray    # (3,) rad/s
    dt: float


class IMUSimulator:
    """
    Generates realistic IMU readings for a moving platform.

    Motion model:
        - Low-frequency sway: sinusoidal drift at 0.5-2 Hz (platform motion)
        - High-frequency vibration: Gaussian noise at 1000 Hz (engine/mechanical)
        - Optional: user-defined angular velocity trajectory
    """

    def __init__(
        self,
        rate_hz: float = 1000.0,
        drift_amplitude: float = 0.05,   # rad/s
        drift_freq: float = 1.0,          # Hz
        noise_std: float = 0.005,         # rad/s (gyroscope noise)
        seed: int = 7,
    ):
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz
        self.drift_amplitude = drift_amplitude
        self.drift_freq = drift_freq
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def reading_at(self, t: float) -> IMUReading:
        """Generate one IMU reading at time t."""
        # Platform slow drift (yaw + pitch oscillation)
        omega_y = self.drift_amplitude * np.sin(2 * np.pi * self.drift_freq * t)
        omega_x = self.drift_amplitude * 0.5 * np.cos(2 * np.pi * self.drift_freq * t * 1.3)
        omega_z = 0.0

        # Add gyroscope noise
        noise = self._rng.normal(0, self.noise_std, 3)
        omega = np.array([omega_x, omega_y, omega_z]) + noise

        return IMUReading(t=t, omega=omega, dt=self.dt)

    def stream(self, t_start: float, t_end: float) -> list[IMUReading]:
        """Generate all IMU readings from t_start to t_end."""
        times = np.arange(t_start, t_end, self.dt)
        return [self.reading_at(t) for t in times]
