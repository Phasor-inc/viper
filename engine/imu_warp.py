"""
Feed-forward IMU motion compensation.

Gyroscope data (ωx, ωy, ωz) at 1000 Hz is integrated into an affine warp
matrix BEFORE events are processed. This pre-stabilizes event coordinates
so the SNN sees a motion-compensated view, not a smeared one.

This is the key innovation over post-hoc optical flow: zero buffering delay.
"""
import numpy as np


def skew(w: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix of angular velocity vector."""
    return np.array([
        [0.0,   -w[2],  w[1]],
        [w[2],   0.0,  -w[0]],
        [-w[1],  w[0],   0.0],
    ])


def gyro_to_warp(omega: np.ndarray, dt: float, focal_length: float = 200.0) -> np.ndarray:
    """
    Convert gyroscope reading to 3×3 affine warp matrix.

    Uses first-order approximation of rotation: R ≈ I - dt * [ω]_×
    The focal_length converts angular displacement to pixel displacement.

    Returns a 3×3 matrix that maps (x, y, 1) → (x', y', 1).
    """
    R = np.eye(3) - dt * skew(omega)
    # Project rotation into image plane via focal length
    K = np.array([[focal_length, 0, 0], [0, focal_length, 0], [0, 0, 1]], dtype=float)
    W = K @ R @ np.linalg.inv(K)
    return W


class IMUWarp:
    """
    Maintains a running warp matrix updated at IMU rate (1000 Hz).
    Events query the latest warp without waiting for a frame boundary.
    """

    def __init__(self, focal_length: float = 200.0, width: int = 346, height: int = 260):
        self.focal_length = focal_length
        self.width = width
        self.height = height
        self._W = np.eye(3)  # identity until first IMU reading

    def update(self, omega: np.ndarray, dt: float) -> None:
        """Integrate one gyroscope reading into the warp matrix."""
        self._W = gyro_to_warp(omega, dt, self.focal_length)

    def warp_event(self, x: float, y: float) -> tuple[float, float]:
        """
        Apply current warp to a single event coordinate.
        Returns stabilized (x', y') clipped to sensor bounds.
        """
        p = self._W @ np.array([x, y, 1.0])
        x_prime = np.clip(p[0] / p[2], 0, self.width - 1)
        y_prime = np.clip(p[1] / p[2], 0, self.height - 1)
        return float(x_prime), float(y_prime)

    def warp_batch(self, coords: np.ndarray) -> np.ndarray:
        """
        Warp N events at once. coords: (N, 2) array of (x, y).
        Returns (N, 2) stabilized coordinates.
        """
        ones = np.ones((len(coords), 1))
        homogeneous = np.hstack([coords, ones])          # (N, 3)
        warped = (self._W @ homogeneous.T).T             # (N, 3)
        xy = warped[:, :2] / warped[:, 2:3]
        xy[:, 0] = np.clip(xy[:, 0], 0, self.width - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, self.height - 1)
        return xy
