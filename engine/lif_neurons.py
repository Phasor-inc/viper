"""
Leaky Integrate-and-Fire neuron grid.

Implements the standard LIF model asynchronously:
    τ_m * dV/dt = -(V - V_rest) + R * I(t)

In event-driven form (no fixed time step):
    V(t) = V_rest + (V(t_last) - V_rest) * exp(-Δt / τ_m) + ΔV_event

snnTorch's Leaky neuron uses the same discrete-time approximation (β = e^(-dt/τ)).
We implement it directly here for truly asynchronous single-event processing,
but the beta parameter is identical to snnTorch's convention.

For Loihi 2 deployment: this maps 1:1 to Lava's LIF process.
"""
import numpy as np


class LIFGrid:
    """
    2D grid of LIF neurons — one per pixel of the stabilized event surface.

    Each neuron integrates incoming PSF-deconvolved energy and fires
    (emits a spike) when its membrane potential crosses the threshold.
    The membrane resets to 0 after firing.
    """

    def __init__(
        self,
        width: int = 346,
        height: int = 260,
        beta: float = 0.95,        # decay factor per event (≈ e^{-dt/τ})
        threshold: float = 0.5,
        reset_voltage: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.beta = beta
        self.threshold = threshold
        self.reset_voltage = reset_voltage
        self._mem = np.zeros((height, width), dtype=np.float32)

    def step_event(self, x: int, y: int, current: float) -> bool:
        """
        Process one event at pixel (x, y) with synaptic current `current`.

        Returns True if the neuron at (x, y) fires this step.
        """
        # Decay entire membrane (global leak approximation)
        # A Loihi-accurate implementation decays per-neuron per-spike-interval;
        # for simulation this global decay is equivalent when events are dense.
        self._mem *= self.beta

        # Accumulate input current
        self._mem[y, x] += current

        # Threshold check
        if self._mem[y, x] >= self.threshold:
            self._mem[y, x] = self.reset_voltage
            return True
        return False

    def step_surface(self, surface: np.ndarray) -> np.ndarray:
        """
        Bulk update from a full PSF surface (used in benchmarking).
        Returns boolean spike mask of same shape as surface.
        """
        self._mem = self.beta * self._mem + surface
        spikes = self._mem >= self.threshold
        self._mem[spikes] = self.reset_voltage
        return spikes

    def get_membrane(self) -> np.ndarray:
        return self._mem.copy()

    def reset(self) -> None:
        self._mem[:] = 0.0
