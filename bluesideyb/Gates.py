from Utils.Scripts import matrix_td_factory
from scipy.sparse import csr_matrix
from typing import Literal
import numpy as np


class BaseNoise:
    """
    Basis of Noise models; this should act as a foundation for new Noise classes build ontop of it like WhiteNoise class.
    """
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def apply_amp(self, amp: float, t: float) -> float:
        return amp

    def apply_phase(self, phase: float, t: float) -> float:
        return phase

    def wrap_amplitude(self, envelope_fn):
        def wrapped(t: float):
            return self.apply_amp(envelope_fn(t), t)
        return wrapped

    def wrap_phase(self, phase_fn):
        def wrapped(t: float):
            return self.apply_phase(phase_fn(t), t)
        return wrapped

    def wrap_complex_envelope(self, amp_fn, phase_fn):
        def wrapped(t: float):
            a = self.apply_amp(amp_fn(t), t)
            p = self.apply_phase(phase_fn(t), t)
            return a, p
        return wrapped


class WhiteNoise(BaseNoise):
    """
    Normal distributed noise, builds ontop of the Base Noise class.
    """
    def __init__(self, amp_sigma: float = 0.0, phase_sigma: float = 0.0, seed: int | None = None):
        super().__init__(seed=seed)
        self.amp_sigma = float(amp_sigma)
        self.phase_sigma = float(phase_sigma)

    def apply_amp(self, amp: float, t: float) -> float:
        return amp + self.rng.normal(0.0, self.amp_sigma)

    def apply_phase(self, phase: float, t: float) -> float:
        return phase + self.rng.normal(0.0, self.phase_sigma)


class Gate:
    def __init__(self, H: csr_matrix, duration: float,
                 envelope: Literal["rect", "gauss"],
                 phase_fn=None,
                 turn_on: float = 0.0,
                 noise: BaseNoise | None = None,
                 amp: float = 1,
                 sigma: float = 0.5):
        """
        This class should be used to set up the gates for the simulation. Here you can apply noise onto the gates as well.
        Converts the matrices to time dependent matrices (so you can controll when they are turned on and when not)
        Parameters:
        H: Matrix / Operation
        duration: for how long it is being run
        envelope: (acts on amplitude) rect if you want it to be perfectly switched on and off or gaussian so it is being turned on/off in a gaussian shape - more realistic
        phase_fn: (acts on phase) should have similar action as envelope but on phase
        turn_on: time from 0.0 when it is supposed to be turned on, somewhat like envelope but ensures 0 when turned off
        noise: Noisemodel you want to apply
        amp: amplitude 1, but to play around I made it a parameter, used for envelopes (so they fully turn on, can be decreased)
        sigma: 0.5  -  parameter for gaussian envelope


        Returns:
        Scheduled Gate
        """
        self.H = H.tocsr() if H is not isinstance(H, csr_matrix) else H
        self.duration = float(duration)
        self.envelope = envelope
        self.phase_fn = phase_fn
        self.turn_on = float(turn_on)
        self.noise = noise
        self.amp = amp
        self.sigma = sigma

    def as_td(self):
        """
        Convert to time dependent Matrices, so turn on/ off can be easily applied as well as noise added.
        """
        amp_env = None
        match self.envelope:
            case "rect" :
                from Utils.Scripts import rect_envelope
                amp_env = rect_envelope(duration=self.duration, t_start=self.turn_on, amp=self.amp)
            case "gauss" :
                from Utils.Scripts import gaussian_envelope
                amp_env = gaussian_envelope(t0=self.turn_on, amp=self.amp, sigma=self.sigma)
            case _:
                raise ValueError("envelope must be rect or gauss")

        if self.noise is not None:
            amp_env = self.noise.wrap_amplitude(amp_env)
            if self.phase_fn is not None:
                phase_fn = self.noise.wrap_phase(self.phase_fn)
            else:
                phase_fn = None
        else:
            phase_fn = self.phase_fn

        if phase_fn is None:
            return matrix_td_factory(self.H, amp_env, turn_on=self.turn_on)

        H = self.H
        Hdag = H.getH()

        def H_of_t(t: float):
            a = amp_env(t)
            ph = phase_fn(t)
            return 0.5 * a * (np.exp(1j*ph) * H + np.exp(-1j*ph) * Hdag)

        def scheduled(t: float):
            if (t < self.turn_on) or (t > self.turn_on + self.duration):
                return csr_matrix(H.shape, dtype=np.complex128)
            return csr_matrix(H_of_t(t))

        return scheduled
