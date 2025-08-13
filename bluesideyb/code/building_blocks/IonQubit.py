'''
Creates the quantum system of 171Yb+ ion. Defines the basis states. Creates a Quantum State in the quantum system to operate on
Potentially later the decay operations and the laser drive operations.  
'''

# Imports
import json
import scipy
import jax as jx    # could be used for jit
import numba as nb  # use for jit, maybe parallelization
import numpy as np  # use for computation
from pathlib import Path
from scipy.sparse import csr_matrix, csr_array, kron, identity  # faster operation and optimization
from scipy.sparse.linalg import expm_multiply
import matplotlib.pyplot as plt  # to showcase the frequency and amplitude - maybe just the pulse so one can see it
from typing import Union, Tuple, List, Any, Callable, Literal, Sequence

# TODO: make @ into np.einsum for faster computation. Think about using numba as well for jit compilation

class System:
    def __init__(self, dimension: int, path: Union[None, str, Path] = None):
        self.constants = self._load_constants(path=path)
        self.dim = dimension
        self._basis = [self.ket(i) for i in range(dimension)]
    
    def _load_constants(self, path: Union[None, str, Path]):
        if path is None:
            # assuming GitHub like structure and finding the constants.json file
            path = Path.cwd() / "constants.json"
        with open(path, 'r') as f:
            self.constants = json.load(f)

    def ket(self, i: int) -> csr_matrix:
        v = np.zeros((self.dim, 1), dtype=complex); v[i, 0] = 1.0
        return csr_matrix(v)
    
    def bra(self, i: int) -> csr_matrix:
        v = np.zeros((1, self.dim), dtype=complex); v[0, i] = 1.0
        return csr_matrix(v)
    
    def proj(self, i: int) -> csr_matrix:
        return self.ket(i) @ self.bra(i)
    
    def xop(self, i: int, j: int) -> csr_matrix:
        return self.ket(i) @ self.bra(j)

    @staticmethod
    def vec(rho: csr_matrix) -> np.ndarray:
        """
        Column-stacking vectorization of a (dim x dim) density matrix (CSR -> dense 1D).
        """
        return np.asarray(rho.todense()).reshape(-1, order="F")

    @staticmethod
    def unvec(v: np.ndarray, dim: int) -> csr_matrix:
        return csr_matrix(v.reshape((dim, dim), order="F"))

    @staticmethod
    def liouvillian(H: csr_matrix, collapses: Sequence[csr_matrix]) -> csr_matrix:
        # TODO: Needs to be generalized to work with time dependant collapses <=> noise models

        """
        L = -i(I⊗H - H^T⊗I) + sum_k (C_k ⊗ C_k^* - 0.5 I⊗(Ck†Ck)^T - 0.5 (Ck†Ck)⊗I)
        All inputs are CSR matrices. Output CSR (dim^2 x dim^2).
        """
        dim = H.shape[0]
        I = identity(dim, dtype=complex, format="csr")
        # coherent part
        L = -1j * (kron(I, H) - kron(H.T, I))
        # dissipators
        for C in collapses:
            CdC = (C.getH() @ C).tocsr()
            L += kron(C, C.conjugate())
            L += -0.5 * kron(I, CdC.T) - 0.5 * kron(CdC, I)
        return L.tocsr()


class IonQubit(System):
    """
    Minimal Raman-addressed 171Yb+ qubit with explicit |P> and |D> levels.
    Basis indices:
      0: |0> = |S1/2, F=0, mF=0>
      1: |1> = |S1/2, F=1, mF=0>
      2: |P> = |2P1/2> (effective)
      3: |D> = |2D3/2>
    """
    def __init__(self, initial_state: Union[None, str, int, csr_matrix] = None, path: Union[None, str, Path] = None):
        '''
        Allows for the following initial states
         - strings: {'0'|'1'|'+'|'-'|'i'|'-i'} 
         - int: {0 | 1}
         - csr_matrix: (4x4 matrix) that is built from the |0> and |1> states from the System class or in the same way created
        Add a path to your specific constants.json file - or if you want to use the default one, which is provided in the 
        git repo then leave it as is.
        '''
        super().__init__(dimension=4, path=path)
        self.i0, self.i1, self.iP, self.iD = 0, 1, 2, 3
        # Cache projectors & couplers for performance increase (at least thats the idea)
        self.P0 = self.proj(self.i0); self.P1 = self.proj(self.i1)  # |0><0| and |1><1| 
        self.PP = self.proj(self.iP); self.PD = self.proj(self.iD)  # |P><P| and |D><D|
        self.P_0 = self.xop(self.iP, self.i0)  # |P><0|
        self.P_1 = self.xop(self.iP, self.i1)  # |P><1|
        # Qubit sigma_z in {|0>,|1>} subspace
        self.sigma_z = (self.P0 - self.P1).tocsr()  # |0><0| - |1><1| == Pauli-Z

        # Create the initial state
        self.state = self.initialize_state(state=initial_state)

    def initialize_state(self, state: Union[None, str, int, csr_matrix]) -> csr_matrix:
        """
        Returns a CSR density matrix for the given initial state.
        Supports:
        - None -> |0><0| as default
        - int 0|1 -> |0><0| or |1><1|
        - str '0','1','+','-','i','-i' -> corresponding pure qubit state
        - csr_matrix -> used directly (checks dim=4)
        """
        # default
        if state is None:
            return self.proj(self.i0)

        # integer 0 or 1
        if isinstance(state, int):
            if state not in [0,1]:
                raise ValueError("Integer state must be 0 or 1")
            return self.proj([self.i0, self.i1][state])

        # string-based qubit states
        if isinstance(state, str):
            state_map = {
                '0': self.ket(self.i0),
                '1': self.ket(self.i1),
                '+': (self.ket(self.i0)+self.ket(self.i1))/np.sqrt(2),
                '-': (self.ket(self.i0)-self.ket(self.i1))/np.sqrt(2),
                'i': (self.ket(self.i0)+1j*self.ket(self.i1))/np.sqrt(2),
                '-i': (self.ket(self.i0)-1j*self.ket(self.i1))/np.sqrt(2)
            }
            if state not in state_map:
                raise ValueError(f"Unknown string state: {state}")
            psi = state_map[state]
            return (psi @ psi.getH()).tocsr()

        if isinstance(state, csr_matrix):
            if state.shape != (4,4):
                raise ValueError("CSR density matrix must be of shape 4x4")
            return state.copy().tocsr()  # make it non-mutative

        raise TypeError("state must be None, int, str, or csr_matrix. Please look at the documentation for more details.")

    # ---- Hamiltonians (CSR matrix) ----
    def H_intrinsic(self, Delta: float, delta: float, eps_D: float = 0.0) -> csr_matrix:
        """
        H0 = -Delta |P><P| + (delta/2) |1><1| + eps_D |D><D|
        Describes the IonQubit at rest. With the corresponding energies relative to |0> with energy E0=0eV
        """
        H = (-Delta)*self.PP + (delta/2.0)*self.P1 + eps_D*self.PD
        return H.tocsr()

    def H_raman(self, Omega1: complex, phi1: float,
                        Omega2: complex, phi2: float) -> csr_matrix:
        """
        H_Raman = (Ω1/2 e^{iφ1}|P><0| + Ω2/2 e^{iφ2}|P><1| + h.c.)
        Ω1, Ω2 may be real amplitudes; phases applied here.
        """
        H = 0.5*(Omega1*np.exp(1j*phi1)*self.P_0 + Omega2*np.exp(1j*phi2)*self.P_1)
        H = H + H.getH()
        return H.tocsr()

    def H_detect(self, Omega_det: float) -> csr_matrix:
        """
        Detection drive on |1> <-> |P|: (Ω/2)(|P><1| + h.c.)
        """
        Hd = 0.5*Omega_det*(self.P_1 + self.P_1.getH())
        return Hd.tocsr()

    # ---- Collapse operators as CSR matrices ----
    @staticmethod
    def _scale(C: csr_matrix, rate: float) -> csr_matrix:
        return np.sqrt(max(rate, 0.0)) * C

    def C_P_to_0(self, Gamma_P0: float) -> csr_matrix:
        return self._scale(self.xop(self.i0, self.iP), Gamma_P0)

    def C_P_to_1(self, Gamma_P1: float) -> csr_matrix:
        return self._scale(self.xop(self.i1, self.iP), Gamma_P1)

    def C_P_to_D(self, Gamma_PD: float) -> csr_matrix:
        return self._scale(self.xop(self.iD, self.iP), Gamma_PD)

    def C_D_to_0(self, W_D0: float) -> csr_matrix:
        return self._scale(self.xop(self.i0, self.iD), W_D0)

    def C_D_to_1(self, W_D1: float) -> csr_matrix:
        return self._scale(self.xop(self.i1, self.iD), W_D1)

    def C_dephase(self, gamma_phi: float) -> csr_matrix:
        return self._scale(self.sigma_z, gamma_phi)

    def C_flip_up(self, gamma_up: float) -> csr_matrix:
        return self._scale(self.xop(self.i1, self.i0), gamma_up)

    def C_flip_dn(self, gamma_dn: float) -> csr_matrix:
        return self._scale(self.xop(self.i0, self.i1), gamma_dn)

    # ---- Liouvillian / evolution ----
    def build_L(self, H: csr_matrix, collapses: Sequence[csr_matrix]) -> csr_matrix:
        return System.liouvillian(H, collapses)

    def step(self, rho: csr_matrix, L: csr_matrix, dt: float) -> csr_matrix:
        """
        One step via expm_multiply on the vectorized state:  vec(rho(t+dt)) = exp(L*dt) vec(rho(t))
        """
        v = System.vec(rho)
        v_next = expm_multiply((L*dt), v)
        return System.unvec(v_next, self.dim)
    
    # --- Expectation and projection operations ---
    def qubit_expectations(self) -> dict:
        """
        Return the expectation values in the standard qubit bases of the current state - without changing the state itself:
          |0>, |1>, |+>, |->, |i>, |-i>
        Only considers the {|0>,|1>} subspace of the IonQubit System.
        """
        ket0 = self.ket(self.i0)
        ket1 = self.ket(self.i1)
        ket_plus  = (ket0 + ket1)/np.sqrt(2)
        ket_minus = (ket0 - ket1)/np.sqrt(2)
        ket_i     = (ket0 + 1j*ket1)/np.sqrt(2)
        ket_minus_i = (ket0 - 1j*ket1)/np.sqrt(2)

        states = {
            '0': ket0, '1': ket1,
            '+': ket_plus, '-': ket_minus,
            'i': ket_i, '-i': ket_minus_i
        }
        # TODO: optimize this matrix mult. maybe with np.einsum / numba for the loop
        expectations = {}
        for name, psi in states.items():
            projector = (psi @ psi.getH()).tocsr()
            expectations[name] = np.real(np.trace(self.state @ projector))
        return expectations

    def expectation(self, O: csr_matrix) -> complex:
        return np.trace(self.state @ O)

    def projections(self) -> dict:
        """
        Return the expectation values for all four levels of the IonQubit.
        Keys to access the data: '0','1','P','D'
        """
        return {
            '0': self.expectation(self.P0).real,
            '1': self.expectation(self.P1).real,
            'P': self.expectation(self.PP).real,
            'D': self.expectation(self.PD).real
        }

    # --- Bloch-sphere representation (for the qubit subspace only!) maybe ---
    # --- adding plotting, and giving options like qutip,qiskit,matplotlib ---
    def bloch_vector(self) -> np.ndarray:
        """
        Returns the Bloch vector [x, y, z] for the qubit subspace {|0>,|1>}.
        Uses pauli_x as sigma_x and y,z respectively, then computes <sigma_x|y|z>
        """
        sigma_x = self.xop(self.i0, self.i1) + self.xop(self.i1, self.i0)
        sigma_y = -1j*self.xop(self.i0, self.i1) + 1j*self.xop(self.i1, self.i0)
        sigma_z = self.sigma_z
        x = self.expectation(sigma_x).real
        y = self.expectation(sigma_y).real
        z = self.expectation(sigma_z).real
        return np.array([x, y, z])

    def plot_vector(self) -> None:
        # To be implemented - maybe use qutip/qiskit/... for visualization
        # Might do my own version in matplotlib though
        pass

        
    



class LaserDrive:
    """
    Represents a Laser Drive in the quantum system 
    """
    def __init__(self, laser_frequency: Union[None, float] = None, amplitude: Union[None, float] = None, 
                 duration: Union[None, float] = None, pulse_shape: Literal['Gaussian', 'Square', 'Steady'] = "Gaussian",
                 phase: Union[None, float] = None) -> None:
        """
        Creates a Laser Drive Pulse or continuous wave. Can be used for operations like pumping and qubit rotations.

        Parameters:
        pulse_shape: Shapes the laser pulse under a Gaussian, Square envelope or just a "steady pulse" meaning infinite length.
        
        Return:
        None
        """
        
        self.laser_frequency = laser_frequency  # Frequency of the laser
        self.amplitude = amplitude              # Amplitude of the laser
        self.duration = duration                # Duration of the laser pulse (use np.inf for continuous)
        self.pulse_shape = pulse_shape          # Shape of the Laser pulse (Gaussian, Square, etc.)
        self.phase = None                       # Phase of the laser pulse
        self.pulse = None                       # The actual pulse data (time_stamp, value) <-- First set this to None needs to be generated 
    
# TODO: Get the basis states from the IonQubit and then create the operators so that one can apply rotations with the 
#       Hamiltonians between some states. Use sparse matrices for this. 