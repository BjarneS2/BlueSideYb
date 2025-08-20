import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from typing import Sequence, Tuple, Union, Optional


class YbQubit:
    """
    The essence of our simulation. Here I created the Qubit and frame for the simulation and a bunch of helper functions.
    """
    def __init__(self, initial_state: Union[None, str, int, csr_matrix] = None,
                 dimension: int = 4, path: Union[None, str, Path] = None):
        """
        We cache some operators to reduce repitition in the code. In addition we load the physical constants from our CSV file.
        If None -> assume the same structure as in the git repo.
        After that we initialize the state. 
        """
        self.constants = self._load_constants(path=path)
        self.dim = dimension
        self._basis = [self.ket(i) for i in range(dimension)]

        self.i0, self.i1, self.iP, self.iD = 0, 1, 2, 3
        self.P0 = self.proj(self.i0); self.P1 = self.proj(self.i1)  # |0><0| and |1><1|
        self.PP = self.proj(self.iP); self.PD = self.proj(self.iD)  # |P><P| and |D><D|
        self.P_0 = self.xop(self.iP, self.i0)                       # |P><0|
        self.P_1 = self.xop(self.iP, self.i1)                       # |P><1|
        self._0_P = self.xop(self.i0, self.iP)                      # |0><P|
        self._1_P = self.xop(self.i1, self.iP)                      # |1><P|
        self._D_P = self.xop(self.iD, self.iP)                      # |D><P|
        self.P01 = self.xop(self.i0, self.i1)                       # |0><1|
        self.P10 = self.xop(self.i1, self.i0)                       # |1><0|
        self.sigma_x, self.sigma_y, self.sigma_z = self.paulis()    # Pauli matrices
        self.state = None
        self.initialize_state(state=initial_state)
    
    def _load_constants(self, path: Union[None, str, Path]):
        if path is None:
            path = Path.cwd() / "constants.json"
        with open(path, 'r') as f:  # for own system specific parameters, if you wanna tweek values or playing around
            self.constants = json.load(f)
        
        # energies relative to |0>
        self.E_0 = 0.0  # S1/2 with F=0 as reference
        self.E_1 = self.constants["levels"]["S1/2_hyperfine"]["F1"]["energy_eV"]
        self.E_P = self.constants["levels"]["P1/2"]["energy_eV"]
        self.E_D = self.constants["levels"]["D3/2"]["energy_eV"]
        # lifetimes / decay rates
        self.tau_P = self.constants["levels"]["P1/2"]["lifetime_ms"]      # ms
        self.gamma_P = self.constants["levels"]["P1/2"]["decay_rate_Hz"]  # Hz
        self.tau_D = self.constants["levels"]["D3/2"]["lifetime_ms"]      # ms
        self.gamma_D = self.constants["levels"]["D3/2"]["decay_rate_Hz"]  # Hz

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
    def vectorize_matrix(rho: csr_matrix) -> np.ndarray:
        return np.asarray(rho.todense()).reshape(-1, order="F")

    @staticmethod
    def unvectorize_matrix(v: np.ndarray, dim: int) -> csr_matrix:
        return csr_matrix(v.reshape((dim, dim), order="F"))

    @staticmethod
    def liouvillian(H: csr_matrix, collapses: Sequence[csr_matrix]) -> csr_matrix:
        dim = H.shape[0]  # type: ignore
        Id = identity(dim, dtype=complex, format="csr")  # type: ignore
        # coherent part
        L = -1j * (kron(Id, H) - kron(H.T, Id))
        # dissipators
        for C in collapses:
            CdC = (C.getH() @ C).tocsr()
            L += kron(C, C.conjugate())
            L += -0.5 * kron(Id, CdC.T) - 0.5 * kron(CdC, Id)
        return L.tocsr()  # type: ignore

    def initialize_state(self, state: Union[None, str, int, csr_matrix] = None):
        """
        As you can see I gave the option for string inputs of the knows qubit states as well as 0 and 1. 
        This way you can easily generate all of the mostly used basis states. More complicated states must be generated through
        Gates or by inputting a csr matrix.
        """
        options = ['0','1','+','-','i','-i', 0, 1, None]
        if state not in options:
            raise ValueError(f"Unknown string state: {state}")
        elif state in options:
            match state:
                case 0 | '0' | None:
                    psi = self.ket(self.i0)
                    self.state = (psi @ psi.getH()).tocsr()
                case 1 | '1':
                    psi = self.ket(self.i1)
                    self.state = (psi @ psi.getH()).tocsr()
                case '+':
                    psi = (self.ket(self.i0)+self.ket(self.i1))/np.sqrt(2)
                    self.state = (psi @ psi.getH()).tocsr()
                case '-':
                    psi = (self.ket(self.i0)-self.ket(self.i1))/np.sqrt(2)
                    self.state = (psi @ psi.getH()).tocsr()
                case 'i':
                    psi = (self.ket(self.i0)+1j*self.ket(self.i1))/np.sqrt(2)
                    self.state = (psi @ psi.getH()).tocsr()
                case '-i':
                    psi = (self.ket(self.i0)-1j*self.ket(self.i1))/np.sqrt(2)
                    self.state = (psi @ psi.getH()).tocsr()

        elif isinstance(state, csr_matrix):
            if state.shape != (4,4):
                raise ValueError("CSR density matrix must be of shape 4x4 to be a valid state matrix for Yb+")
            self.state = state.copy().tocsr()  # make it non-mutative just in case
        else:
            raise TypeError("state must be None, int, str, or csr_matrix.")

    def expectation(self, P: csr_matrix) -> complex:
        return (self.state @ P).diagonal().sum()

    def projections(self) -> dict:
        """
        Return the expectation values for all four levels of the IonQubit.
        Keys to access the data: '0','1','P','D'
        I thought this might be cool for the simulation but this is deprecated, I just left it so one can
        verify the initial states expectations and for playing around.
        """
        return {
            '0': self.expectation(self.P0).real,
            '1': self.expectation(self.P1).real,
            'P': self.expectation(self.PP).real,
            'D': self.expectation(self.PD).real
        }

    def paulis(self) -> Tuple[csr_matrix,csr_matrix,csr_matrix]:
        """
        Generate the Pauli matrices in our 4 dimensional hilbert space
        """
        paulix = self.P01 + self.P10
        pauliy = 1j* (-self.P01 + self.P10)
        pauliz = self.P0 - self.P1
        return (paulix, pauliy, pauliz)


    # Convenience Hamiltonians
    def H_intrinsic(self) -> csr_matrix:
        """
        Defines:        H0 = -Delta |P><P| + (delta/2) |1><1| + eps_D |D><D|

        Should describe the Qubit at rest  ---  in the simulation sometimes it makes the state explode (expectations >1e10)
        I couldn't figure out in time if it was the constants from the csv file or some other error.
        """
        H = self.E_P * self.PP + self.E_1 * self.P1 + self.E_D * self.PD
        return H.tocsr()

    def H_raman(self, Omega0: complex, phi0: float, Omega1: complex, phi1: float, Delta: float):
        """
        Should implement the Raman drive of the qubit and would resemble X/Y rotations depending on the Omega/Phi values.
        In my simulation this is somehow also not working - I tried without the detuning of the P level - with no change. 
        I sadly also didn't have time to check why the error occurs here. 
        """
        H = 0.5 * (
            Omega0 * np.exp(1j*phi0) * self.P_0 +
            Omega1 * np.exp(1j*phi1) * self.P_1 +
            np.conjugate(Omega0) * np.exp(-1j*phi0) * self._0_P +
            np.conjugate(Omega1) * np.exp(-1j*phi1) * self._1_P
        ) - Delta * self.PP
        H = (H + H.getH()).tocsr()
        return H

    def omega_z_ac_stark(self, Delta: float, Omega1: float, Omega2: float,
                     alpha0: float = 1.0, alpha1: float = 1.0) -> float:
        if Delta == 0: 
            raise ValueError("Delta must be nonzero.")
        return ((Omega1**2 + Omega2**2)/(4.0*Delta)) * (alpha0 - alpha1)

    def H_z_from_ac_stark(self, Delta: Optional[float] = None, Omega1: Optional[float] = None, Omega2: Optional[float] = None,
                        alpha0: float = 1.0, alpha1: float = 1.0, omegaZ: Optional[float] = None) -> csr_matrix:
        """
        The AC stark effect is a physical effect that occurs when off resonant light shifts the energy levels of |0> and |1>
        With two far-detuned Raman beams, each beam shifts both qubit states; the difference is a differential AC Stark shift 
        that acts like a Z-rotation. For this I made this
        """
        if omegaZ is None:
            assert Delta is not None, "Delta must be given"
            assert Omega1 is not None, "Omega1 must be given"
            assert Omega2 is not None, "Omega2 must be given"
            omegaZ = self.omega_z_ac_stark(Delta, Omega1, Omega2, alpha0, alpha1)
        return 0.5 * omegaZ * (self.P0 - self.P1)  # type: ignore

    # Convenience Collapse-operators and scaling if provided with a decay rate
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

    def C_0_1(self, gamma_up: float) -> csr_matrix:
        return self._scale(self.xop(self.i1, self.i0), gamma_up)

    def C_1_0(self, gamma_dn: float) -> csr_matrix:
        return self._scale(self.xop(self.i0, self.i1), gamma_dn)