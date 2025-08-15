# --- Data and file handling ---
import json;  ''' Import constants.json to initialize the system '''
from pathlib import Path;  ''' Used for path management when finding the constants.json file '''

# --- numerics and computation ---
import numpy as np; ''' Used for numerical operations and array manipulations in addition to sparse matrices in scipy'''
from scipy.sparse import csr_matrix, csr_array, kron, identity; ''' Sparse matrices and arrays for faster computation '''
from scipy.sparse.linalg import expm_multiply, LinearOperator;  ''' Sparse linear algebra operations for efficient matrix exponential '''

# --- type hinting for nicer code and easier usage ---
from typing import Callable, Sequence, Tuple, List, Union, Any, Optional, Literal, Dict

# --- Matplotlib for plotting of course ---
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['toolbar'] = 'none'

# --- additional tools ---
import warnings;  ''' for reference so user knows about data heavy/ saving operations '''
from tqdm import tqdm; ''' Check for progress in simulation '''
from utils.A import _choose_save_indices, _warn_memory, _to_dense, parse_c_ops, liouvillian_from_H_and_Cs, _rk4_step
from utils.B import blochvector
# --- numba for speed-up (njit for compilation and parallelization)
from numba import types
from numba.typed import List as NumbaList
''' Provide a Fallback: If _NUMBA_OK is False, the program knows Numba is unavailable and can execute a standard, 
    pure Python version of the same code. 
'''
try:
    from numba import njit, prange
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False



class Simulator:
    """
    mode='krylov' -> sparse Liouvillian + expm_multiply (general default)
    mode='rk4_jit' -> Numba-JIT dense RK4 (very fast for a single small qubit if numba available)
    """
    def __init__(self, mode: str = "krylov"):
        self.mode = mode

    @staticmethod
    def vec(rho: csr_matrix) -> np.ndarray:
        return np.asarray(rho.toarray(), dtype=np.complex128).reshape(-1, order='F')

    @staticmethod
    def unvec(v: np.ndarray, d: int) -> csr_matrix:
        arr = np.asarray(v, dtype=np.complex128).reshape((d, d), order='F')
        return csr_matrix(arr)

    def run(self,
            state: csr_matrix,
            c_hamiltonians: Sequence[csr_matrix],
            td_hamiltonians: Sequence[Callable[[float], csr_matrix]],
            c_ops: Sequence[Union[csr_matrix, Tuple[Any, ...]]],
            tlist: np.ndarray,
            n_savepoints: Optional[int] = None) -> Tuple[np.ndarray, List[csr_matrix]]:

        d = state.shape[0]  # type: ignore
        tlist = np.asarray(tlist, dtype=float)  # just to make sure, becasue type hinting in python is not actually preventing anything
        if tlist.ndim != 1 or len(tlist) < 2:
            raise ValueError("tlist must be 1D with at least two points otherwise the simulation is not valid")
        n_steps = len(tlist) - 1  # steps for calculating the time evolution
        save_idx = _choose_save_indices(n_steps, n_savepoints)  # select the states that are to be saved
        _warn_memory(d, len(save_idx))  # check if the memory usage is reasonable

        # constant H
        Hc = csr_matrix((d, d), dtype=complex)
        for H in c_hamiltonians:
            Hc = (Hc + H).tocsr()

        C_scaled_csr = parse_c_ops(c_ops)

        if self.mode == "krylov":
            """
            The Krylov subspace method for time evolution. It basically makes use of the fact that you don't need the full
            e^iH matrix hamiltonian (which scales terrible with larger H), but just need the action on one state. (your initial state)
            So there is no need to save/compute the full e^iH.

            The rough idea: the e^iH is approximately a polynomial(H). So the Krylov subspace is the span{|state>,H|state>,H^2|state>,...}
            Up to an amount "m" with some accuracy epsilon. Applying all these polynomials of degree (m-1) to the initial state
            are now in the Krylov space.
            
            For more information please go to docs to see references.
            """

            # precompute constant Liouvillian parts
            I = identity(d, dtype=complex, format='csr')  # type: ignore 
            L_constH = (-1j) * (kron(I, Hc) - kron(Hc.T, I))
            if len(C_scaled_csr) > 0:  # if we have collapse operators
                L_diss = liouvillian_from_H_and_Cs(csr_matrix((d, d), dtype=complex), C_scaled_csr)
            else:
                L_diss = csr_matrix((d*d, d*d), dtype=complex)
            L_const = (L_constH + L_diss).tocsr()

            v = self.vec(state)
            states: List[csr_matrix] = []
            # save initial if in save set
            if 0 in save_idx:
                states.append(csr_matrix(state))
            for k in tqdm(range(n_steps), desc="Simulation progress"):
                t0, t1 = tlist[k], tlist[k+1]
                dt = t1 - t0
                # time-dependent addition (midpoint)
                if len(td_hamiltonians) > 0:
                    tmid = 0.5 * (t0 + t1)
                    Htd = csr_matrix((d, d), dtype=complex)
                    for Hfun in td_hamiltonians:
                        Htd = (Htd + Hfun(tmid)).tocsr()
                    Ltd = (-1j) * (kron(I, Htd) - kron(Htd.T, I))
                    L = (L_const + Ltd).tocsr()
                else:
                    L = L_const
                v = expm_multiply(L * dt, v)
                if (k+1) in save_idx:
                    states.append(self.unvec(v, d))
            
            self.solution = (tlist[save_idx], states)
            return self.solution

        elif self.mode == "rk4_jit":   # Numba optimized mode switch from scipy sparse to numpy dense matrices
            if not _NUMBA_OK:
                raise RuntimeError("numba not available; install numba or use mode='krylov'")
            """
            Convert sparse CSR matrices to dense Numpy arrays for Numba compatibility, because Numba doesn't support CSR apparently.
            """
            
            Hc_dense = _to_dense(Hc)
            Cs_dense = NumbaList.empty_list(types.complex128[:, :])  # needs to be a list of 2D arrays of type complex128 to be compatible with _to_dense(C) and for Numba to have the right type
            for C in C_scaled_csr:
                Cs_dense.append(_to_dense(C))

            rho = state.toarray().astype(np.complex128)
            states: List[csr_matrix] = []
            if 0 in save_idx:
                states.append(csr_matrix(rho))

            for k in tqdm(range(n_steps), desc="Simulation progress", total=n_steps):
                t0, t1 = tlist[k], tlist[k+1]
                dt = t1 - t0
                Htd = np.zeros_like(Hc_dense)  # for time-dependent Hamiltonians
                for Hfun in td_hamiltonians:
                    Htd += _to_dense(Hfun(0.5*(t0+t1)))  # convert time-dependent Hamiltonians to dense
                H = Hc_dense + Htd
                rho = _rk4_step(H, Cs_dense, rho, dt)  # run the RK4 solver
                # enforce Hermiticity and trace = 1 -- if numerical issues are a source of shrinking/expanding of the state
                rho = 0.5*(rho + rho.conj().T)
                tr = np.trace(rho)
                if np.abs(tr) > 0:
                    rho /= tr
                if (k+1) in save_idx:
                    states.append(csr_matrix(rho))  #  create CSR matrix again from dense matrices
            self.solution = (tlist[save_idx], states)
            return self.solution

        else:
            raise ValueError("mode must be 'krylov' or 'rk4_jit'")


def simulate(state: csr_matrix,
             c_hamiltonians: Sequence[csr_matrix] = (),
             td_hamiltonians: Sequence[Callable[[float], csr_matrix]] = (),
             c_ops: Sequence[Union[csr_matrix, Tuple[Any, ...]]] = (),
             tlist: Optional[np.ndarray] = None,
             dt: float = 1e-7, duration: Optional[float] = None,
             n_savepoints: Optional[int] = None,
             mode: str = "krylov",
             export_states: bool = False,
             exp_to: Optional[str] = None) -> Tuple[np.ndarray, List[csr_matrix]]:
    """
    Convenience wrapper for the simulator.

    Parameters
    - state: Initial state vector (csr_matrix)
    - c_hamiltonians: List of time-independent Hamiltonians (csr_matrix)
    - td_hamiltonians: List of time-dependent Hamiltonians (functions of time)
    - c_ops: List of collapse operators (csr_matrix or tuple)
    - dt: your 'time scale'. If tlist is None, we create tlist = np.arange(0, duration, dt).
           (For pulses you usually supply tlist yourself.)
    - duration: Optional, if tlist is not given will then calculate with dt
    - n_savepoints: how many states to store (None => store everything). Always includes t0 and t_end.
    - mode: 'krylov' (scipy sparse), 'rk4_jit' (Numba dense)

    Return: Simulation results (time_stamps, states)
    """
    if tlist is None:
        if duration is None and dt is not None:
            raise ValueError("Pass tlist (e.g. np.linspace(0, duration, N)). Or pass duration, because dt alone isn't enough to infer full duration.")
        else:
            tlist = np.arange(0.0, duration, dt)
    sim = Simulator(mode=mode)
    result = sim.run(state, c_hamiltonians, td_hamiltonians, c_ops, tlist, n_savepoints)
    if export_states:
        warnings.warn("Exporting states is deprecated. Use the return value of simulate() directly")
        # TODO: Please add export_states function
        # sim.export_states(exp_to, state, c_hamiltonians, td_hamiltonians, c_ops, tlist, n_savepoints)
    return result
    
