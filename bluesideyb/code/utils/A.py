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
from tqdm.auto import tqdm; ''' Check for progress in simulation '''
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




# ---------- Envelopes / pulses ----------

def gaussian_envelope(t: float, t0: float, sigma: float, amp: float = 1.0) -> float:
    '''
    Creates a Gaussian envelope function
    '''
    if sigma <= 0.0:
        return 0.0
    x = (t - t0) / sigma
    return amp * np.exp(-0.5 * x**2)

def rect_envelope(t: float, t_start: float, duration: float, amp: float = 1.0) -> float:
    '''
    Returns amp in a rectangular envelope - checks if t is in [t_start, t_start + duration]
    (If you want a callable go to make_rect_window)
    '''
    return amp if (t_start <= t <= t_start + duration) else 0.0

def make_rect_window(duration, amp=1.0):
    """
    Returns an envelope function f(t) that is amp for 0 <= t <= duration and 0 elsewhere.
    t is relative to the pulse's own start (matrix_td_factory will handle turn_on offset).
    """
    return lambda t: amp if (0.0 <= t <= duration) else 0.0

def periodic_envelope(base_env: Callable[[float], float], period: float) -> Callable[[float], float]:
    """Repeat base_env over period (cache-friendly)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    def env(t: float) -> float:
        # fold into [0, period)
        return base_env(t % period)
    return env

def matrix_td_factory(base: csr_matrix,
                      envelope: Callable[[float], float],
                      turn_on: float = 0.0) -> Callable[[float], csr_matrix]:
    '''
    Create a time-dependent matrix function Ht(t) based on the given envelope function.
    '''
    zero = csr_matrix(base.shape, dtype=complex)
    def Ht(t: float) -> csr_matrix:
        tt = t - turn_on
        if tt < 0.0:
            return zero
        a = envelope(tt)
        if a == 0.0:
            return zero
        return (a * base).tocsr()
    return Ht



# ---------- Collapse parsing ----------
def parse_c_ops(c_ops: Sequence[Union[csr_matrix, Tuple[Any, ...]]]) -> List[csr_matrix]:
    """
    Parse collapse operators (c_ops) into a list of scaled CSR matrices

    Supported c_ops parameter types:
      - csr_matrix already scaled as C = sqrt(gamma) * L
      - (L, tau)              -> gamma = 1/tau
      - (L, 'rate', gamma)    -> direct gamma

    Return list of scaled CSR C.
    """
    out = []
    for item in c_ops:
        if isinstance(item, csr_matrix):
            out.append(item.tocsr())
        elif isinstance(item, (tuple, list)):
            if len(item) == 2 and isinstance(item[1], (int, float)):
                L, tau = item
                if tau <= 0:
                    raise ValueError("tau must be > 0")
                g = 1.0 / float(tau)
                out.append(np.sqrt(g) * L.tocsr())
            elif len(item) == 3 and item[1] == 'rate':
                L, _, g = item
                if g < 0:
                    raise ValueError("gamma must be >= 0")
                out.append(np.sqrt(float(g)) * L.tocsr())
            else:
                raise ValueError("Bad c_ops tuple; use (L,tau) or (L,'rate',gamma)")
        else:
            raise ValueError("c_ops entries must be csr_matrix or tuple")
    return out

# ---------- Liouvillian (sparse) ----------
def liouvillian_from_H_and_Cs(H: csr_matrix, C_list: Sequence[csr_matrix]) -> csr_matrix:
    '''
    Create a Liouvillian superoperator from a Hamiltonian and a list of Collapse operators
    '''
    d = H.shape[0]  # type: ignore
    I = identity(d, dtype=complex, format='csr')  # type: ignore
    L = (-1j) * (kron(I, H) - kron(H.T, I))
    for C in C_list:
        CdC = (C.getH() @ C).tocsr()
        L = L + kron(C, C.conjugate())
        L = L - 0.5 * kron(I, CdC.T) - 0.5 * kron(CdC, I)
    return L.tocsr()  # type: ignore

# ---------- Dense JIT RHS (single qubit, but general d) ----------
def _to_dense(A: csr_matrix) -> np.ndarray:
    '''
    Convert a sparse CSR matrix to a dense Numpy array
    '''
    return A.toarray().astype(np.complex128, copy=False)

if _NUMBA_OK:
    # @njit(fastmath=True, parallel=True)  # type: ignore
    def _commutator(H: np.ndarray, rho: np.ndarray, out: np.ndarray):
        '''
        Computes the commutator [H, rho], by setting up two temporary buffer matrices that save the intermediate matrix products.
        The result will be saved into out. (mutated)
        '''
        d = H.shape[0]
        
        tmp1 = np.zeros((d, d), dtype=np.complex128)
        tmp2 = np.zeros((d, d), dtype=np.complex128)
        # tmp1 = H @ rho
        for i in prange(d):  # type: ignore
            for j in range(d):
                s = 0.0 + 0.0j
                for k in range(d):
                    s += H[i, k] * rho[k, j]
                tmp1[i, j] = s
        # tmp2 = rho @ H
        for i in prange(d):  # type: ignore
            for j in range(d):
                s = 0.0 + 0.0j
                for k in range(d):
                    s += rho[i, k] * H[k, j]
                tmp2[i, j] = s
        # out = tmp1 - tmp2
        for i in prange(d):  # type: ignore
            for j in range(d):
                out[i, j] = tmp1[i, j] - tmp2[i, j]

    @njit(fastmath=True, parallel=True)  # type: ignore
    def _dissipator(Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
        '''
        Computes the dissipator D(rho) = sum ( C@rho@Cd - 0.5*{Cd@C, rho} ) -- where {A, B} = A@B + B@A
        Uses temporary buffer to store the intermediate results -- same as _commutator.
        Stores the result in the out array with mutating it.
        '''
        d = rho.shape[0]
        tmpA = np.zeros((d, d), dtype=np.complex128)
        tmpB = np.zeros((d, d), dtype=np.complex128)
        out[:, :] = 0.0 + 0.0j
        for idx in range(len(Cs)):
            C = Cs[idx]
            # store tempA = C@ρ
            for i in prange(d):  # type: ignore
                for j in range(d):
                    s = 0.0 + 0.0j
                    for k in range(d):
                        s += C[i, k] * rho[k, j]
                    tmpA[i, j] = s
            # store tempB = A@C†
            for i in prange(d):  # type: ignore
                for j in range(d):
                    s = 0.0 + 0.0j
                    for k in range(d):
                        s += tmpA[i, k] * np.conjugate(C[j, k])
                    tmpB[i, j] = s
            # store K = C†@C
            K = np.zeros((d, d), dtype=np.complex128)
            for i in prange(d):  # type: ignore
                for j in range(d):
                    s = 0.0 + 0.0j
                    for k in range(d):
                        s += np.conjugate(C[k, i]) * C[k, j]
                    K[i, j] = s
            # compute K@ρ and ρ@K save those in buffers as well
            Krho = np.zeros((d, d), dtype=np.complex128)
            rhoK = np.zeros((d, d), dtype=np.complex128)
            for i in prange(d):  # type: ignore
                for j in range(d):
                    s1 = 0.0 + 0.0j
                    s2 = 0.0 + 0.0j
                    for k in range(d):
                        s1 += K[i, k] * rho[k, j]
                        s2 += rho[i, k] * K[k, j]
                    Krho[i, j] = s1
                    rhoK[i, j] = s2
            # out += tempB - 0.5*(K@rho + rho@K), where K@rho and rho@K are stored in buffers
            for i in prange(d):   # type: ignore
                for j in range(d):
                    out[i, j] += tmpB[i, j] - 0.5 * (Krho[i, j] + rhoK[i, j])

    # @njit(fastmath=True)  # type: ignore
    def _drho_dt(H: np.ndarray, Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
        '''
        Compute the derivative of the density matrix rho with respect to time according to the Master/Lindblad equation.
        Again, saving the result in the out array by mutating it.
        '''
        # out = -i [H, rho] + D(rho)
        tmp = np.zeros_like(rho)
        _commutator(H, rho, tmp)
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                out[i, j] = -1j * tmp[i, j]
        if len(Cs) > 0:
            D = np.zeros_like(rho)
            _dissipator(Cs, rho, D)
            for i in range(rho.shape[0]):
                for j in range(rho.shape[1]):
                    out[i, j] += D[i, j]

    #@njit(fastmath=True)  # type: ignore
    def _rk4_step(H: np.ndarray, Cs: List[np.ndarray], rho: np.ndarray, dt: float) -> np.ndarray:
        ''' 
        The Runge-Kutta (explicit) solvers that follow the same principle. There exist all kinds of solvers, but this
        is one of the most widely used due to its simplicity but effectiveness.
        It works like this:
        Define a initial value problem. (In our case start in a state rho and define the time evolution operator - Lindblad eqn.)
        
        But let's say y(t) is our unknown function of time t and y0 being the initial value. It is described by the following:
        dy/dt = f(y,t); y0 = const.
        Then to solve this we go in small time steps h:
        y_n+1 = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)  # the choices for the constants can be looked up in the docs
        t_n+1 = t_n + h

        Where y_0 = y_0 and t_0 = 0.

        Then the k1-4 are being computed by these:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h/2 * k1)
        k3 = f(t_n + h/2, y_n + h/2 * k2)
        k4 = f(t_n + h, y_n + h * k3)

        The careful reader has observed that we are only approximating the function y(t) by discretizing on a finite grid of time.
        This is defined by the user in the simulation class by either giving dt,duration or the tlist. The smaller the steps the
        more close will the solution be to the true trajectory of the system. Of course it is also limited by the numerical 
        stability and machine precision. This Runge-Kutta method is of O(h^4) - 4th order accumulated error and 5th order local
        truncation error.

        In the docs you will find an explanation of explicit Runge-Kutta solvers, which is the class RK4 belongs to. Feel free to
        check it out.
        '''
        k1 = np.zeros_like(rho)
        k2 = np.zeros_like(rho)
        k3 = np.zeros_like(rho)
        k4 = np.zeros_like(rho)
        tmp = np.zeros_like(rho)

        _drho_dt(H, Cs, rho, k1)  # compute the derivative using k1 as empty storage

        # Now we can save the intermediate result in tmp
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                tmp[i, j] = rho[i, j] + 0.5 * dt * k1[i, j]
        _drho_dt(H, Cs, tmp, k2)  # compute the derivative but store in k2

        # Now repeat this until k4
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                tmp[i, j] = rho[i, j] + 0.5 * dt * k2[i, j]
        _drho_dt(H, Cs, tmp, k3)

        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                tmp[i, j] = rho[i, j] + dt * k3[i, j]
        _drho_dt(H, Cs, tmp, k4)

        out = np.empty_like(rho)  
        # Now that we have created our output array we can put everything together
        for i in range(rho.shape[0]):
            for j in range(rho.shape[1]):
                out[i, j] = rho[i, j] + (dt/6.0) * (k1[i, j] + 2.0*k2[i, j] + 2.0*k3[i, j] + k4[i, j])
        return out

# ---------- Save-strategy & memory warning ----------
def _choose_save_indices(n_steps: int, n_savepoints: Optional[int]) -> np.ndarray:
    if n_steps <= 0:
        # nothing to solve
        return np.array([], dtype=int)
    if n_savepoints is None or n_savepoints <= 0 or n_savepoints >= n_steps + 1:
        # save every step
        return np.arange(n_steps + 1, dtype=int)
    
    # always include first and last, distribute interior roughly evenly
    idx = np.linspace(0, n_steps, n_savepoints, dtype=int)  # create a list of indices of saved steps (convenient with dtype=int)
    idx[0] = 0
    idx[-1] = n_steps
    return np.unique(idx)

def _warn_memory(d: int, n_saved: int):
    '''
    I wanted to have a memory warning if the saved states exceed a certain amount - because I don't have any other flag
    and I want to avoid memory issues or crashes. This is just a warning - one could also implement a sophisticated error
    message that takes into account the specific memory usage and patterns of the simulation and checks how much space 
    is even available. I just don't know how to do that and I don't want to spend time on that.
    '''
    bytes_per_rho = (d*d) * 16  # since we are using complex128
    total = bytes_per_rho * n_saved
    # Warn above ~500 MB change to you preferences
    if total > 500 * (1024**2):
        GB = total / (1024**3)
        warnings.warn(f"Memory warning: saving ~{GB:.2f} GiB of density matrices "
                      f"({n_saved} states of size {d}x{d}). Consider reducing n_savepoints.")
        u = input("If you are sure you want to continue, press Enter...")
        if u != "":
            raise RuntimeError("User aborted simulation due to high memory usage.")
