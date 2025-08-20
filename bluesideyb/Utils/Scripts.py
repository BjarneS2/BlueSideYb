""" Here are the hidden backend solver for the rk4 method and some other convenient functions that I intend to use """
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from typing import Callable, Sequence, Tuple, List, Union, Any, Optional
import warnings
try:
    from numba import njit
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False
    warnings.warn("Numba not installed - please make sure you have numba properly installed.")


def make_const_phase(phi0: float):
    def phase_fn(t: float):
        return float(phi0)
    return phase_fn

def gaussian_envelope(t0: float, sigma: float, amp: float = 1.0):
    '''
    Creates a Gaussian envelope for pulse generation
    '''
    if sigma <= 0.0:
        return lambda t: 0.0
    return lambda t: amp * np.exp(-0.5 * ((t - t0) / sigma)**2)

def rect_envelope(t_start: float, duration: float, amp: float = 1.0) -> Callable:
    '''
    Returns amp in a rectangular envelope - checks if t is in [t_start, t_start + duration]
    (If you want a callable go to make_rect_window)
    '''
    return lambda t: amp if (t_start <= t <= t_start + duration) else 0.0

def make_rect_window(duration, amp=1.0):
    """
    Returns an envelope function f(t) that is amp for 0 <= t <= duration and 0 elsewhere.
    t is relative to the pulse's own start (matrix_td_factory will handle turn_on offset).
    """
    return lambda t: amp if (0.0 <= t <= duration) else 0.0

def periodic_envelope(base_env: Callable[[float], float], period: float) -> Callable[[float], float]:
    """Repeat base_env over period."""
    if period <= 0:
        raise ValueError("period must be > 0")
    def env(t: float) -> float: # fold into [0, period)
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

def bloch_from_rho_qubit(rho: csr_matrix, sx: csr_matrix, sy: csr_matrix, sz: csr_matrix):
    A = rho.toarray()
    x = np.trace(A @ sx.toarray()).real
    y = np.trace(A @ sy.toarray()).real
    z = np.trace(A @ sz.toarray()).real
    return np.array([x, y, z], dtype=float)


def parse_c_ops(c_ops: Sequence[Union[csr_matrix, Tuple[Any, ...]]]) -> List[csr_matrix]:
    """
    Parse collapse operators (c_ops) into a list of CSR matrices

    Supports c_ops:
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

def _to_dense(A: csr_matrix) -> np.ndarray:
    '''
    Convert a sparse CSR matrix to a dense Numpy array
    '''
    return A.toarray().astype(np.complex128, copy=False)


# ---------- RK4 Solvers -----------
if _NUMBA_OK:  # first check if numba is present --> if not don't load.
    @njit(fastmath=True, parallel=True)  # type: ignore
    def _commutator(H: np.ndarray, rho: np.ndarray, out: np.ndarray):
        '''
        Computes the commutator [H, rho], by setting up two temporary buffer matrices that save the intermediate matrix products.
        The result will be saved into out. (mutated)
        '''
        tmp1 = H @ rho
        tmp2 = rho @ H
        out[:] = tmp1 - tmp2

    @njit(fastmath=True, parallel=True)  # type: ignore
    def _dissipator(Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
        '''
        Computes the dissipator D(rho) = sum ( C@rho@Cd - 0.5*{Cd@C, rho} ) -- where {A, B} = A@B + B@A
        Uses temporary buffer to store the intermediate results -- same as _commutator.
        Stores the result in the out array with mutating it.
        '''
        d = rho.shape[0]

        K = np.zeros((d,d),dtype=np.complex128)
        Krho = np.zeros((d,d),dtype=np.complex128)
        rhoK = np.zeros((d,d),dtype=np.complex128)
        tmpA = np.zeros((d, d), dtype=np.complex128)
        tmpB = np.zeros((d, d), dtype=np.complex128)
        out[:, :] = 0.0 + 0.0j
        for C in Cs:
            tmpA[:] = C@rho
            tmpB[:] = tmpA@C.conj().T
            K[:] = C.conj().T@C
            Krho[:] = K@rho
            rhoK[:] = rho@K
            out += tmpB - 0.5 * (Krho+rhoK)
            # ------- Old version with explicit matrix multiplication --------
            # store tempA = C@ρ
            # for i in prange(d):  # type: ignore
            #     for j in range(d):
            #         s = 0.0 + 0.0j
            #         for k in range(d):
            #             s += C[i, k] * rho[k, j]
            #         tmpA[i, j] = s
            # store tempB = A@C†
            # for i in prange(d):  # type: ignore
            #     for j in range(d):
            #         s = 0.0 + 0.0j
            #         for k in range(d):
            #             s += tmpA[i, k] * np.conjugate(C[j, k])
            #         tmpB[i, j] = s
            # store K = C†@C

            # for i in prange(d):  # type: ignore
            #     for j in range(d):
            #         s = 0.0 + 0.0j
            #         for k in range(d):
            #             s += np.conjugate(C[k, i]) * C[k, j]
            #         K[i, j] = s
            # compute K@ρ and ρ@K save those in buffers as well

            # for i in prange(d):  # type: ignore
            #     for j in range(d):
            #         s1 = 0.0 + 0.0j
            #         s2 = 0.0 + 0.0j
            #         for k in range(d):
            #             s1 += K[i, k] * rho[k, j]
            #             s2 += rho[i, k] * K[k, j]
            #         Krho[i, j] = s1
            #         rhoK[i, j] = s2
            # out += tempB - 0.5*(K@rho + rho@K), where K@rho and rho@K are stored in buffers
            # for i in prange(d):   # type: ignore
            #     for j in range(d):
            #         out[i, j] += tmpB[i, j] - 0.5 * (Krho[i, j] + rhoK[i, j])
            

    @njit(fastmath=True)  # type: ignore
    def _drho_dt(H: np.ndarray, Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
        '''
        Compute the derivative of the density matrix rho with respect to time according to the Master/Lindblad equation.
        Again, saving the result in the out array by mutating it.
        '''
        # out = -i [H, rho] + D(rho)
        tmp = np.zeros_like(rho)
        _commutator(H, rho, tmp)
        out[:] = -1j * tmp[:]
        # for i in range(rho.shape[0]):
        #     for j in range(rho.shape[1]):
        #         out[i, j] = -1j * tmp[i, j]
        if len(Cs) > 0:
            D = np.zeros_like(rho)
            _dissipator(Cs, rho, D)
            out += D
            # _dissipator(Cs, rho, D)
            # for i in range(rho.shape[0]):
            #     for j in range(rho.shape[1]):
            #         out[i, j] += D[i, j]


    @njit(fastmath=True)  # type: ignore
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
        # for i in range(rho.shape[0]):
        #     for j in range(rho.shape[1]):
        #         tmp[i, j] = rho[i, j] + 0.5 * dt * k1[i, j]
        tmp = rho + 0.5 * dt * k1
        _drho_dt(H, Cs, tmp, k2)  # compute the derivative but store in k2

        # Now repeat this until k4
        # for i in range(rho.shape[0]):
        #     for j in range(rho.shape[1]):
        #         tmp[i, j] = rho[i, j] + 0.5 * dt * k2[i, j]
        tmp = rho + 0.5 * dt * k2
        _drho_dt(H, Cs, tmp, k3)

        # for i in range(rho.shape[0]):
        #     for j in range(rho.shape[1]):
        #         tmp[i, j] = rho[i, j] + dt * k3[i, j]
        tmp = rho + dt * k3 
        _drho_dt(H, Cs, tmp, k4)

        # out = np.empty_like(rho)  
        # Now that we have created our output array we can put everything together
        # for i in range(rho.shape[0]):
        #     for j in range(rho.shape[1]):
        #         out[i, j] = rho[i, j] + (dt/6.0) * (k1[i, j] + 2.0*k2[i, j] + 2.0*k3[i, j] + k4[i, j])
        out = rho + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return out


def _commutator_no_jit(H: np.ndarray, rho: np.ndarray, out: np.ndarray):
    tmp1 = H @ rho
    tmp2 = rho @ H
    out[:] = tmp1 - tmp2

def _dissipator_no_jit(Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
    d = rho.shape[0]

    K = np.zeros((d,d),dtype=np.complex128)
    Krho = np.zeros((d,d),dtype=np.complex128)
    rhoK = np.zeros((d,d),dtype=np.complex128)
    tmpA = np.zeros((d, d), dtype=np.complex128)
    tmpB = np.zeros((d, d), dtype=np.complex128)
    out[:, :] = 0.0 + 0.0j
    for C in Cs:
        tmpA[:] = C@rho
        tmpB[:] = tmpA @ C.conj().T
        K[:] = C.conj().T @ C
        Krho[:] = K@rho
        rhoK[:] = rho@K
        out += tmpB - 0.5 * (Krho+rhoK)

def _drho_dt_no_jit(H: np.ndarray, Cs: List[np.ndarray], rho: np.ndarray, out: np.ndarray):
    tmp = np.zeros_like(rho)
    _commutator_no_jit(H, rho, tmp)
    out[:] = -1j * tmp[:]
    if len(Cs) > 0:
        D = np.zeros_like(rho)
        _dissipator_no_jit(Cs, rho, D)
        out += D

def _rk4_step_no_jit(H: np.ndarray, Cs: List[np.ndarray], rho: np.ndarray, dt: float) -> np.ndarray:
    k1 = np.zeros_like(rho)
    k2 = np.zeros_like(rho)
    k3 = np.zeros_like(rho)
    k4 = np.zeros_like(rho)

    _drho_dt_no_jit(H, Cs, rho, k1)
    tmp = rho + 0.5 * dt * k1
    _drho_dt_no_jit(H, Cs, tmp, k2) 
    tmp = rho + 0.5 * dt * k2
    _drho_dt_no_jit(H, Cs, tmp, k3)
    tmp = rho + dt * k3 
    _drho_dt_no_jit(H, Cs, tmp, k4)
    return rho + (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ---------- Save-strategy & memory warning -----------
def _choose_save_indices(n_steps: int, n_savepoints: Optional[int]) -> np.ndarray:
    if n_steps <= 0:
        return np.array([], dtype=int)
    if n_savepoints is None or n_savepoints <= 0 or n_savepoints >= n_steps + 1:
        return np.arange(n_steps + 1, dtype=int)
    idx = np.linspace(0, n_steps, n_savepoints, dtype=int)
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
    bytes_per_rho = (d*d) * 16
    total = bytes_per_rho * n_saved
    # Warn above ~500 MB
    if total > 500 * (1024**2):
        GB = total / (1024**3)
        warnings.warn(f"Memory warning: saving ~{GB:.2f} GiB of density matrices "
                      f"({n_saved} states of size {d}x{d}). Consider reducing n_savepoints.")
        u = input("If you are sure you want to continue, press Enter...")
        if u != "":
            raise RuntimeError("User aborted simulation due to high memory usage.")



