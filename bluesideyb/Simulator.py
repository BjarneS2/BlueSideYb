from decimal import DivisionByZero
import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import expm_multiply
from typing import Callable, Sequence, Tuple, List, Union, Any, Optional
from Utils.Scripts import _choose_save_indices, _warn_memory, _to_dense, parse_c_ops, liouvillian_from_H_and_Cs, _rk4_step_no_jit, _rk4_step
from tqdm.auto import tqdm
from numba import types
from numba.typed import List as NumbaList
try:
    from numba import njit, prange
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False



class Simulator:
    """
    mode='krylov'  -> sparse Liouvillian + expm_multiply (just as an alternative, can be used with less coarse grid but takes longer)
    mode='rk4_jit' -> Numba-JIT dense RK4 (overhead gives no advantage over rk4 so just use that)
    mode='rk4'     -> No Numba used so no jit. (fast and accurate == O(h^4): default)
    """
    def __init__(self, mode: str = "rk4"):
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
        tlist = np.asarray(tlist, dtype=float)
        if tlist.ndim != 1 or len(tlist) < 2:
            raise ValueError("tlist must be 1D with at least two points otherwise the simulation is not valid")
        n_steps = len(tlist) - 1
        save_idx = _choose_save_indices(n_steps, n_savepoints)
        _warn_memory(d, len(save_idx))

        Hc = csr_matrix((d, d), dtype=complex)
        for H in c_hamiltonians:
            Hc = (Hc + H).tocsr()

        C_scaled_csr = parse_c_ops(c_ops)

        if self.mode == "krylov":
            """
            The Krylov subspace method for time evolution. It basically makes use of the fact that you don't need the full
            e^iH matrix hamiltonian (which scales terrible with larger H), but just need the action on one state. (your initial state)
            So there is no need to save/compute the full e^iH.

            The rough idea: the e^iH is approximately a polynomial(H) [Taylor expansion]. So the Krylov subspace is the span{|state>,H|state>,H^2|state>,...}
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
                    states.append(self.unvec(v, d))  # by default I wanted to save density matrices and operate on those
                    # because Krylov subspace method acts on the vectorized form I need to convert back and forth.
            
            self.solution = (tlist[save_idx], states)  # output times and states for easier handling maybe saving
            return self.solution

        elif self.mode == "rk4_jit":
            """
            Same as rk4 method but here we convert everything to np.dense for compatibility. For an explanation of the rk4 method go to Scripts and rk4
            """

            if not _NUMBA_OK:
                raise RuntimeError("numba not available; install numba or use mode='rk4'")
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
                rho = _rk4_step(H, Cs_dense, rho, dt)  # type: ignore
                # enforce Hermiticity and trace = 1 -- if numerical issues are a source of shrinking/expanding of the state
                # We can do it here, because the full 4D state should always have trace 1
                rho = 0.5*(rho + rho.conj().T)
                if not np.isclose(np.trace(rho),0):
                    rho /= np.trace(rho)  
                else:
                    raise DivisionByZero("Something is not right with the operators. The state disappears so we encounter a Division by zero error.")
                
                if (k+1) in save_idx:
                    states.append(csr_matrix(rho))  #  create CSR matrix again from dense matrices
            self.solution = (tlist[save_idx], states)
            return self.solution
        
        elif self.mode == "rk4":
            """
            Same as rk4_jit just without jit compilation overhead. I tested this with the conversion to dense and it appears
            to not have a big effect on the runtime. The idea is to switch to numpy dense matrices, but for that I would need
            to change all the classes and operations I have so far in Gates and IonQubit
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
                Htd = np.zeros_like(Hc_dense)
                for Hfun in td_hamiltonians:
                    Htd += _to_dense(Hfun(0.5*(t0+t1)))
                H = Hc_dense + Htd
                rho = _rk4_step_no_jit(H, Cs_dense, rho, dt)
                # enforce Hermiticity and trace = 1 -- if numerical issues are a source of shrinking/expanding of the state
                rho = 0.5*(rho + rho.conj().T)
                if not np.isclose(np.trace(rho),0):
                    rho /= np.trace(rho)  
                else:
                    raise DivisionByZero("Something is not right with the operators. The state disappears so we encounter a Division by zero error.")
                
                if (k+1) in save_idx:
                    states.append(csr_matrix(rho))  #  create CSR matrix again from dense matrices
            self.solution = (tlist[save_idx], states)
            return self.solution
    
        else:
            raise ValueError("mode must be 'krylov', 'rk4_jit' or 'rk4'")
        
    def export_states(self, filename: str, format: str = "hdf5"):
        """
        I thought about saving the data, once I have a good run. This sadly didn't occur so I just have this lying around.
        Could be used to save the states in the end.
        """
        times, states = self.solution
        if format == "npz":
            np.savez(filename, times=times, states=[s.toarray() for s in states]) # numpy has a convenient zip if you prefer that
        elif format == "hdf5":
            import h5py
            with h5py.File(filename, "w") as f:
                f.create_dataset("times", data=times)
                g = f.create_group("states")
                for i, s in enumerate(states):
                    g.create_dataset(str(i), data=s.toarray())
        else:
            raise ValueError("Unsupported format")

def simulate(state: csr_matrix,
             c_hamiltonians: Sequence[csr_matrix] = (),
             td_hamiltonians: Sequence[Callable[[float], csr_matrix]] = (),
             c_ops: Sequence[Union[csr_matrix, Tuple[Any, ...]]] = (),
             tlist: Optional[np.ndarray] = None,
             dt: float = 1e-7, duration: Optional[float] = None,
             n_savepoints: Optional[int] = None,
             mode: str = "krylov",
             export_states: bool = False,
             exp_to: str = "states",
             format: str = "hdf5") -> Tuple[np.ndarray, List[csr_matrix]]:
    """
    Convenience wrapper for the simulator. With this one can more easily start and run the Simulation. That way 
    you save some lines in the script where you wanna run this. That's the idea at least.

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
        sim.export_states(filename=exp_to, format=format)
    return result
    
