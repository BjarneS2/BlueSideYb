import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from typing import List, Optional
from line_profiler import LineProfiler

# Assuming these are defined in separate files and can be imported
from IonQubit import IonQubit
from Simulator import Simulator, _rk4_step  # <-- Import the _rk4_step function here
from utils.A import matrix_td_factory, gaussian_envelope, make_rect_window, _drho_dt, _commutator

# --- Helper functions (assuming these are defined in your environment) ---
def ket(i, dim):
    v = np.zeros((dim, 1), dtype=np.complex128); v[i, 0] = 1.0; return v
def bra(i, dim): return ket(i, dim).conj().T
def op(i, j, dim): return csr_matrix(ket(i, dim) @ bra(j, dim))

def bloch(ρ):
    """
    Function to compute the Bloch vector from a density matrix ρ.
    """
    # Assuming sx, sy, sz are globally defined sparse matrices
    ρA = ρ.toarray()
    x = np.trace(ρA @ sx.toarray()).real
    y = np.trace(ρA @ sy.toarray()).real
    z = np.trace(ρA @ sz.toarray()).real
    return np.array([x, y, z])

# --- Main script starts here ---
if __name__ == '__main__':
    # Load constants and initialize qubit
    with open("constants.json", "r") as f:
        CONST = json.load(f)

    q = IonQubit('0')
    rho0 = q.state
    d = rho0.shape[0]

    i0 = q.i0
    i1 = q.i1
    
    # Define Pauli matrices as sparse matrices
    sx = op(i0, i1, d) + op(i1, i0, d)
    sy = -1j * op(i0, i1, d) + 1j * op(i1, i0, d)
    sz = op(i0, i0, d) - op(i1, i1, d)

    c_hams = []
    
    # Define pulse parameters
    Ωx = 2 * np.pi * 1e6
    θ_H = 0.5 * np.pi
    tH = θ_H / Ωx
    HX = 0.5 * Ωx * sx

    Δz_T = 2 * np.pi * 1e6
    φ_T = 0.25 * np.pi
    tT = φ_T / Δz_T
    HZ_T = 0.5 * Δz_T * sz

    Δz_Z = 2 * np.pi * 1e6
    φ_Z = np.pi
    tZ = φ_Z / Δz_Z
    HZ_Z = 0.5 * Δz_Z * sz

    # Define pulse windows
    t0_H = 0.0
    t0_T = t0_H + tH
    t0_Z = t0_T + tT
    T_total = t0_Z + tZ

    H_H_td = matrix_td_factory(HX, make_rect_window(tH, amp=1.0), turn_on=t0_H)
    H_T_td = matrix_td_factory(HZ_T, make_rect_window(tT, amp=1.0), turn_on=t0_T)
    H_Z_td = matrix_td_factory(HZ_Z, make_rect_window(tZ, amp=1.0), turn_on=t0_Z)

    td_hams = [H_H_td, H_T_td, H_Z_td]
    c_ops = []

    N = 4001
    tlist = np.linspace(0.0, T_total, N)
    n_save = 400

    # --- Profiling setup starts here ---
    # Create an instance of the Simulator class.
    sim_instance = Simulator(mode="rk4_jit")

    # Initialize the profiler
    lp = LineProfiler()

    # Add the specific functions you want to profile.
    lp.add_function(sim_instance.run)
    lp.add_function(_rk4_step)  # <-- Add the _rk4_step function here
    lp.add_function(_drho_dt)
    lp.add_function(_commutator)
    # --- Encapsulate the main simulation and plotting logic ---
    def main_run_and_plot():
        print("Starting simulation...")
        t_out, rhos = sim_instance.run(state=rho0,
                                       c_hamiltonians=c_hams,
                                       td_hamiltonians=td_hams,
                                       c_ops=c_ops,
                                       tlist=tlist,
                                       n_savepoints=n_save)
        print("Simulation complete.")

        print("Calculating initial and final Bloch vectors...")
        b0 = bloch(rhos[0])
        bN = bloch(rhos[-1])

        print("Initial Blochvector:", b0)
        print("Final   Blochvector:", bN)

        # Plotting code
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4)
        ax.quiver(0, 0, 0, 1, 0, 0, length=1, color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0, 1, 0, length=1, color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, 1, length=1, color='k', linewidth=1)
        ax.scatter([b0[0]], [b0[1]], [b0[2]], s=40, label='start')
        ax.scatter([bN[0]], [bN[1]], [bN[2]], s=40, label='end')
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # --- Run the profiler and print results ---
    lp.run('main_run_and_plot()')
    lp.print_stats()