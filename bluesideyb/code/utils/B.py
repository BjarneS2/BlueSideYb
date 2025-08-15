
""" Useful functions to make it easier to set up the simulation """
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl


def ket(dim: int, i: int) -> csr_matrix:
    data, rows, cols = [1.0], [i], [0]
    return csr_matrix((data, (rows, cols)), shape=(dim, 1), dtype=complex)

def bra(dim: int, i: int) -> csr_matrix:
    data, rows, cols = [1.0], [0], [i]
    return csr_matrix((data, (rows, cols)), shape=(1, dim), dtype=complex)

def proj(dim: int, i: int) -> csr_matrix:
    ket_i = ket(dim, i)
    bra_i = bra(dim, i)
    return ket_i @ bra_i

def xop(dim: int, i: int, j: int) -> csr_matrix:
    ket_i = ket(dim, i)
    bra_j = bra(dim, j)
    return ket_i @ bra_j

def blochvector(rho):
    ket0, ket1, bra0, bra1 = ket(rho.shape[0], 0), ket(rho.shape[0], 1), bra(rho.shape[0], 0), bra(rho.shape[0], 1)
    pauli_x = ket0 @ bra1 + ket1 @ bra0
    pauli_y = -1j * (ket0 @ bra1 - ket1 @ bra0)
    pauli_z = ket0 @ bra0 - ket1 @ bra1

    rhoA = rho.toarray()
    x = np.trace(rhoA @ pauli_x).real
    y = np.trace(rhoA @ pauli_y).real
    z = np.trace(rhoA @ pauli_z).real
    bv = np.array([x,y,z])
    return bv / np.linalg.norm(bv) if np.linalg.norm(bv) > 1 else bv  # only change if somehow the state is not normalized otherwise assume mixed contents in the state

def density(blochvec):
    x, y, z = blochvec
    ket0, ket1, bra0, bra1 = ket(2, 0), ket(2, 1), bra(2, 0), bra(2, 1)
    pauli_x = ket0 @ bra1 + ket1 @ bra0
    pauli_y = -1j * (ket0 @ bra1 - ket1 @ bra0)
    pauli_z = ket0 @ bra0 - ket1 @ bra1
    return 0.5 * (np.eye(2, dtype=complex) + x*pauli_x + y*pauli_y + z*pauli_z)

def plot_vector_matplotlib(blochvector, show_arrow=True, figsize=(8, 8), background="#222831", sphere_color="#393e46", 
                point_color="#00adb5", arrow_color="#f8b400", label_color="#eeeeee", alpha=0.15):
    """
    WARNING!: Does not work with UV - UV is headless, hence you will not see any plots. Please run your code
    in a local environment with "python your_favorite_program.py"
    Advanced 3D Bloch sphere visualization with state, axes, and key points.
    """
    # Bloch sphere setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])  # type: ignore
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_box_aspect([1,1,1])

    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color=sphere_color, alpha=alpha, linewidth=0, zorder=0) # , shade=True)

    # Draw axes
    axis_len = 1.
    ax.plot([-axis_len, axis_len], [0,0], [0,0], color="#aaaaaa", lw=1.5, zorder=1)
    ax.plot([0,0], [-axis_len, axis_len], [0,0], color="#aaaaaa", lw=1.5, zorder=1)
    ax.plot([0,0], [0,0], [-axis_len, axis_len], color="#aaaaaa", lw=1.5, zorder=1)

    # Key points on sphere give them some nice colors afterwards
    key_states = {
        '|0⟩': np.array([0, 0, 1]),
        '|1⟩': np.array([0, 0, -1]),
        '|+⟩': np.array([1, 0, 0]),
        '|-⟩': np.array([-1, 0, 0]),
        '|i⟩': np.array([0, 1, 0]),
        '|-i⟩': np.array([0, -1, 0])
    }
    key_colors = {
        '|0⟩': "#A91616",
        '|1⟩': "#f84600",
        '|+⟩': "#0f3bc9",
        '|-⟩': "#490ca5",
        '|i⟩': "#119405",
        '|-i⟩': "#0db02b"
    }
    for label, vec in key_states.items():
        ax.scatter(*vec, color=key_colors[label], s=80, edgecolor="#222831", zorder=3)
        ax.text(*(1.15*vec), label, color=label_color, fontsize=14, weight='bold', ha='center', va='center', zorder=4) # type: ignore

    # Draw latitude and longitude lines for aesthetics
    for theta in np.linspace(0, np.pi, 7)[1:-1]:
        ax.plot(np.cos(u)*np.sin(theta), np.sin(u)*np.sin(theta), np.full_like(u, np.cos(theta)),
                color="#444", lw=0.5, alpha=0.7, zorder=0)
    for phi in np.linspace(0, 2*np.pi, 13):
        ax.plot(np.cos(phi)*np.sin(v), np.sin(phi)*np.sin(v), np.cos(v),
                color="#444", lw=0.5, alpha=0.7, zorder=0)

    # Plot Bloch vector (state)
    norm = np.linalg.norm(blochvector)
    if norm > 1:  # Clamp to sphere
        blochvector = blochvector / norm
        norm = 1
    ax.scatter(*blochvector, color=point_color, s=120, edgecolor="#eeeeee", zorder=5)
    if show_arrow:
        ax.quiver(0, 0, 0, *blochvector, color=arrow_color, lw=3, arrow_length_ratio=0.18, zorder=6, alpha=0.95)

    # Add subtle shadow for the state point
    # ax.scatter(bloch[0], bloch[1], -1.01, color=point_color, s=60, alpha=0.2, zorder=2)

    # Remove axes panes
    ax.xaxis.pane.set_edgecolor(background)  # type: ignore
    ax.yaxis.pane.set_edgecolor(background)  # type: ignore
    ax.zaxis.pane.set_edgecolor(background)  # type: ignore
    ax.xaxis.pane.set_facecolor(background)  # type: ignore
    ax.yaxis.pane.set_facecolor(background)  # type: ignore
    ax.zaxis.pane.set_facecolor(background)  # type: ignore
    ax.xaxis.line.set_color((0,0,0,0))       # type: ignore
    ax.yaxis.line.set_color((0,0,0,0))       # type: ignore
    ax.zaxis.line.set_color((0,0,0,0))       # type: ignore

    # Set limits and view so the sphere is nicely centered and presented (since it is very laggy when you wanna move around)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=25, azim=45)

    # Title and annotation
    ax.set_title("Bloch Sphere Representation", color=label_color, fontsize=18, pad=20, weight='bold')
    ax.text2D(0.05, 0.95, f"State: [{blochvector[0]:.2f}, {blochvector[1]:.2f}, {blochvector[2]:.2f}]", 
                transform=ax.transAxes, color=label_color, fontsize=12, alpha=0.8)

    plt.tight_layout()
    plt.show()


