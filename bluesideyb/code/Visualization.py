import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['toolbar'] = 'none'
from typing import Tuple, List, Optional


class BlochsphereVisualizer:
    """
    This is the maintained visualizer class. I want to make this more modular and reusable and take a bit of code from the
    IonQubit and Simulator classes.
    """

    def __init__(self, results: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        if results is not None:
            self.results = results
    
    def plot_vector_matplotlib(self, IonQubit, show_arrow=True, figsize=(8, 8), background="#222831", sphere_color="#393e46", 
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

        # Key points on sphere
        key_states = {
            '|0⟩': np.array([0, 0, 1]),
            '|1⟩': np.array([0, 0, -1]),
            '|+⟩': np.array([1, 0, 0]),
            '|-⟩': np.array([-1, 0, 0]),
            '|i⟩': np.array([0, 1, 0]),
            '|-i⟩': np.array([0, -1, 0])
        }
        key_colors = {
            '|0⟩': "#00adb5",
            '|1⟩': "#f8b400",
            '|+⟩': "#ff6363",
            '|-⟩': "#a66cff",
            '|i⟩': "#43e97b",
            '|-i⟩': "#f6416c"
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
        bloch = IonQubit.bloch_vector()
        norm = np.linalg.norm(bloch)
        if norm > 1:  # Clamp to sphere
            bloch = bloch / norm
            norm = 1
        ax.scatter(*bloch, color=point_color, s=120, edgecolor="#eeeeee", zorder=5)
        if show_arrow:
            ax.quiver(0, 0, 0, *bloch, color=arrow_color, lw=3, arrow_length_ratio=0.18, zorder=6, alpha=0.95)

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

        # Set limits and view
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.view_init(elev=25, azim=45)

        # Title and annotation
        ax.set_title("Bloch Sphere Representation", color=label_color, fontsize=18, pad=20, weight='bold')
        ax.text2D(0.05, 0.95, f"State: [{bloch[0]:.2f}, {bloch[1]:.2f}, {bloch[2]:.2f}]", 
                    transform=ax.transAxes, color=label_color, fontsize=12, alpha=0.8)

        plt.tight_layout()
        plt.show()


    def plot_blochsphere_evolution(self,
                                   figsize=(8, 8), background="#222831", sphere_color="#393e46",
                                   point_color="#00adb5", arrow_color="#f8b400", label_color="#eeeeee",
                                   alpha=0.15,
                                   results: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Plot the evolution of the Bloch sphere representation of the quantum state over time.
        """
        if not hasattr(self, 'tlist') or not hasattr(self, 'states'):
            if results is None:
                raise RuntimeError("BlochsphereVisualizer not initialized with simulation data, nor given when called.")
            else:
                self.results = results
                
        bloch_vectors = [blochvector(state) for state in self.states]
        
        self._plot_multiple_vectors_matplotlib(
            bloch_vectors, self.tlist, figsize=figsize, background=background,
            sphere_color=sphere_color, point_color=point_color,
            arrow_color=arrow_color, label_color=label_color, alpha=alpha
        )

    @staticmethod
    def _plot_multiple_vectors_matplotlib(
        bloch_vectors: List[np.ndarray], tlist: np.ndarray,
        figsize=(8, 8), background="#222831", sphere_color="#393e46",
        point_color="#00adb5", arrow_color="#f8b400", label_color="#eeeeee",
        alpha=0.15
    ):
        """
        Helper function to plot multiple Bloch vectors on a single Bloch sphere.
        This is a more efficient approach for plotting the evolution.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_box_aspect([1, 1, 1])

        # Draw sphere
        u, v = np.mgrid[0:2 * np.pi:200j, 0:np.pi:100j]
        x_s, y_s, z_s = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
        ax.plot_surface(x_s, y_s, z_s, color=sphere_color, alpha=alpha, linewidth=0, zorder=0)

        # Draw axes and key points (same as before)
        axis_len = 1.
        ax.plot([-axis_len, axis_len], [0, 0], [0, 0], color="#aaaaaa", lw=1.5, zorder=1)
        ax.plot([0, 0], [-axis_len, axis_len], [0, 0], color="#aaaaaa", lw=1.5, zorder=1)
        ax.plot([0, 0], [0, 0], [-axis_len, axis_len], color="#aaaaaa", lw=1.5, zorder=1)

        key_states = {
            '|0⟩': np.array([0, 0, 1]), '|1⟩': np.array([0, 0, -1]),
            '|+⟩': np.array([1, 0, 0]), '|−⟩': np.array([-1, 0, 0]),
            '|i⟩': np.array([0, 1, 0]), '|−i⟩': np.array([0, -1, 0])
        }
        key_colors = {
            '|0⟩': "#00adb5", '|1⟩': "#f8b400", '|+⟩': "#ff6363",
            '|−⟩': "#a66cff", '|i⟩': "#43e97b", '|−i⟩': "#f6416c"
        }
        for label, vec in key_states.items():
            ax.scatter(*vec, color=key_colors[label], s=80, edgecolor="#222831", zorder=3)
            ax.text(*(1.15 * vec), label, color=label_color, fontsize=14, weight='bold', ha='center', va='center', zorder=4)

        # Draw latitude and longitude lines
        for theta in np.linspace(0, np.pi, 7)[1:-1]:
            ax.plot(np.cos(u) * np.sin(theta), np.sin(u) * np.sin(theta), np.full_like(u, np.cos(theta)),
                    color="#444", lw=0.5, alpha=0.7, zorder=0)
        for phi in np.linspace(0, 2 * np.pi, 13):
            ax.plot(np.cos(phi) * np.sin(v), np.sin(phi) * np.sin(v), np.cos(v),
                    color="#444", lw=0.5, alpha=0.7, zorder=0)
        
        # Plot the evolution points and path
        bloch_vectors_arr = np.array(bloch_vectors)
        ax.plot(bloch_vectors_arr[:, 0], bloch_vectors_arr[:, 1], bloch_vectors_arr[:, 2],
                color=point_color, marker='o', markersize=4, linestyle='--', linewidth=1.5, zorder=5, alpha=0.8)
        
        # Add a marker and arrow for the final state
        final_bv = bloch_vectors_arr[-1]
        ax.scatter(*final_bv, color=point_color, s=120, edgecolor="#eeeeee", zorder=6)
        ax.quiver(0, 0, 0, *final_bv, color=arrow_color, lw=3, arrow_length_ratio=0.18, zorder=7, alpha=0.95)

        # Remove axes panes
        ax.xaxis.pane.set_edgecolor(background)
        ax.yaxis.pane.set_edgecolor(background)
        ax.zaxis.pane.set_edgecolor(background)
        ax.xaxis.pane.set_facecolor(background)
        ax.yaxis.pane.set_facecolor(background)
        ax.zaxis.pane.set_facecolor(background)
        ax.xaxis.line.set_color((0, 0, 0, 0))
        ax.yaxis.line.set_color((0, 0, 0, 0))
        ax.zaxis.line.set_color((0, 0, 0, 0))

        # Set limits and view
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.view_init(elev=25, azim=45)

        # Title and annotation
        ax.set_title("Bloch Sphere State Evolution", color=label_color, fontsize=18, pad=20, weight='bold')
        ax.text2D(0.05, 0.95, f"Evolution: {len(tlist)} points",
                  transform=ax.transAxes, color=label_color, fontsize=12, alpha=0.8)

        plt.show()