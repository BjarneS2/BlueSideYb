import numpy as np
from typing import Tuple, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['toolbar'] = 'none'


class BlochsphereVisualizer:
    """
    I wanted to make this more modular and reusable and take a bit of code from the
    Yb+ and Simulator classes. I thought visualization is important so I wanted a nice
    Visualizer class, where one can visualize evolutions and single states.
    """

    def __init__(self, results: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        if results is not None:
            self.results = results
    
    def plot_vector_matplotlib(self, IonQubit, show_arrow=True, figsize=(8, 8), background="#222831", sphere_color="#393e46", 
                    point_color="#00adb5", arrow_color="#f8b400", label_color="#eeeeee", alpha=0.15):
        """
        WARNING!: Does not work with UV - UV is headless, hence you will not see any plots. Please run your code
        in a local environment with "python your_favorite_program.py"
        3D Bloch sphere visualization with state, axes, and highlighted basis states.
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

        # Draw axes so they don't go out of the Blochsphere, I thought this looks a bit nicer.
        axis_len = 1.
        ax.plot([-axis_len, axis_len], [0,0], [0,0], color="#aaaaaa", lw=1.5, zorder=1)
        ax.plot([0,0], [-axis_len, axis_len], [0,0], color="#aaaaaa", lw=1.5, zorder=1)
        ax.plot([0,0], [0,0], [-axis_len, axis_len], color="#aaaaaa", lw=1.5, zorder=1)


        """ I wanted to make explicit symbols on the axes so one can more easily navigate the sphere. """
        key_states = {
            '|0⟩': np.array([0, 0, 1]),
            '|1⟩': np.array([0, 0, -1]),
            '|+⟩': np.array([1, 0, 0]),
            '|-⟩': np.array([-1, 0, 0]),
            '|i⟩': np.array([0, 1, 0]),
            '|-i⟩': np.array([0, -1, 0])
        }
        # Give it some colors, randomly chosen, pick other ones if you'd like.
        key_colors = {
            '|0⟩': "#00adb5",
            '|1⟩': "#f8b400",
            '|+⟩': "#ff6363",
            '|-⟩': "#a66cff",
            '|i⟩': "#43e97b",
            '|-i⟩': "#f6416c"
        }
        # Plot the |.> symbols
        for label, vec in key_states.items():
            ax.scatter(*vec, color=key_colors[label], s=80, edgecolor="#222831", zorder=3)
            ax.text(*(1.15*vec), label, color=label_color, fontsize=14, weight='bold', ha='center', va='center', zorder=4) # type: ignore

        # Draw latitude and longitude lines to make the sphere a bit more visually pleasing
        for theta in np.linspace(0, np.pi, 7)[1:-1]:
            ax.plot(np.cos(u)*np.sin(theta), np.sin(u)*np.sin(theta), np.full_like(u, np.cos(theta)),
                    color="#444", lw=0.5, alpha=0.7, zorder=0)
        for phi in np.linspace(0, 2*np.pi, 13):
            ax.plot(np.cos(phi)*np.sin(v), np.sin(phi)*np.sin(v), np.cos(v),
                    color="#444", lw=0.5, alpha=0.7, zorder=0)

        # Plot Bloch vector (state)
        bloch = IonQubit.bloch_vector()
        norm = np.linalg.norm(bloch)
        if norm > 1:  # Clamp to sphere --> shoudl not be necessary if the simulation worked!
            bloch = bloch / norm
            norm = 1
        ax.scatter(*bloch, color=point_color, s=120, edgecolor="#eeeeee", zorder=5)
        if show_arrow:
            ax.quiver(0, 0, 0, *bloch, color=arrow_color, lw=3, arrow_length_ratio=0.18, zorder=6, alpha=0.95)

        # Add subtle shadow for the state point
        # ax.scatter(bloch[0], bloch[1], -1.01, color=point_color, s=60, alpha=0.2, zorder=2)
        # This would be visible under the bloch sphere on the plane but I decided not to put it, you can add it again if you like

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

        # Set limits and view so one has a good overview of the bloch sphere immediately
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.view_init(elev=25, azim=45)

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
        I haven't had the time to write something here, but I tought this will be very important after having a simulation ready.
        """
        pass
