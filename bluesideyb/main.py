import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from IonQubit import YbQubit as IonQubit
from Simulator import simulate
from Utils.Scripts import matrix_td_factory, make_rect_window

with open("constants.json","r") as f:
    CONST = json.load(f)

tau_P = CONST["levels"]["P1/2"]["lifetime_s"]

q = IonQubit('0')             # start in |0>
rho0 = q.state                # CSR density matrix, shape (d,d)
d = rho0.shape[0]  # type: ignore

i0 = q.i0   # basis index of |0>
i1 = q.i1   # basis index of |1>
def ket(i, dim=d):
    v = np.zeros((dim,1), dtype=np.complex128); v[i,0] = 1.0; return v
def bra(i, dim=d): return ket(i,dim).conj().T
def op(i,j, dim=d): return csr_matrix(ket(i,dim) @ bra(j,dim))
P0 = op(i0,i0,d); P1 = op(i1,i1,d)
sx = op(i0,i1,d) + op(i1,i0,d)
sy = -1j*op(i0,i1,d) + 1j*op(i1,i0,d)
sz = op(i0,i0,d) - op(i1,i1,d)
I = csr_matrix(np.eye(d, dtype=np.complex128))

c_hams = []

Ωx = 2*np.pi * 1e6      # 1 MHz Rabi
θ_H = 0.5*np.pi         # π/2 rotation
tH  = θ_H / Ωx          # duration for Hadamard-like X(π/2)
HX  = 0.5*Ωx * sx       # (Ω/2) σ_x

# Z-rotations via AC-Stark / detuning: H_Z = (Δ/2) σ_z, angle φ = Δ * t
Δz_T = 2*np.pi * 1e6    # 1 MHz detuning
φ_T  = 0.25*np.pi       # π/4
tT   = φ_T / Δz_T
HZ_T = 0.5*Δz_T * sz

Δz_Z = 2*np.pi * 1e6
φ_Z  = np.pi
tZ   = φ_Z / Δz_Z
HZ_Z = 0.5*Δz_Z * sz

# Gaussian edges for realism; center each pulse in its own window
sigma_frac = 0.15

# Build TD hams as callables with turn_on offsets and non-overlapping schedules
t0_H = 0.0
t0_T = t0_H + tH
t0_Z = t0_T + tT
T_total = t0_Z #+ tZ

H_H_td = matrix_td_factory(HX, make_rect_window(tH, amp=1.0), turn_on=t0_H)  # type: ignore
H_T_td = matrix_td_factory(HZ_T, make_rect_window(tT, amp=1.0), turn_on=t0_T)  # type: ignore
H_Z_td = matrix_td_factory(HZ_Z, make_rect_window(tZ, amp=1.0), turn_on=t0_Z)  # type: ignore
td_hams = [H_H_td, H_T_td] # , H_Z_td]
c_ops = []
N = 4001
tlist = np.linspace(0.0, T_total, N)
# Save ~400 points
n_save = 400
t_out, rhos = simulate(state=rho0, # type: ignore
                       c_hamiltonians=c_hams,
                       td_hamiltonians=td_hams,
                       c_ops=c_ops,
                       tlist=tlist,
                       dt=tlist[1]-tlist[0],
                       n_savepoints=n_save,
                       mode="rk4")  # rk4 is the fastest method

# TODO: Add to utils
def bloch(ρ):
    ρA = ρ.toarray()
    x = np.trace(ρA @ sx.toarray()).real
    y = np.trace(ρA @ sy.toarray()).real
    z = np.trace(ρA @ sz.toarray()).real
    return np.array([x,y,z])

b0 = bloch(rhos[0])
bN = bloch(rhos[-1])

print("Initial Blochvector:", b0)
print("Final   Blochvector:", bN)

# ---------- Plot start/end Bloch vectors ----------
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')
# sphere wireframe
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_wireframe(xs, ys, zs, linewidth=0.3, alpha=0.4)

ax.quiver(0,0,0,1,0,0,length=1,color='k',linewidth=1)
ax.quiver(0,0,0,0,1,0,length=1,color='k',linewidth=1)
ax.quiver(0,0,0,0,0,1,length=1,color='k',linewidth=1)

ax.scatter([b0[0]],[b0[1]],[b0[2]], s=40, label='start')  # type: ignore
ax.scatter([bN[0]],[bN[1]],[bN[2]], s=40, label='end')  # type: ignore

ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
ax.set_box_aspect([1,1,1])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()
# plt.savefig("bloch_vectors.png", dpi=300)
