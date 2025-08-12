# Outline of the project

**This will be used as an outline for the project and gives a timeline to see which goals are feasible and to keep a good track of the progress**

---

**Day 1**

- Collect reference numbers & write a YAML/JSON file with all atomic numbers (wavelengths, τ, branching).
- Create code skeleton: classes Level, Laser, MotionalMode, System. Implement operator builders and caching. Set up Git repo/Jupyter.
- Think of potential example usages that make it easy for third parties to use intuitively, maybe something like the following:
```python
from bluesideyb import IonQubit, LaserDrive, simulate

# Initialize qubit system with default parameters
qubit = IonQubit()

# Define a resonant π-pulse laser drive
pi_pulse = LaserDrive(frequency=qubit.hyperfine_freq, duration=pi/qubit.rabi_freq)

# Run simulation with drive and no noise
result = simulate(qubit, drives=[pi_pulse])

# Plot population dynamics
result.plot_populations()
```
Or if I get that far with 2 qubit implementation:
```python
from bluesideyb import TwoIonSystem, MolmerSorensenDrive, simulate

system = TwoIonSystem()

ms_gate = MolmerSorensenDrive(detuning=system.trap_freq + 2e3, duration=50e-6)

result = simulate(system, drives=[ms_gate])

result.plot_fidelity()
```

---

**Day 2**

- Implement the Hamiltonian for internal + motional mode (carrier + sideband). Implement collapse ops. Add 369 nm detection pulses and 935 nm repump as coherent pump (or effective incoherent repump).
- Validate: steady-state under continuous cooling; simulate fluorescence counts. Plot scattering vs detuning.

---

**Day 3**

- Implement microwave carrier and off-resonant Raman drive (two-beam model or effective Ω and Δ). Include spontaneous Raman scattering model (use Ozeri 2007 formula as a starting point). Add laser noise models (phase/amp OU).
- Validate: Rabi flops with noise; compute T₂ from phase noise. Parameter sweep to estimate gate infidelity.

---

**Day 4**

- Add second ion’s internal space, shared motional mode. Implement bichromatic MS Hamiltonian in Lamb-Dicke limit. Simulate small two-ion gate (choose trap freq ∼1–3 MHz, η~0.05 – choose paramaters and evaluate fidelity vs thermal occupation).
- Optimize sparse / Krylov integrator for Liouvillian propagation of the gate pulse.

---

**Day 5**

- Run parameter sweeps (laser detuning, Rabi, gate time, n_max) in parallel and cache results. Create summary plots: gate fidelity vs detuning, effect of laser linewidth, effect of motional heating.
- Package minimal scripts, short README, and data table of atomic constants + citations.

---

**Day 6-7**

- Add proper documentation and very all tests and examples. Update README file and make sure requirements are up to date.
- If time allows, optimize the computations and speed things up. (Use: scipy.sparse -> expm_multiply and spsolve, numpy, optionally qutip, jit from numba (njit(parallel=True)) or jax, lru_cache, lprun for diagnostics, ...)

