# BlueSideYb

**Numerical simulation of Yb⁺ trapped-ion quantum dynamics with gate operations and open-system effects.**

---

## Introduction

Trapped ¹⁷¹Yb⁺ ions are a leading platform for quantum computing due to their well-characterized hyperfine qubits, long coherence times, and accessible optical transitions. This repository provides a numerical framework to simulate the quantum dynamics of Yb⁺ qubits under laser driving, including dissipative effects modeled by the Lindblad master equation. The goal is to enable realistic studies of gate fidelities, noise impact, and potentially multi-qubit coupling via motional modes. This one-week-project was created in the scope of a course I took at DTU (Scientific Computing for Quantum Information Science). 

---

## Theoretical Background

- **Qubit encoding:**  
  Qubits are encoded in the ground-state hyperfine manifold of ¹⁷¹Yb⁺.

- **Key transitions:**  
  - Cooling/detection: 2S₁/₂ ↔ 2P₁/₂ at 369.5 nm  
  - Repump: at 935 nm  

- **Hamiltonian model:**  
  Includes internal energy splitting, laser-driven carrier and sideband drives, and potentially Mølmer–Sørensen entangling interactions. Motional modes modeled as quantized harmonic oscillators truncated at n_max phonons.

- **Open system dynamics:**  
  Lindblad collapse operators represent spontaneous emission, laser phase/amplitude noise, and motional heating effects. Collapse rates are based on experimentally measured lifetimes and branching ratios.

  If you want to know more specific I recommend checking the \docs\theory.md and \docs\references.md
---

## Features (potentially)

- Single and two-qubit Yb⁺ simulations with motional mode coupling.  
- Configurable laser drives: Raman beams, microwaves, repumps.  
- Noise models for laser amplitude, phase fluctuations, and spontaneous Raman scattering.  
- Efficient numerical solvers leveraging sparse matrices, caching, and parallel parameter sweeps.  
- Visualization tools for population dynamics, coherence, and gate fidelity metrics.

---

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
