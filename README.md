# BlueSideYb

**Numerical simulation of Yb⁺ trapped-ion quantum dynamics with gate operations and open-system effects.**

---

## Introduction

Trapped ¹⁷¹Yb⁺ ions are a leading platform for quantum computing due to their well-characterized hyperfine qubits, long coherence times, and accessible optical transitions. This repository provides a numerical framework to simulate the quantum dynamics of Yb⁺ qubits under laser driving, including dissipative effects. The goal is to enable realistic studies of gate fidelities, noise impact, and potentially multi-qubit coupling via motional modes. This one-week-project was created in the scope of a course Scientific Computing for Quantum Information Science at DTU. 

**Some self reflection to begin with:**
The Scope of this project and the bar was already from the beginning pretty high. And I must admit that I failed harder than I anticipated. I started off with many ideas and energy - but I realized soon enough that I began something a little too big for just a weeks worth project. Since I have to hand in something close to presentable I pulled the breaks during the last 3 days and just made the best from what I had. What I said earlier was just the scope and the idea I had in mind before the project. And I have to admit in hindisght... that was probably about to be expected when I had so many ideas to begin with. I learned a lot nevertheless and know that next time I should just start off building something small and then keep going from that. I realized that I wanted to go for the endproduct right away. In the meantime I had many reasoning issues in my code and started again. 

Right now I am quite sure my code can solve a pde but can not create any physical meaning or even simulate physical behavior. I will keep progressing and improving this project when I find the time. But I figured out that I went at it the wrong way at the beginning.

---

## Theoretical Background

- **Qubit encoding:**  
  Qubits are encoded in the ground-state hyperfine manifold of ¹⁷¹Yb⁺.

- **Key transitions:**
  - Raman drives between 2S₁/₂ F=0 and F=1 for single qubit gates.
  - Cooling/detection: 2S₁/₂ ↔ 2P₁/₂ at 369.5 nm  
  - Repump: at 935 nm  

- **Hamiltonian model:**  
  Includes internal energy splitting, laser-driven carrier and sideband drives, and potentially Mølmer–Sørensen entangling interactions. Motional modes modeled as quantized harmonic oscillators truncated at n_max phonons.
  (what I actually have is explained in the notebook, so feel free to check that out - this is just what I had in mind, so also creating a second IonQubit instance and then coupling those two)

- **Open system dynamics:**  
  Lindblad collapse operators represent spontaneous emission, laser phase/amplitude noise, and motional heating effects. Collapse rates are based on experimentally measured lifetimes and branching ratios.
  (Those are present, also kind of working, but also since the rest is not physical it is just something that will be relevant later)

---

## Features (potentially)

- Single and two-qubit Yb⁺ simulations with motional mode coupling.  
- Configurable laser drives: Raman beams, microwaves, repumps.  
- Noise models for laser amplitude, phase fluctuations, and spontaneous Raman scattering.  
- Efficient numerical solvers leveraging sparse matrices, caching, and parallel parameter sweeps.  
- Visualization tools for population dynamics, coherence, and gate fidelity metrics.
  
  **(Actual Features: Qubit instance and Gate initialization as well as a Simulator class with 2 (and a half) kind of solvers. The Hamiltonians in the Qubit class are still wrong but that is what I ended up with.)**
---

## Installation

Requires Python 3.9+.
```bash
pip install -r requirements.txt
