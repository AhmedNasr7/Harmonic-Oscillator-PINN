# Harmonic-Oscillator-PINN

A **Conditional** Physics-Informed Neural Network (PINN) that solves the damped harmonic oscillator across a continuous range of damping ratios — learning an entire family of solutions with a single model.

## Problem Statement

The damped harmonic oscillator is governed by:

$$\frac{d^2x}{dz^2} + 2\xi \frac{dx}{dz} + x = 0$$

| Parameter | Description | Value / Range |
|-----------|-------------|---------------|
| $x(z)$ | Displacement as a function of time | — |
| $\xi$ | Damping ratio (conditioning parameter) | $[0.1,\; 0.4]$ |
| $x(0)$ | Initial displacement | $0.7$ |
| $\dot{x}(0)$ | Initial velocity | $1.2$ |
| Domain | Time interval | $z \in [0, 20]$ |

The PINN is **conditioned** on $\xi$, so a single trained model can predict the solution for any damping ratio within the specified range — including values never seen during training.

## Approach

### 1. Conditional Architecture
The network takes $(z, \xi)$ as a 2D input and outputs displacement $x(z; \xi)$. This allows continuous conditioning on the damping ratio rather than training separate models for each value.

**Architecture:** 5 hidden layers × 64 neurons (~16.9K parameters)

### 2. WaveAct Activation
Each hidden layer uses a learnable **WaveAct** activation ([arXiv:2307.11833](https://arxiv.org/abs/2307.11833)):

$$\text{WaveAct}(x) = w_1 \sin(x) + w_2 \cos(x)$$

with independent learnable parameters $(w_1, w_2)$ per layer, naturally matching the oscillatory structure of the solution and overcoming spectral bias.

### 3. Physics-Informed Loss
The loss combines three components:
- **PDE residual:** $d^2x/dz^2 + 2\xi\, dx/dz + x = 0$ at 500 collocation points
- **IC displacement:** $x(0; \xi) = 0.7$
- **IC velocity:** $\dot{x}(0; \xi) = 1.2$

Initial condition terms are weighted by $\lambda_{\text{ic}} = 10$ to enforce boundary constraints.

### 4. Two-Phase Training
| Phase | Optimizer | Epochs | Purpose |
|-------|-----------|--------|---------|
| 1 | Adam (cosine LR: 1e-3 → 1e-5) | 3,000 | Fast landscape navigation |
| 2 | L-BFGS (strong Wolfe) | 500 | Fine-tuning with second-order info |

During each Adam step, 8 random $\xi$ values are sampled uniformly from $[0.1, 0.4]$, giving continuous parameter coverage rather than discrete grid training.

## Key Features

- **Continuous conditioning** — single model covers the full $\xi \in [0.1, 0.4]$ range
- **Generalisation to unseen $\xi$** — evaluated at $\xi = 0.15, 0.25, 0.35$ (never explicitly seen during training)
- **Analytical validation** — predictions compared against the closed-form underdamped solution
- **Phase portraits** — velocity vs. displacement trajectories confirm correct dissipative dynamics (inward spiral)
- **Activation ablation** — WaveAct vs Tanh comparison demonstrating the benefit of learnable sinusoidal activations

## Requirements

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/AhmedNasr7/Harmonic-Oscillator-PINN.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Harmonic_Oscillator_PINN_Conditional.ipynb
   ```

## Results

The trained conditional PINN accurately reproduces the analytical solution across all tested damping ratios. The model generalises to unseen $\xi$ values with comparable accuracy, demonstrating true continuous conditioning rather than memorisation of discrete parameter values.

### Notebook Outputs
- **Training loss curves** — total loss + PDE/IC component breakdown
- **MAE & RMSE over training** — accuracy progression tracked every 200 epochs
- **PINN vs Analytical plots** — overlay comparisons for $\xi = 0.10, 0.20, 0.30, 0.40$
- **Absolute error vs time** — spatial error distribution for each damping ratio
- **Generalisation test** — predictions at unseen $\xi = 0.15, 0.25, 0.35$
- **Phase portraits** — velocity vs displacement for all tested $\xi$ values
- **Activation ablation** — WaveAct vs Tanh convergence comparison

## Connection to GENIE/PINNDE

The conditional architecture here — where a single PINN is conditioned on a physical parameter ($\xi$) — is a direct prototype of what the [PINNDE project](https://ml4sci.org/) requires: a PINN conditioned on the parameters of a target probability density, enabling fast sampling from the reverse-time diffusion equation without retraining for each new density.
