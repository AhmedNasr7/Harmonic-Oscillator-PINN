# Harmonic-Oscillator-PINN
Physics-Informed Neural Network (PINN) for solving the damped harmonic oscillator.

## Problem Statement
The goal is to solve the **damped harmonic oscillator equation** using a PINN. The governing equation of the damped harmonic oscillator is:

\[
$\frac{d^2 z}{dt^2} + 2 \zeta \frac{dz}{dt} + \omega^2 z = 0$
\]

Where:
- \ $z(t)$ \ is the **displacement** of the system as a function of time.
- $\ \zeta  \$ is the **damping ratio**, conditioned within the range [0.1, 0.4].
- $\ $z_0$ \$ is the **initial displacement**.
- $\ $\dot{z}_0$ \$ is the **initial velocity**.

The solution is computed over the domain $\z \in [0, 20] \$, and the PINN is trained to learn the solution conditioned on the damping ratio \( \zeta \).

## Approach
1. **Define the PDE and Initial Conditions:**  
   - The damped harmonic oscillator equation is formulated as a loss function in the PINN.  
   - The initial conditions are incorporated as part of the loss.  

2. **Build the PINN Architecture:**  
   - The neural network is constructed to solve the PDE by learning the **displacement** \( z(t) \) and **velocity** \( \dot{z}(t) \).  
   - The model is conditioned on the **damping ratio** \( \zeta \) between 0.1 and 0.4.  

3. **Training the Model:**  
   - Optimization is performed using **Adam** or **LBFGS** to minimize the **PDE residual loss** and the **initial condition loss**.  
   - The network learns the solution while dynamically adapting to different damping ratios.  

## Requirements
- Python 3.9+  
- PyTorch
- NumPy  
- Matplotlib  

## Usage
1. Clone the repository:  
   ```bash
   git clone https://https://github.com/AhmedNasr7/Harmonic-Oscillator-PINN.git
   ```  
2. Install the required packages:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run the notebook:  
   ```bash
   jupyter notebook Harmonic_Oscillator_PINN.ipynb
   ```  

## Results
The trained model will output the predicted displacement and velocity for various damping ratios within the specified range. The model's predictions can be compared with the analytical solution for validation.  


