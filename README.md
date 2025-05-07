# Perfusion Simulation

A Python-based implementation for simulating perfusion phenomena in tissues using particle methods. This simulation tracks particle movement through capillary networks to model perfusion processes.

## Overview

This project provides a framework for simulating particle transport through capillary networks, which is useful for studying perfusion in various tissues. The simulation uses statistical distributions to model capillary orientations and lengths, and can run in both serial and parallel execution modes.

## Features

- Monte Carlo simulation of perfusion with customizable parameters
- Parallel processing support for faster computation
- Configurable capillary distributions (Watson, VonMises, Isotropic)
- Customizable velocity profiles
- Gradient sequence configurations via YAML files

## Requirements

- Python 3.7+
- NumPy
- SciPy
- PyYAML
- Matplotlib (for visualization)

## Installation

```
git clone https://github.com/yourusername/perfusion_python.git
cd perfusion_python
pip install -r requirements.txt
```

## Usage

### Basic simulation

```python
from perfusion import compute_perfusion
from PerfusionArguments import PerfusionArguments
from create import create_seq

# Load sequence configuration
config_files = ['sequence.yml']
dt, gG, g_strength, T = create_seq(config_files)

# Define velocity function
def velocity_func(time):
    return 1 + 0 * time

# Set up simulation parameters
N_spins = 10000
init_pos = # your initial positions array
velocity = # your velocity array

# Create perfusion arguments
gradient_data = {'gG': gG, 'dt': dt}
perfusion_args = PerfusionArguments(
    n_particles=N_spins,
    init_pos=init_pos,
    velocity_func=velocity_func,
    velocity_values=velocity,
    gradient=gradient_data,
    T=T,
    G=g_strength,
    seed=1
)

# Run simulation
phase, position = compute_perfusion(perfusion_args, "parallel", n_cores=8)
```

## Core Components

- `perfusion.py`: Main simulation engine that computes particle trajectories and phases
- `PerfusionArguments.py`: Class for managing simulation parameters
- `create.py`: Utilities for creating and managing gradient sequences
- `run_simulation.py`: Example script demonstrating a complete simulation

## Customization

### Capillary Orientation Distributions

The simulation supports multiple distribution types for capillary orientations:
- Watson distribution
- VonMises distribution
- Isotropic distribution

### Capillary Length Distribution

Currently supports Weibull distribution for capillary lengths.

## License

[Insert your license information here]

## Citation

If you use this code in your research, please cite:

[Your citation information]

## Contact

[Your contact information] 
