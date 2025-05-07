import numpy as np
from scipy.stats import truncnorm
from perfusion import compute_perfusion
from PerfusionArguments import PerfusionArguments
from create import create_seq

# Define velocity_func as a regular function instead of a lambda
def velocity_func(time):
    return 1 + 0 * time  # Same as the lambda you were using

if __name__ == "__main__":
    config_files = ['sequence.yml']  # List of YAML files
    dt, gG, g_strength, T = create_seq(config_files)

    # Step 2: Initialize Simulation Parameters
    N_spins = 10000  # Number of spins (particles)
    seed = 1  # Random seed
    np.random.seed(seed)

    # Initialize particle positions (uniform distribution)
    x = np.random.uniform(0, 2800, N_spins)
    y = np.linspace(0, 2800, N_spins)
    z = np.zeros(N_spins)
    init_pos = np.column_stack((x, y, z))

    # Set up the velocity distribution (truncated normal)
    vel = 0.5  # Mean velocity
    sigma = 0.15  # Standard deviation
    a, b = 0, 1  # Truncation limits
    rng = np.random.default_rng(seed)
    trunc_dist = truncnorm((a - vel) / sigma, (b - vel) / sigma, loc=vel, scale=sigma)
    velocity = trunc_dist.rvs(N_spins, random_state=rng)
    
    fv = 1 / T * np.trapz(x=np.linspace(0, T, 1000), y=velocity_func(np.linspace(0, T, 1000)))
    assert fv == 1, "The velocity function should be normalized."

    gradient_data = {'gG': gG, 'dt': dt}
    perfusion_args = PerfusionArguments(
        n_particles=N_spins,
        init_pos=init_pos,
        velocity_func=velocity_func,  # Pass regular function
        velocity_values=velocity,
        gradient=gradient_data,
        T=T,
        G=g_strength,
        seed=seed
    )

    # Now call the compute_perfusion in parallel mode
    phase, position = compute_perfusion(perfusion_args, "parallel",n_cores=8)
    print(position)
