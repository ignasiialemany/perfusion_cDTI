import yaml
import numpy as np
import matplotlib.pyplot as plt

# Function to load and combine multiple YAML config files
def create_seq(config_files):
    # Load the YAML config files
    configs = [yaml.safe_load(open(file)) for file in config_files]
    
    # Combine all the loaded configs into one
    config = combine_dicts(*configs)

    # Extract the sequence type
    seq_type = config['sequence']['type']
    gamma = config["sequence"]["gamma"]

    # Select the sequence data based on the type
    if seq_type in config['sequence']:
        seq_data = config['sequence'][seq_type]
        seq_data['gamma'] = config['sequence']['gamma']
    else:
        raise ValueError(f"Sequence type {seq_type} not found in the YAML file")

    # Extract parameters for creating the sequence
    seq_Nt = config['sequence']['N_t']
    seq_dtmax = np.array(config['sequence']['dt_max'])  # [dt_free, dt_grad]

    # Create the MRI sequence using the extracted parameters
    sequence = create(seq_Nt, seq_dtmax, seq_type, **seq_data)

    # Normalize dt and gG
    dt = sequence['dt']
    gG = sequence['gG']
        
    dt = np.array(np.concatenate([[0], np.cumsum(dt)]))
    gG = np.array(np.concatenate([[0], gG]))
    g_strength = np.max(gG)
    T = np.max(dt)
    
    dt = dt/T
    gG = gG/g_strength
    
    return dt, gG, g_strength, T

# Function to create the MRI sequence object
def create(NT, dt_max, SeqName, **kwargs):
    # Convert the SeqName to uppercase for consistent handling
    SeqName = SeqName.upper()

    if SeqName in ['PGSE', 'STEAM', 'STEAM_BVALUE600', 'STEAM_BVALUE150', 'PGSE_BVALUE150', 'PGSE_BVALUE600']:
        Gmax, alpha90, alphaRO, epsilon, Delta, delta, gamma = parse(kwargs, 
            ['Gmax', 'alpha90', 'alphaRO', 'epsilon', 'Delta', 'delta', 'gamma'])
        durations = [alpha90, epsilon, delta, epsilon, Delta - (2 * epsilon + delta),
                     epsilon, delta, epsilon, alphaRO]
        ids = [0, 1, 2, 3, 0, -1, -2, -3, 0]
        dt, gG = discretize(durations, ids, NT, dt_max, Gmax)

    elif SeqName in ['MCSE', 'M2SE', 'MCSE_BVALUE600', 'MCSE_BVALUE150']:
        Gmax, alpha90, alphaRO, epsilon, delta1, delta2, gamma = parse(kwargs, 
            ['Gmax', 'alpha90', 'alphaRO', 'epsilon', 'delta1', 'delta2', 'gamma'])
        del1 = delta1 + 2 * epsilon
        del2 = delta2 + 2 * epsilon
        Delta = (del2 * (-2 * del1 + epsilon) + del1 * epsilon) / (del1 - del2)
        durations = [alpha90, epsilon, delta1, epsilon, epsilon, delta2, epsilon, 
                     Delta - (del1 + del2), epsilon, delta2, epsilon, epsilon, delta1, epsilon, alphaRO]
        ids = [0, 1, 2, 3, -1, -2, -3, 0, 1, 2, 3, -1, -2, -3, 0]
        dt, gG = discretize(durations, ids, NT, dt_max, Gmax)

    else:
        # Dummy sequence
        gamma = 1
        dt = np.ones(NT) * dt_max[0]
        gG = np.zeros(NT)

    # Return the sequence as a dictionary
    return {
        'dt': dt,
        'gG': gG
    }

# Helper function to parse arguments or extract values from a dictionary
def parse(args, names):
    return [args.get(name) for name in names]

# Discretize the sequence into dt and gG
def discretize(durations, ids, NT, dt_max, Gmax):
    # Calculate target time step
    dt_aim = sum(durations) / NT

    # Limit the dt values
    dt_free = min(dt_aim, dt_max[0])
    dt_grad = min(dt_aim, dt_max[1])

    # Calculate the number of steps per interval, based on dt_free or dt_grad
    Nt_intervals = np.zeros_like(durations)
    Nt_intervals[ids == 0] = np.ceil(durations[ids == 0] / dt_free).astype(int)
    Nt_intervals[ids != 0] = np.ceil(durations[ids != 0] / dt_grad).astype(int)

    dt = []
    gG = []

    for i in range(len(durations)):
        Nt_i = int(Nt_intervals[i])
        dt_i = durations[i] / Nt_i
        dt.extend([dt_i] * int(Nt_i))

        id_i = ids[i]
        if abs(id_i) == 0:  # Flat gradient off
            gA, gB = 0, 0
        elif abs(id_i) == 1:  # Ramp-up gradient
            gA, gB = 0, np.sign(id_i) * Gmax
        elif abs(id_i) == 2:  # Flat gradient on
            gA, gB = np.sign(id_i) * Gmax, np.sign(id_i) * Gmax
        elif abs(id_i) == 3:  # Ramp-down gradient
            gA, gB = np.sign(id_i) * Gmax, 0
        
        # Create gradient values for this interval
        gvals = np.linspace(gA, gB, int(Nt_i) + 1)
        gG.extend(gvals[:-1])  # Exclude the last value to match MATLAB's behavior
        
    return np.array(dt), np.array(gG)

# Recursive function to combine multiple dictionaries (like combine_structs in MATLAB)
def combine_dicts(*dicts):
    combined = {}
    for d in dicts:
        for key, value in d.items():
            if isinstance(value, dict):
                combined[key] = combine_dicts(combined.get(key, {}), value)
            else:
                combined[key] = value
    return combined

# Example usage

