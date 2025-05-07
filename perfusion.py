import numpy as np
from numpy.linalg import norm
from scipy.special import gamma
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from numpy import trapz
from concurrent.futures import ProcessPoolExecutor


def compute_perfusion(inputs, execution_mode, n_cores=None):
    # Inputs
    num_streams = inputs.n_particles
    seed = inputs.seed
    pos = inputs.init_pos
    gG = inputs.gradient['gG']
    dt = inputs.gradient['dt']
    T = inputs.T
    velocity_values = inputs.velocity_values
    velocity_func = inputs.velocity_func

    # Output initialization
    phase = np.zeros(pos.shape)
    final_pos = np.zeros(pos.shape)
    rng_master = np.random.default_rng(seed)
    rngs = rng_master.bit_generator._seed_seq.spawn(num_streams)
    
    if execution_mode == "parallel":
        # Parallel execution using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = executor.map(process_particle_wrapper, range(num_streams), [inputs]*num_streams, [pos]*num_streams, [T]*num_streams, [velocity_values]*num_streams, [velocity_func]*num_streams, [dt]*num_streams, [gG]*num_streams, rngs)
                   
        for i, (phase_val, final_position) in enumerate(results):
            phase[i] = phase_val
            final_pos[i] = final_position
    else:
        # Serial execution
        for i in range(inputs.n_particles):
            rng = np.random.default_rng(rngs[i])  # Independent RNG per particle
            phase_val, final_position = process_particle(i, inputs, rng, np.copy(pos), T, velocity_values, velocity_func, dt, gG)
            phase[i] = phase_val
            final_pos[i] = final_position
    return phase, final_pos

def process_particle_wrapper(i, obj, pos, T, velocity_values, velocity_func, dt, gG, rng_state):
    # Create an independent RNG using the passed state
    rng = np.random.default_rng(rng_state)
    return process_particle(i, obj, rng, np.copy(pos), T, velocity_values, velocity_func, dt, gG)

def process_particle(i, inputs, rng, pos, T, velocity_values, velocity_func, dt, gG):
    pos_particle = pos[i]
    velocity = velocity_values[i]
    # Call the phase_one_particle function for this particle
    phase_val, final_position = phase_one_particle(inputs, rng, pos_particle, T, velocity, velocity_func, dt, gG)
    return phase_val, final_position

def phase_one_particle(inputs, rng, pos_particle, T, velocity, v_func, dt, gG):
    phase = np.zeros_like(pos_particle)
    t_init = 0
    t_end = 0
    phase = np.zeros((1,3))
    pos = pos_particle
    # Assuming that "setCapillaryDistributions" sets capillary directions and lengths

    while t_end < T:
        
        #get capillary segment direction and length
        default_azimuth = 0.
        
        if inputs.fiberRotation:
            dy = 392.04  # Realistic slice width
            y_minvals = np.arange(0, 8000 + dy, dy)  # y slices
            y_minvals = np.unique(np.concatenate((y_minvals, -y_minvals)))  # Positive and negative slices
            y_slice_minmax = np.array([y_minvals[:-1], y_minvals[1:]])  # Slice min/max pairs
            
            # Find the correct y_slice for the particle's current y position
            y_slice = find_yslice(pos_particle[1], y_slice_minmax)
            
            # Compute rotation angle based on y_slice
            if abs(y_slice) < dy / 2:
                angle = np.pi / 2  # Set to pi/2 for slice [0, dy]
            else:
                angle = np.pi / 2 - np.deg2rad(0.01) * y_slice

            # Convert spherical to Cartesian using the rotation angle
            default_azimuth = 0  # Set the azimuth value if needed
            dir1, dir2, dir3 = sph2cart(np.deg2rad(default_azimuth), angle)
        else:
            dir1, dir2, dir3 = inputs.meandir
        
        capillary_direction = computeCapillaryOrientation(inputs.capillaryDistribution,rng,inputs.kvalue, meandir=[dir1,dir2,dir3])
        capillary_length = computeCapillaryLength(inputs.lengthDistribution,rng)
        
        if t_init == 0:
                #get a random value from stream to determine the length of the capillary segment
            x= rng.uniform()
            capillary_length = x*capillary_length
            
        t_end = compute_time_in_capillary(velocity, v_func, capillary_length, t_init,T)
        if t_end>T:
            t_end = 1
        
        delta_pos = capillary_length * capillary_direction
        pos = pos + delta_pos
        phase = phase + compute_phase(t_init/T, t_end/T, T, dt, gG, v_func, capillary_direction)
        t_init = t_end
        
    return phase, pos


def vonmises(stream, mean_direction, k):
    # Convert mean_direction to spherical coordinates
    phi_0, theta_0 = cart2sph(mean_direction)
    theta_0 = np.pi / 2 - theta_0
    
    R_z = np.array([[np.cos(-phi_0), -np.sin(-phi_0), 0],
                    [np.sin(-phi_0), np.cos(-phi_0), 0],
                    [0, 0, 1]])
    R_y = np.array([[np.cos(-theta_0), 0, np.sin(-theta_0)],
                    [0, 1, 0],
                    [-np.sin(-theta_0), 0, np.cos(-theta_0)]])
    R = R_y @ R_z

    theta = np.linspace(-np.pi, np.pi, 10000)
    p_theta = (k) / (4 * np.pi * np.sinh(k)) * np.exp(k * np.cos(theta))
    p_theta /= np.trapz(p_theta, theta)

    cfd_theta = cumulative_trapezoid(p_theta, theta, initial=0)
    zenith = interp1d(cfd_theta, theta)(stream.uniform())

    azimuth = stream.uniform() * 2 * np.pi
    xp, yp, zp = sph2cart(zenith, azimuth)

    direction = np.linalg.inv(R) @ np.array([xp, yp, zp])
    return direction / norm(direction)


def watson(stream, mean_direction, k):
    """
    Watson distribution for sampling directions.
    
    :param stream: Random number generator (numpy Generator).
    :param mean_direction: Mean direction in 3D (array-like).
    :param k: Concentration parameter (positive for concentration along the axis).
    :return: A direction vector sampled from the Watson distribution.
    """
    
    # Convert mean_direction to spherical coordinates
    phi_0, theta_0 = cart2sph(mean_direction)
    theta_0 = np.pi / 2 - theta_0

    # Rotation matrices R_z and R_y
    R_z = np.array([
        [np.cos(-phi_0), -np.sin(-phi_0), 0],
        [np.sin(-phi_0), np.cos(-phi_0), 0],
        [0, 0, 1]
    ])
    R_y = np.array([
        [np.cos(-theta_0), 0, np.sin(-theta_0)],
        [0, 1, 0],
        [-np.sin(-theta_0), 0, np.cos(-theta_0)]
    ])
    R = R_y @ R_z  # Combined rotation matrix

    # Set up theta and x for integration
    theta = np.linspace(0, np.pi / 2, 10000)
    x = np.linspace(0, 1, 10000)

    # u_func is the integral of exp(2 * k * x^2)
    u_func = np.trapz(np.exp(2 * k * x**2), x)

    # p_theta is the probability density function for theta
    p_theta = (u_func)**-1 * np.sin(theta) * np.exp(2 * k * np.cos(theta)**2)
    p_theta = p_theta / np.trapz(p_theta, theta)  # Normalize p_theta

    # Cumulative distribution function for theta
    cfd_theta = cumulative_trapezoid(p_theta, theta, initial=0)
    
    # Remove duplicates and keep unique values
    cfd_theta, unique_idx = np.unique(cfd_theta, return_index=True)
    
    # Inverse transform sampling to get zenith angle
    zenith = interp1d(cfd_theta, theta[unique_idx])(stream.uniform())

    # Uniform azimuth sampling
    azimuth = stream.uniform() * 2 * np.pi

    # Cartesian coordinates from spherical coordinates
    xp = np.sin(zenith) * np.cos(azimuth)
    yp = np.sin(zenith) * np.sin(azimuth)
    zp = np.cos(zenith)

    # Direction in cartesian coordinates
    direction = np.linalg.inv(R) @ np.array([xp, yp, zp])
    direction = direction / norm(direction)  # Normalize the direction

    return direction


def computeCapillaryOrientation(type, stream, k_value, meandir=[0,0,1]):
    """Compute capillary orientation based on the distribution."""
    if type == "VonMises":
        return vonmises(stream, meandir, k_value)
    elif type == "Isotropic":
        return isotropic(stream)
    elif type == "Watson":
        return watson(stream, meandir, k_value)
    else:
        raise ValueError("Unknown distribution type for capillary orientation.")
    
def computeCapillaryLength(type, stream): 
    
    """Compute capillary length based on the distribution."""
    if type == "Weibull":
        return weibull(stream, 60, 40)
    else:
        raise ValueError("Unknown distribution type for capillary length.")
        
def isotropic(stream):
    direction = stream.standard_normal(size=3)
    return direction / norm(direction)


def weibull(stream, average, sigma):
    x = np.linspace(0, 600, 10000)
    k = (sigma / average) ** -1.086
    c = average / (gamma(1 + 1 / k))
    cfd = 1 - np.exp(-(x / c) ** k)

    rand_value = stream.uniform()
    L = interp1d(cfd, x)(rand_value * 0.999)
    return L

# Utility functions for spherical <-> cartesian conversions
def cart2sph(cart):
    x, y, z = cart
    hxy = np.hypot(x, y)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, hxy)
    return azimuth, elevation

def sph2cart(zenith, azimuth):
    xp = np.sin(zenith) * np.cos(azimuth)
    yp = np.sin(zenith) * np.sin(azimuth)
    zp = np.cos(zenith)
    return xp, yp, zp

def find_yslice(pos_y, y_slice_minmax):
    """
    Find the y_slice for a given y position.
    
    :param pos_y: The y position of the particle.
    :param y_slice_minmax: The min/max values for each y slice.
    :return: The y_slice corresponding to the y position.
    """
    # Find the slice where pos_y is between min and max of each slice
    i_slice = np.where((pos_y >= y_slice_minmax[0, :]) & (pos_y < y_slice_minmax[1, :]))[0]
    
    if len(i_slice) == 0:
        print(f"Error: y position {pos_y} not found in any slice.")
        raise ValueError("Corresponding slice not found")
    
    # Return the corresponding y_slice
    return y_slice_minmax[0, i_slice[0]]

def compute_phase(s1, s2, T, dt, gG, v, velocity_dir):
    
    cum_integral = cumulative_trapezoid(x=gG, y=dt,initial=0)
    
    #now we declare the space in which we are integrating
    dt_interval = np.linspace(s1,s2,10000)
    interpolated_cum_integral = interp1d(x=dt,y=cum_integral)(dt_interval)
    dt_interval_T = dt_interval * T
    velocity_func = v(dt_interval_T)
    
    factor = interpolated_cum_integral * velocity_func
    
    integral = np.trapz(x=dt_interval, y=factor)
    
    phase = np.zeros((1,3)) 
    for i in range(3):
        phase[0,i] = velocity_dir[i] * integral
    return phase

    
def compute_time_in_capillary(velocity, velocity_func, capillary_length, t_init,T):
    """
    Compute the time spent by a particle in a capillary segment.
    
    :param velocity: The velocity of the particle.
    :param velocity_func: The velocity function.
    :param capillary_length: The length of the capillary segment.
    :param t_init: The initial time.
    :return: The time spent by the particle in the capillary segment.
    """
    # Compute the time spent in the capillary segment
    time = np.linspace(t_init,T,1000)
    velocity_func = velocity_func(time) * velocity
    integral = cumulative_trapezoid(x=time, y=velocity_func,initial=0)
    #we need to find at which time the integral is equal to the capillary length
    integral = np.abs(integral - capillary_length)
    t_end = time[np.argmin(integral)]
    return t_end
    
    
        