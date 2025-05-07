import numpy as np
from scipy.integrate import cumulative_trapezoid

class PerfusionArguments:
    def __init__(self, n_particles=1000, init_pos=None, velocity_func=None, velocity_values=None, gradient=None, T=1, G=1, seed=1):
        self.n_particles = self.validate_n_particles(n_particles)
        self.init_pos = self.validate_init_pos(init_pos)
        self.velocity_func = velocity_func
        self.velocity_values = velocity_values
        self.T = T
        self.G = G
        self.seed = seed
        self.gradient = self.set_gradient(gradient)
        self.fiberRotation = True
        self.lengthDistribution  = "Weibull"
        self.capillaryDistribution = "Watson"
        self.kvalue = 3.25

    def validate_n_particles(self, n_particles):
        if isinstance(n_particles, int) and n_particles > 0:
            return n_particles
        else:
            raise ValueError("n_particles must be a positive integer")

    def validate_init_pos(self, init_pos):
        if isinstance(init_pos, np.ndarray) and init_pos.ndim == 2:
            return init_pos
        else:
            raise ValueError("init_pos must be a 2D numpy array")

    def set_gradient(self, gradient):
        if gradient is not None:
            gradient['gG'] = gradient['gG'] / np.max(gradient['gG'])  # Normalize gradient gG
            gradient['dt'] = gradient['dt'] / np.max(gradient['dt'])  # Normalize gradient dt
            dt = gradient['dt']
            gG = gradient['gG']
            integral = cumulative_trapezoid(x=dt, y= gG,initial=0)**2
            integral2 = np.trapz(x=dt, y=integral)
            sqrt_integral = np.sqrt(integral2)
            gradient['a'] = sqrt_integral  # Compute a
            return gradient
        else:
            return {'gG': None, 'dt': None, 'a': None}
