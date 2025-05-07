from PerfusionArguments import PerfusionArguments
from perfusion import compute_perfusion,vonmises,isotropic,weibull,watson
import numpy as np

class PerfusionSolver:
    def __init__(self, perfusion_arguments):
        self.inputs = perfusion_arguments
        self.average_length = 60  # Default value
        self.sigma_length = 40  # Default value
        self.length_func = None  # Placeholder for length function
        self.direction_func = None  # Placeholder for direction function
        self.fibreRotation = True  # Default to true
        self.meandir = np.array([0,0,1])  # Placeholder for mean direction
        self.seed = self.inputs.seed









