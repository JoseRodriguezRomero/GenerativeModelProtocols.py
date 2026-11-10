from juliacall import Main as jl

jl.seval("using GenerativeModelProtocols")
jl.seval("using Optimisers")
jl.seval("using Lux")

from .auxiliary.optimisers import *
from .auxiliary.lux import *
from .auxiliary.devices import *

from .generative_model_protocol import *
from .auxiliary.tabular_denoiser import *

from .variational_autoencoder import *
from .generative_adversarial_network import *
from .gaussian_mixture_model import *
from .normalizing_flow import *
from .diffusion_model import *


def load_differential_equations_extension():
    """
    Loads the `DifferentialEquations.jl` package in Julia.
    """

    jl.seval('using DifferentialEquations')


def load_distribution_distance_extension():
    """
    Loads the distribution distance extension packages in Julia, including 
    `Tulip.jl`, `Distances.jl`, and `OptimalTransport.jl`.
    """
    jl.seval('using Tulip')
    jl.seval('using Distances')
    jl.seval('using OptimalTransport')


def load_save_load_extension():
    """
    Loads the save and load extension packages in Julia, including `FileIO.jl` 
    and `HDF5.jl`.
    """
    jl.seval("using FileIO")
    jl.seval("using HDF5")


__all__ = [
    # constructors
    "VariationalAutoencoder",
    "GenerativeAdversarialNetwork",
    "GaussianMixtureModel",
    "NormalizingFlow",
    "DiffusionModel",
    "TabularDenoiser",
    "GenerativeModelProtocol",

    # other auxiliary
    "TrainingDataNormalization",
    "Device",

    # convenience constructors
    "default_variational_autoencoder", 
    "default_generative_adversarial_network",
    "default_gaussian_mixture_model",
    "default_normalizing_flow",
    "default_diffusion_model",
    "default_tabular_denoiser",

    # dependency loaders
    "load_differential_equations_extension",
    "load_distribution_distance_extension",
    "load_save_load_extension",

    # lux components
    "Dense",
    "Chain",
    "ActivationFunction",
]

