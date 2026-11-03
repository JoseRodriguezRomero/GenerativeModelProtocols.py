from juliacall import Main as jl

jl.seval("using GenerativeModelProtocols")
jl.seval("using Optimisers")
jl.seval("using Lux")

from .auxiliary.optimisers import *
from .auxiliary.lux import *

from .generative_model_protocol import *
from .auxiliary.tabular_denoiser import *

from .variational_autoencoder import *
from .generative_adversarial_network import *
from .gaussian_mixture_model import *
from .normalizing_flow import *
from .diffusion_model import *


def load_differential_equations():
    jl.seval("using DifferentialEquations")


def load_reactant():
    jl.seval("using Reactant")


def load_reactant_cpu():
    load_reactant()
    jl.seval("Reactant.set_backend(\"cpu\")")


def load_reactant_gpu():
    load_reactant()
    jl.seval("Reactant.set_backend(\"gpu\")")


def load_reactant_cuda():
    load_reactant()
    jl.seval("Reactant.set_backend(\"cuda\")")


def load_distribution_distance_extension():
    jl.seval("using Tulip")
    jl.seval("using Distances")
    jl.seval("using OptimalTransport")


def load_save_load_exension():
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

    # convenience constructors
    "default_variational_autoencoder", 
    "default_generative_adversarial_network",
    "default_gaussian_mixture_model",
    "default_normalizing_flow",
    "default_diffusion_model",
    "default_tabular_denoiser",

    # dependency loaders
    "load_differential_equations",
    "load_reactant",
    "load_reactant_cpu",
    "load_reactant_gpu",
    "load_reactant_cuda",
    "load_distribution_distance_extension",
    "load_save_load_exension",

    # lux components
    "Dense",
    "Chain",
    "ActivationFunction",
]

