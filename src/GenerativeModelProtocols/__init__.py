from juliacall import Main as jl

jl.seval("using GenerativeModelProtocols")
jl.seval("using Optimisers")
jl.seval("using Lux")

from .generative_model_protocol import AbstractGenerativeModel
from .variational_autoencoder import default_variational_autoencoder
from .generative_adversarial_network import default_generative_adversarial_network
from .gaussian_mixture_model import default_gaussian_mixture_model
from .normalizing_flow import default_normalizing_flow
from .diffusion_model import default_diffusion_model

from .variational_autoencoder import VariationalAutoencoder
from .normalizing_flow import NormalizingFlow

from auxiliary.lux import Dense, Chain, ActivationFunction


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


__all__ = [
    "AbstractGenerativeModel",

    # constructors
    "VariationalAutoencoder",
    "NormalizingFlow",

    # convenience constructors
    "default_variational_autoencoder", 
    "default_generative_adversarial_network",
    "default_gaussian_mixture_model",
    "default_normalizing_flow",
    "default_diffusion_model",

    # dependency loaders
    "load_differential_equations",
    "load_reactant",
    "load_reactant_cpu",
    "load_reactant_gpu",
    "load_reactant_cuda",
    "load_distribution_distance_extension",

    # lux components
    "Dense",
    "Chain",
    "ActivationFunction"
]

