from juliacall import Main as jl
from .generative_model_protocol import AbstractGenerativeModel

def default_gaussian_mixture_model(dimensions : int, k : int):
    _jl_model = jl.seval(f"GenerativeModelProtocols.GaussianMixtureModel({dimensions}, {k})")
    return AbstractGenerativeModel(_jl_model)

