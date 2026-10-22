from juliacall import Main as jl
from .generative_model_protocol import AbstractGenerativeModel

def default_diffusion_model(dimensions : int, T : int = 5, beta_start : float = 1.0E-4, beta_end : float = 0.02):
    _jl_model = jl.seval(f"GenerativeModelProtocols.DiffusionModel({dimensions}, {T}, {beta_start}, {beta_end})")
    return AbstractGenerativeModel(_jl_model)

