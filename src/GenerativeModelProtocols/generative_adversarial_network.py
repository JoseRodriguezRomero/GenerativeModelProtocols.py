from juliacall import Main as jl
from .generative_model_protocol import AbstractGenerativeModel

def default_generative_adversarial_network(dimensions : int, latent_dimension : int = 1, latent_layers : int = 1):
    _jl_model = jl.seval(f"GenerativeModelProtocols.GenerativeAdversarialNetwork({dimensions}, {latent_dimension}, {latent_layers})")
    return AbstractGenerativeModel(_jl_model)

