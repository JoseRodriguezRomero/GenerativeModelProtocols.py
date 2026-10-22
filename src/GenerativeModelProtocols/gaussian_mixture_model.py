import numpy as np
from juliacall import Main as jl
from .auxiliary.lux import Chain
from .generative_model_protocol import AbstractGenerativeModel


class GaussianMixtureModel(AbstractGenerativeModel):
    def __init__(
        self,
        log_variance: np.ndarray = None,
        means: np.ndarray = None,
        predictor_network: Chain = None,
        weights: np.ndarray = None,
        _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            super().__init__(
                jl.GenerativeModelProtocols.GaussianMixtureModel(
                    **{
                        "log_σ²": jl.Array(log_variance),
                        "μ": jl.Array(means),
                        "predictor_network": predictor_network._jl_chain,
                        "p": jl.Vector(weights),
                    }
                )
            )


def default_gaussian_mixture_model(dimensions: int, k: int):
    _jl_model = jl.seval(f"GenerativeModelProtocols.GaussianMixtureModel({dimensions}, {k})")
    return GaussianMixtureModel(_jl_model = _jl_model)

