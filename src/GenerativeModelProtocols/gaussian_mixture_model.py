import numpy as np
from juliacall import Main as jl
from .auxiliary.lux import Chain
from .abstract_generative_model import AbstractGenerativeModel


class GaussianMixtureModel(AbstractGenerativeModel):
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_model",)

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
            self._jl_model =  jl.GenerativeModelProtocols.GaussianMixtureModel(
                **{
                    "log_σ²": jl.Matrix(log_variance),
                    "μ": jl.Matrix(means),
                    "predictor_network": predictor_network._jl_chain,
                    "p": jl.Vector(weights),
                }
            )

    @property
    def log_variance(self) -> np.ndarray:
        return np.asarray(getattr(self._jl_model, "log_σ²"))

    @property
    def means(self) -> np.ndarray:
        return np.asarray(getattr(self._jl_model, "μ"))

    @property
    def predictor_network(self) -> Chain:
        return Chain(_jl_chain = self._jl_model.predictor_network)

    @property
    def weights(self) -> np.ndarray:
        return np.asarray(self._jl_model.p)


def default_gaussian_mixture_model(dimensions: int, k: int):
    _jl_model = jl.seval(f"GenerativeModelProtocols.GaussianMixtureModel({dimensions}, {k})")
    return GaussianMixtureModel(_jl_model = _jl_model)

