import numpy as np
from juliacall import Main as jl
from .auxiliary.capture_julia_display import capture_julia_display

from .abstract_generative_model import AbstractGenerativeModel
from .variational_autoencoder import VariationalAutoencoder
from .generative_adversarial_network import GenerativeAdversarialNetwork
from .gaussian_mixture_model import GaussianMixtureModel
from .diffusion_model import DiffusionModel
from .normalizing_flow import NormalizingFlow


class GenerativeModelProtocol:
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_protocol",)

    def __init__(
            self, 
            model: AbstractGenerativeModel, 
            training_data: np.ndarray = np.empty((0, 0)), 
            normalize_data: bool = True
        ):
        self._jl_protocol = jl.GenerativeModelProtocol(
            model._jl_model,
            jl.Matrix(training_data),
            normalize_data = normalize_data
        )

    def __setattr__(self, name, value):
        if name == "_jl_model" and hasattr(self, "_jl_model"):
            raise AttributeError("_jl_model is read-only")
        super().__setattr__(name, value)

    @property
    def model(self) -> AbstractGenerativeModel:
        _jl_model = self._jl_protocol.model
        if jl.isa(_jl_model, jl.GenerativeModelProtocols.VariationalAutoencoder):
            return VariationalAutoencoder(_jl_model = _jl_model)
        elif jl.isa(_jl_model, jl.GenerativeModelProtocols.GenerativeAdversarialNetwork):
            return GenerativeAdversarialNetwork(_jl_model = _jl_model)
        elif jl.isa(_jl_model, jl.GenerativeModelProtocols.GaussianMixtureModel):
            return GaussianMixtureModel(_jl_model = _jl_model)
        elif jl.isa(_jl_model, jl.GenerativeModelProtocols.DiffusionModel):
            return DiffusionModel(_jl_model = _jl_model)
        elif jl.isa(_jl_model, jl.GenerativeModelProtocols.NormalizingFlow):
            return NormalizingFlow(_jl_model = _jl_model)
        else:
            raise TypeError("_jl_model is not an AbstractGenerativeModel")

    @property
    def input_size(self) -> int:
        return jl.input_size(self._jl_protocol)

    @property
    def latent_size(self) -> int:
        return jl.latent_size(self._jl_protocol)

    @property
    def normalize_data(self) -> bool:
        return self._jl_protocol.normalize_data

    @property
    def training_data(self) -> np.ndarray:
        return np.asarray(self._jl_protocol.training_data)

    def train(self, **kwargs):
        key_map = {
            "beta": "β",
            "gamma_vae": "γ_vae",
            "gamma_wgan": "γ_wgan",
            "lambda_": "λ",
        }

        clean_kwargs = {
            key_map.get(k, k): v
            for k, v in kwargs.items()
            if v is not None
        }

        return jl.seval("train!")(
            self._jl_protocol,
            **{str(k): v for k, v in clean_kwargs.items()},
        )

    def __call__(self, n_samples: int = 1, **kwargs) -> np.ndarray:
        return np.asarray(self._jl_protocol(
            n_samples,
            **{str(k): v for k, v in kwargs.items()},
        ))

    def encode(self, x: np.ndarray, **kwargs) -> np.ndarray:
        return np.asarray(jl.encode(
            self._jl_protocol,
            jl.Matrix(x),
            **{str(k): v for k, v in kwargs.items()},
        ))

    def decode(self, z: np.ndarray, **kwargs) -> np.ndarray:
        return np.asarray(jl.decode(
            self._jl_protocol,
            jl.Matrix(z),
            **{str(k): v for k, v in kwargs.items()},
        ))

    def __repr__(self):
        return capture_julia_display(self._jl_protocol)

    def __str__(self):
        return self.__repr__()

