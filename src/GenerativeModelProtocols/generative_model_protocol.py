import os
import pickle
import subprocess
import sys
import tempfile
from threading import Lock
import numpy as np
from enum import Enum
from juliacall import Main as jl
from .auxiliary.capture_julia_display import capture_julia_display

from .abstract_generative_model import AbstractGenerativeModel
from .variational_autoencoder import VariationalAutoencoder
from .generative_adversarial_network import GenerativeAdversarialNetwork
from .gaussian_mixture_model import GaussianMixtureModel
from .diffusion_model import DiffusionModel
from .normalizing_flow import NormalizingFlow


_TRAIN_LOCK = Lock()

_TRAIN_WORKER = r'''
import pickle
import sys
from juliacall import Main as jl
import GenerativeModelProtocols

jl.seval("using Serialization")
read_serialized = jl.seval("path -> open(Serialization.deserialize, path)")
write_serialized = jl.seval(
    '(path, value) -> open(io -> Serialization.serialize(io, value), path, "w")'
)

with open(sys.argv[1], "rb") as stream:
    kwargs = pickle.load(stream)
protocol = read_serialized(sys.argv[2])
train_log = jl.seval("train!")(protocol, **kwargs)
write_serialized(sys.argv[3], (protocol, train_log))
'''


def _julia_serialization_functions():
    jl.seval("using Serialization")
    read_serialized = jl.seval("path -> open(Serialization.deserialize, path)")
    write_serialized = jl.seval(
        '(path, value) -> open(io -> Serialization.serialize(io, value), path, "w")'
    )
    return read_serialized, write_serialized


class TrainingDataNormalization(Enum):
    FULL_NORMALIZATION   = jl.GenerativeModelProtocols.full_normalization
    NO_NORMALIZATION     = jl.GenerativeModelProtocols.no_normalization
    HYBRID_NORMALIZATION = jl.GenerativeModelProtocols.hybrid_normalization


class GenerativeModelProtocol:
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_protocol",)

    def __init__(
            self, 
            model: AbstractGenerativeModel, 
            training_data: np.ndarray = np.zeros((0, 0)),
            normalize_data = TrainingDataNormalization.HYBRID_NORMALIZATION
        ):
        if isinstance(normalize_data, TrainingDataNormalization):
            normalize_data = normalize_data.value
        self._jl_protocol = jl.GenerativeModelProtocol(
            model._jl_model,
            jl.Matrix(training_data),
            normalize_data = normalize_data
        )

    def __setattr__(self, name, value):
        if name == "_jl_protocol" and hasattr(self, "_jl_protocol"):
            raise AttributeError("_jl_protocol is read-only")
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
    def normalize_data(self) -> tuple[bool, ...]:
        return tuple(self._jl_protocol.normalize_data)

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

        if 'optimiser' in kwargs:
            kwargs['optimiser'] = kwargs['optimiser']._jl_optimiser

        clean_kwargs = {
            key_map.get(k, k): v
            for k, v in kwargs.items()
            if v is not None
        }

        with _TRAIN_LOCK, tempfile.TemporaryDirectory(prefix="gmp-train-") as temp_dir:
            kwargs_path = os.path.join(temp_dir, "kwargs.pkl")
            protocol_path = os.path.join(temp_dir, "protocol.jls")
            result_path = os.path.join(temp_dir, "result.jls")

            with open(kwargs_path, "wb") as stream:
                pickle.dump(
                    {str(k): v for k, v in clean_kwargs.items()},
                    stream,
                    protocol = pickle.HIGHEST_PROTOCOL,
                )

            read_serialized, write_serialized = _julia_serialization_functions()
            write_serialized(protocol_path, self._jl_protocol)

            result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "juliacall-handle-signals=yes",
                    "-c",
                    _TRAIN_WORKER,
                    kwargs_path,
                    protocol_path,
                    result_path,
                ],
                check = False,
            )
            if result.returncode != 0:
                if result.returncode < 0:
                    detail = f"terminated by signal {-result.returncode}"
                else:
                    detail = f"exited with status {result.returncode}"
                raise RuntimeError(f"Julia training worker {detail}")

            trained_protocol, train_log = read_serialized(result_path)
            object.__setattr__(self, "_jl_protocol", trained_protocol)
            return train_log

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


__all__ = [
    "TrainingDataNormalization",
    "GenerativeModelProtocol"
]