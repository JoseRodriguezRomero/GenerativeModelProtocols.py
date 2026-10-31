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

from .abstract_generative_model import *
from .variational_autoencoder import *
from .generative_adversarial_network import *
from .gaussian_mixture_model import *
from .diffusion_model import *
from .normalizing_flow import *

from .auxiliary.optimisers import *


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
            if isinstance(kwargs['optimiser'], tuple):
                optimisers = kwargs['optimiser']
                _jl_optimiser_1 = optimisers[0]._jl_optimiser
                _jl_optimiser_2 = optimisers[1]._jl_optimiser
                kwargs['optimiser'] = jl.convert(
                    jl.Tuple,
                    (_jl_optimiser_1, _jl_optimiser_2),
                )
            else:
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


def _get_jl_optimiser_struct(_jl_optimiser):
    if jl.isa(_jl_optimiser, jl.Descent):
        return Descent(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.Momentum):
        return Momentum(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.Nesterov):
        return Nesterov(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.Rprop):
        return Rprop(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.RMSProp):
        return RMSProp(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.Adam):
        return Adam(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.RAdam):
        return RAdam(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AdaMax):
        return AdaMax(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.OAdam):
        return OAdam(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AdaGrad):
        return AdaGrad(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AdaDelta):
        return AdaDelta(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AMSGrad):
        return AMSGrad(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.NAdam):
        return NAdam(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AdamW):
        return AdamW(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.AdaBelief):
        return AdaBelief(_jl_optimiser=_jl_optimiser)
    elif jl.isa(_jl_optimiser, jl.Lion):
        return Lion(_jl_optimiser=_jl_optimiser)
    else:
        raise TypeError(f"Unsupported Julia optimiser type: {jl.typeof(_jl_optimiser)}")


def default_optimiser(protocol: GenerativeModelProtocol):
    _jl_optimiser = jl.GenerativeModelProtocols.default_optimiser(protocol._jl_protocol)

    if isinstance(_jl_optimiser, tuple):
        return (_get_jl_optimiser_struct(_jl_optimiser[0]), _get_jl_optimiser_struct(_jl_optimiser[1]))
    else:
        return _get_jl_optimiser_struct(_jl_optimiser)


__all__ = [
    "TrainingDataNormalization",
    "GenerativeModelProtocol",
    "default_optimiser"
]
