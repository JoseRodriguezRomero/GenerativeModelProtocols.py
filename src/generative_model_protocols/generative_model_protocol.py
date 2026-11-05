import os
import pickle
import subprocess
import sys
import tempfile
from threading import Lock
import numpy as np
from enum import Enum
from juliacall import Main as jl

from .abstract_generative_model import *
from .variational_autoencoder import *
from .generative_adversarial_network import *
from .gaussian_mixture_model import *
from .diffusion_model import *
from .normalizing_flow import *

from .auxiliary.optimisers import *

from .auxiliary.capture_julia_display import *


_TRAIN_LOCK = Lock()

_TRAIN_WORKER = r'''
import pickle
import sys
from juliacall import Main as jl
import generative_model_protocols

jl.seval("using Serialization")
read_serialized = jl.seval("path -> open(Serialization.deserialize, path)")
write_serialized = jl.seval(
    '(path, value) -> open(io -> Serialization.serialize(io, value), path, "w")'
)

with open(sys.argv[1], "rb") as stream:
    kwargs = pickle.load(stream)

device = kwargs.get("device")
if device is not None:
    if "reactant" in device:
        jl.seval("""
            using Logging

            struct JAXWarningFilter <: Logging.AbstractLogger
                logger::Logging.AbstractLogger
            end

            Logging.min_enabled_level(l::JAXWarningFilter) = Logging.min_enabled_level(l.logger)
            Logging.shouldlog(l::JAXWarningFilter, args...) = Logging.shouldlog(l.logger, args...)
            Logging.catch_exceptions(l::JAXWarningFilter) = Logging.catch_exceptions(l.logger)

            function Logging.handle_message(l::JAXWarningFilter, level, message, mod, group, id, file, line; kwargs...)
                if !occursin("Failed to import jax", string(message))
                    Logging.handle_message(l.logger, level, message, mod, group, id, file, line; kwargs...)
                end
            end

            let logger = global_logger()
                global_logger(JAXWarningFilter(logger))
                try
                    using Reactant
                finally
                    global_logger(logger)
                end
            end
        """)

    if device == "cpu":
        kwargs["device"] = jl.cpu_device()
    elif device == "reactant_cpu":
        jl.seval('Reactant.set_default_backend("cpu")')
        kwargs["device"] = jl.seval("reactant_device()")
    elif device == "reactant_gpu":
        jl.seval('Reactant.set_default_backend("gpu")')
        kwargs["device"] = jl.seval("reactant_device()")
    elif device == "reactant_tpu":
        jl.seval('Reactant.set_default_backend("tpu")')
        kwargs["device"] = jl.seval("reactant_device()")
    else:
        raise ValueError(f"Unsupported device selection: {device!r}")

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
    """
    An enumeration of training data normalization options.

    Attributes:
        FULL_NORMALIZATION
        NO_NORMALIZATION
        HYBRID_NORMALIZATION
    """
    
    FULL_NORMALIZATION   = jl.GenerativeModelProtocols.full_normalization
    NO_NORMALIZATION     = jl.GenerativeModelProtocols.no_normalization
    HYBRID_NORMALIZATION = jl.GenerativeModelProtocols.hybrid_normalization


class GenerativeModelProtocol:
    """
    A protocol containing a user-specified generative model, and all of its 
    parameters needed to train it and evaluate it. By default the training of 
    the generative model is done in the CPU, however, this can be changed by 
    setting `device` to a GPU device of preference.

    Attributes:
        model (`AbstractGenerativeModel`): The generative model used by the protocol.
        training_data (`np.ndarray`): The training data used by the protocol.
        normalize_data (`tuple[bool, ...]`): The normalization settings for the training data.
        input_size (`int`): The input size of the generative model.
        latent_size (`int`): The latent size of the generative model.
        _jl_protocol: The underlying Julia protocol instance. User should not modify this directly.
    """

    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_protocol",)

    def __init__(
            self, 
            model: AbstractGenerativeModel = None, 
            training_data: np.ndarray = np.zeros((0, 0)),
            normalize_data = TrainingDataNormalization.HYBRID_NORMALIZATION,
            _jl_protocol = None
        ):
        if _jl_protocol is not None:
            self._jl_protocol = _jl_protocol
        else:
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

        if 'device' in kwargs:
            kwargs['device'] = kwargs['device'].value

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
    """
    Returns the default optimiser for a given generative model protocol.
    
    Arguments:
        protocol (`GenerativeModelProtocol`): The generative model protocol for which to get the default optimiser.
    """

    _jl_optimiser = jl.GenerativeModelProtocols.default_optimiser(protocol._jl_protocol)

    if isinstance(_jl_optimiser, tuple):
        return (_get_jl_optimiser_struct(_jl_optimiser[0]), _get_jl_optimiser_struct(_jl_optimiser[1]))
    else:
        return _get_jl_optimiser_struct(_jl_optimiser)


def _default_main_group_name() -> str:
    return jl.seval("GenerativeModelProtocols.@default_main_group_name")


def _default_metadata_group_name() -> str:
    return jl.seval("GenerativeModelProtocols.@default_metadata_group_name")


def _default_generative_model_group_name() -> str:
    return jl.seval("GenerativeModelProtocols.@default_generative_model_group_name")


def save(
        protocol: GenerativeModelProtocol, 
        file_name: str,
        main_group_name : str = _default_main_group_name(),
        metadata_group_name : str = _default_metadata_group_name(),
        generative_model_group_name : str = _default_generative_model_group_name()
    ):
    """
    Saves a generative model protocol to a file. The file will be saved as an 
    HDF5 file.

    Arguments:
        protocol (`GenerativeModelProtocol`): The generative model protocol to save.
        file_name (`str`): The name of the file where the protocol will be saved.
        main_group_name (`str`, optional): The name of the main group in the HDF5 file.
        metadata_group_name (`str`, optional): The name of the metadata group in the HDF5 file.
        generative_model_group_name (`str`, optional): The name of the generative model group in the HDF5 file.
    """

    jl.save(
        file_name, 
        protocol._jl_protocol,
        main_group_name = main_group_name,
        metadata_group_name = metadata_group_name,
        generative_model_group_name = generative_model_group_name
    )


def load(
        file_name: str, 
        training_data: np.ndarray = np.zeros((0, 0)),
        main_group_name : str = _default_main_group_name(),
        metadata_group_name : str = _default_metadata_group_name(),
        generative_model_group_name : str = _default_generative_model_group_name()
    ) -> GenerativeModelProtocol:
    """
    Loads a generative model protocol from a file. The file must be an HDF5 
    file.

    Arguments:
        file_name (`str`): The name of the file from which to load the protocol.
        training_data (`np.ndarray`, optional): The training data to use for the loaded protocol.
        main_group_name (`str`, optional): The name of the main group in the HDF5 file.
        metadata_group_name (`str`, optional): The name of the metadata group in the HDF5 file.
        generative_model_group_name (`str`, optional): The name of the generative model group in the HDF5 file.
    """

    _jl_protocol = None
    if training_data is not None:
        _jl_protocol = jl.GenerativeModelProtocol(
            file_name,
            main_group_name = main_group_name,
            metadata_group_name = metadata_group_name,
            generative_model_group_name = generative_model_group_name
        )
    else:
        _jl_protocol = jl.GenerativeModelProtocol(
            file_name, 
            jl.Matrix(training_data),
            main_group_name = main_group_name,
            metadata_group_name = metadata_group_name,
            generative_model_group_name = generative_model_group_name
        )
    
    return GenerativeModelProtocol(_jl_protocol = _jl_protocol)


__all__ = [
    "TrainingDataNormalization",
    "GenerativeModelProtocol",
    "default_optimiser",
    "save",
    "load"
]
