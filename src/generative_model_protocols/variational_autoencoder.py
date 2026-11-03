from juliacall import Main as jl
from .auxiliary.lux import Chain
from .abstract_generative_model import AbstractGenerativeModel


class VariationalAutoencoder(AbstractGenerativeModel):
    """
    A structure containing the general parameters needed to evaluate and train 
    a Variational Autoencoder (VAE). Once trained, its decoder can be used as a 
    generative model.

    Attributes:
        encoders (`tuple[Chain, ...]`): The encoders of the VAE.
        decoders (`tuple[Chain, ...]`): The decoders of the VAE.
        _jl_model: The underlying Julia model instance. User should not modify this directly.
    """
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_model",)

    def __init__(
            self, 
            encoders: tuple[Chain, ...] = None,
            decoders: tuple[Chain, ...] = None,
            _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            self._jl_model = jl.GenerativeModelProtocols.VariationalAutoencoder(
                jl.convert(jl.Tuple, tuple(encoder._jl_chain for encoder in encoders)),
                jl.convert(jl.Tuple, tuple(decoder._jl_chain for decoder in decoders)),
            )

    @property
    def encoders(self) -> tuple[Chain, ...]:
        return tuple(
            Chain(_jl_chain=encoder)
            for encoder in self._jl_model.encoders
        )

    @property
    def decoders(self) -> tuple[Chain, ...]:
        return tuple(
            Chain(_jl_chain=decoder)
            for decoder in self._jl_model.decoders
        )


def default_variational_autoencoder(
        dimensions: int, 
        latent_dimension: int = 1, 
        latent_layers: int = 1
    ):
    """
    Creates a default Variational Autoencoder (VAE) with the specified 
    dimensions, latent dimension, and number of latent layers.

    Args:
        dimensions (`int`): The number of dimensions of the data.
        latent_dimension (`int`, optional): The dimension of the latent space. Defaults to 1.
        latent_layers (`int`, optional): The number of layers in the latent space. Defaults to 1.

    Returns:
        `VariationalAutoencoder`: The default VAE instance.
    """
    _jl_model = jl.seval(f"GenerativeModelProtocols.VariationalAutoencoder({dimensions}, {latent_dimension}, {latent_layers})")
    return VariationalAutoencoder(_jl_model = _jl_model)


__all__ = [
    "VariationalAutoencoder",
    "default_variational_autoencoder"
]

