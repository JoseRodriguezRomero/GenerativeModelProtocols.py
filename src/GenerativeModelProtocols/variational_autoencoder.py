from juliacall import Main as jl
from .auxiliary.lux import Chain
from .generative_model_protocol import AbstractGenerativeModel


class VariationalAutoencoder(AbstractGenerativeModel):
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


def default_variational_autoencoder(dimensions: int, latent_dimension: int = 1, latent_layers: int = 1):
    _jl_model = jl.seval(f"GenerativeModelProtocols.VariationalAutoencoder({dimensions}, {latent_dimension}, {latent_layers})")
    return VariationalAutoencoder(_jl_model = _jl_model)

