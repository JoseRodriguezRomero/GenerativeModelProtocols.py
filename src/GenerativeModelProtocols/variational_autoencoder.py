from juliacall import Main as jl
from .auxiliary.lux import Chain
from .generative_model_protocol import AbstractGenerativeModel


class VariationalAutoencoder(AbstractGenerativeModel):
    def __init__(
            self, 
            encoders: tuple[Chain, ...] = None,
            decoders: tuple[Chain, ...] = None,
            _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            super().__init__(
                jl.GenerativeModelProtocols.VariationalAutoencoder(
                    jl.convert(jl.Tuple, tuple(encoder._jl_chain for encoder in encoders)),
                    jl.convert(jl.Tuple, tuple(decoder._jl_chain for decoder in decoders)),
                )
            )


def default_variational_autoencoder(dimensions: int, latent_dimension: int = 1, latent_layers: int = 1):
    _jl_model = jl.seval(f"GenerativeModelProtocols.VariationalAutoencoder({dimensions}, {latent_dimension}, {latent_layers})")
    return VariationalAutoencoder(_jl_model = _jl_model)

