from juliacall import Main as jl
from .auxiliary.lux import Chain
from .generative_model_protocol import AbstractGenerativeModel
from .variational_autoencoder import VariationalAutoencoder


class GenerativeAdversarialNetwork(AbstractGenerativeModel):
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_model",)

    def __init__(
        self,
        discriminator: Chain = None,
        vae_model: VariationalAutoencoder = None,
        _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            self._jl_model = jl.GenerativeModelProtocols.GenerativeAdversarialNetwork(
                discriminator=discriminator._jl_chain,
                vae_model=vae_model._jl_model,
            )

    @property
    def discriminator(self) -> Chain:
        return Chain(_jl_chain = self._jl_model.discriminator)

    @property
    def vae_model(self) -> VariationalAutoencoder:
        return VariationalAutoencoder(_jl_model = self._jl_model.vae_model)


def default_generative_adversarial_network(dimensions: int, latent_dimension: int = 1, latent_layers: int = 1):
    _jl_model = jl.seval(f"GenerativeModelProtocols.GenerativeAdversarialNetwork({dimensions}, {latent_dimension}, {latent_layers})")
    return GenerativeAdversarialNetwork(_jl_model = _jl_model)

