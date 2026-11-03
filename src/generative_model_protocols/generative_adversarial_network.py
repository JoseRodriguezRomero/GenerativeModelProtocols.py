from juliacall import Main as jl
from .auxiliary.lux import Chain
from .abstract_generative_model import AbstractGenerativeModel
from .variational_autoencoder import VariationalAutoencoder


class GenerativeAdversarialNetwork(AbstractGenerativeModel):
    """
    A class containing the general parameters needed to evaluate and train a 
    Generative Adversarial Network (GAN). Once trained, it can be used a 
    generative model.

    Attributes:
        discriminator (`Chain`): The discriminator network used by the GAN.
        vae_model (`VariationalAutoencoder`): The VAE model used by the GAN.
        _jl_model: The underlying Julia model instance. User should not modify this directly.
    """
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


def default_generative_adversarial_network(
        dimensions: int, 
        latent_dimension: int = 1, 
        latent_layers: int = 1
    ):
    """
    Creates a default Generative Adversarial Network (GAN) with the specified 
    dimensions, latent dimension, and number of latent layers.

    Args:
        dimensions (`int`): The number of dimensions of the data.
        latent_dimension (`int`, optional): The dimension of the latent space. Defaults to 1.
        latent_layers (`int`, optional): The number of layers in the latent space. Defaults to 1.

    Returns:
        `GenerativeAdversarialNetwork`: The default GAN instance.
    """
    _jl_model = jl.seval(f"GenerativeModelProtocols.GenerativeAdversarialNetwork({dimensions}, {latent_dimension}, {latent_layers})")
    return GenerativeAdversarialNetwork(_jl_model = _jl_model)


__all__ = [
    "GenerativeAdversarialNetwork",
    "default_generative_adversarial_network"
]