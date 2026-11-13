import numpy as np
from juliacall import Main as jl
from .auxiliary.tabular_denoiser import TabularDenoiser
from .abstract_generative_model import AbstractGenerativeModel
from .auxiliary.tabular_denoiser import _max_period
from .auxiliary.lux import ActivationFunction


class DiffusionModel(AbstractGenerativeModel):
    """A class containing the general parameters needed to evaluate and train 
    a Diffusion Model (DM). Once trained, its denoiser model can be used as a 
    generative model.

    Attributes:
        denoiser_model (`TabularDenoiser`): The denoiser model used by the diffusion model.
        noise_schedule (`np.ndarray`): The noise schedule (β) used by the diffusion model.
        _jl_model: The underlying Julia model instance. User should not modify this directly.
    """

    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_model",)

    def __init__(
        self,
        noise_schedule: np.ndarray = None,
        denoiser_model: TabularDenoiser = None,
        _jl_model = None,
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            self._jl_model = jl.GenerativeModelProtocols.DiffusionModel(
                tuple(noise_schedule.tolist()),
                denoiser_model._jl_denoiser,
            )

    @property
    def denoiser_model(self) -> TabularDenoiser:
        return TabularDenoiser(_jl_denoiser = self._jl_model.denoiser_model)

    @property
    def noise_schedule(self) -> np.ndarray:
        return np.asarray(getattr(self._jl_model, "β"))


def default_diffusion_model(
        dimensions: int, 
        T: int = 5, 
        beta_start: float = 1.0E-4, 
        beta_end: float = 0.02,
        res_net_layers: int = 3,
        hidden_layers:int = 5,
        hidden_layer_size:int = 32,
        activation_function: ActivationFunction = ActivationFunction.SWISH,
        max_period: float = _max_period,
    ):
    """
    Creates a default diffusion model with the specified dimensions and noise schedule.

    Args:
        dimensions (`int`): The number of dimensions of the data.
        T (`int`, optional): The number of diffusion steps. Defaults to 5.
        beta_start (`float`, optional): The starting value of the noise schedule. Defaults to 1.0E-4.
        beta_end (`float`, optional): The ending value of the noise schedule. Defaults to 0.02.
        res_net_layers (`int`, optional): The number of residual layers in the denoiser model. Defaults to 3.
        hidden_layers (`int`, optional): The number of hidden layers in the time embedding mlp of the denoiser model. Defaults to 5.
        hidden_layer_size (`int`, optional): The size of the hidden layers in the denoiser model. Defaults to 32.
        activation_function (`ActivationFunction`, optional): The activation function used by the denoiser model. Defaults to `ActivationFunction.SWISH`.
        max_period (`float`, optional): The maximum period for the time embedding. Defaults to 10000.0.

    Returns:
        `DiffusionModel`: The default diffusion model instance.
    """

    _jl_model = jl.seval(f"GenerativeModelProtocols.DiffusionModel(\
        {dimensions}, {T}, {beta_start}, {beta_end};\
        res_net_layers = {res_net_layers},\
        hidden_layers = {hidden_layers},\
        hidden_layer_size = {hidden_layer_size},\
        activation_function = {activation_function.value},\
        max_period = {max_period}\
        )"
    )
    return DiffusionModel(_jl_model = _jl_model)


__all__ = [
    "DiffusionModel",
    "default_diffusion_model"
]

