import numpy as np
from juliacall import Main as jl
from .auxiliary.tabular_denoiser import TabularDenoiser
from .abstract_generative_model import AbstractGenerativeModel


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
        beta_end: float = 0.02
    ):
    """
    Creates a default diffusion model with the specified dimensions and noise schedule.

    Args:
        dimensions (`int`): The number of dimensions of the data.
        T (`int`, optional): The number of diffusion steps. Defaults to 5.
        beta_start (`float`, optional): The starting value of the noise schedule. Defaults to 1.0E-4.
        beta_end (`float`, optional): The ending value of the noise schedule. Defaults to 0.02.

    Returns:
        `DiffusionModel`: The default diffusion model instance.
    """

    _jl_model = jl.seval(f"GenerativeModelProtocols.DiffusionModel({dimensions}, {T}, {beta_start}, {beta_end})")
    return DiffusionModel(_jl_model = _jl_model)


__all__ = [
    "DiffusionModel",
    "default_diffusion_model"
]

