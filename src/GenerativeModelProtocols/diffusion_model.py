from juliacall import Main as jl
from .auxiliary.tabular_denoiser import TabularDenoiser
from .generative_model_protocol import AbstractGenerativeModel


class DiffusionModel(AbstractGenerativeModel):
    def __init__(
        self,
        noise_schedule: tuple[float, ...] = None,
        denoiser_model: TabularDenoiser = None,
        _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            self._jl_model = jl.GenerativeModelProtocols.DiffusionModel(
                noise_schedule,
                denoiser_model._jl_denoiser,
            )


def default_diffusion_model(dimensions: int, T: int = 5, beta_start: float = 1.0E-4, beta_end: float = 0.02):
    _jl_model = jl.seval(f"GenerativeModelProtocols.DiffusionModel({dimensions}, {T}, {beta_start}, {beta_end})")
    return DiffusionModel(_jl_model = _jl_model)

