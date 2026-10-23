from juliacall import Main as jl
from .auxiliary.lux import Chain
from .generative_model_protocol import AbstractGenerativeModel


class NormalizingFlow(AbstractGenerativeModel):
    __module__ = "GenerativeModelProtocols"

    def __init__(
            self, 
            velocity_field: Chain = None,
            _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            super().__init__(
                jl.GenerativeModelProtocols.NormalizingFlow(
                    velocity_field = velocity_field._jl_chain
                )
            )

    @property
    def velocity_field(self) -> Chain:
        return Chain(_jl_chain = self._jl_model.velocity_field)


def default_normalizing_flow(dimensions: int):
    _jl_model = jl.seval(f"GenerativeModelProtocols.NormalizingFlow({dimensions})")
    return NormalizingFlow(_jl_model = _jl_model)

