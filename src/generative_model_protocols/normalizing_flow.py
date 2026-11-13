from juliacall import Main as jl
from .auxiliary.lux import ActivationFunction, Chain
from .abstract_generative_model import AbstractGenerativeModel


class NormalizingFlow(AbstractGenerativeModel):
    """
    A structure containing the general parameters needed to evaluate and train a 
    Continuous Normalizing Flow (CNF). Once trained, it can be used a generative 
    model.

    Attributes:
        velocity_field (`Chain`): The velocity field of the normalizing flow.
        _jl_model: The underlying Julia model instance. User should not modify this directly.
    """
    __module__ = "GenerativeModelProtocols"
    __slots__ = ("_jl_model",)

    def __init__(
            self, 
            velocity_field: Chain = None,
            _jl_model = None
    ):
        if _jl_model is not None:
            self._jl_model = _jl_model
        else:
            self._jl_model = jl.GenerativeModelProtocols.NormalizingFlow(
                velocity_field = velocity_field._jl_chain
            )

    @property
    def velocity_field(self) -> Chain:
        return Chain(_jl_chain = self._jl_model.velocity_field)


def default_normalizing_flow(
        dimensions: int,
        hidden_layers: int = 5,
        hidden_layer_size: int = 32,
        activation_function: ActivationFunction = ActivationFunction.SWISH
    ):
    """
    Creates a default Continuous Normalizing Flow (CNF) with the specified 
    dimensions.

    Args:
        dimensions (`int`): The number of dimensions of the data.
        hidden_layers (`int`, optional): The number of hidden layers in the velocity field. Defaults to 5.
        hidden_layer_size (`int`, optional): The size of the hidden layers in the velocity field. Defaults to 32.
        activation_function (`ActivationFunction`, optional): The activation function used by the velocity field. Defaults to `ActivationFunction.SWISH`.

    Returns:
        `NormalizingFlow`: The default CNF instance.
    """
    _jl_model = jl.seval(f"GenerativeModelProtocols.NormalizingFlow(\
        {dimensions},\
        hidden_layers = {hidden_layers},\
        hidden_layer_size = {hidden_layer_size},\
        activation_function = {activation_function.value}\
        )"
    )
    return NormalizingFlow(_jl_model = _jl_model)


__all__ = [
    "NormalizingFlow",
    "default_normalizing_flow"
]

