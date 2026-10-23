from enum import Enum
from juliacall import Main as jl
from .capture_julia_display import capture_julia_display


class ActivationFunction(Enum):
    RELU       = "relu"
    GELU       = "gelu"
    TANH       = "tanh"
    SIGMOID    = "sigmoid"
    SOFTMAX    = "softmax"
    LEAKYRELU  = "leakyrelu"
    ELU        = "elu"
    SWISH      = "swish"
    IDENTITY   = "identity"


class Dense:
    __module__ = "GenerativeModelProtocols.Auxiliary"

    def __init__(
        self,
        num_inputs: int = None,
        num_outputs: int = None,
        activation_function: ActivationFunction = ActivationFunction.SWISH,
        _jl_dense = None
    ):
        if _jl_dense is not None:
            self._jl_dense = _jl_dense
        else:
            self._jl_dense = jl.seval(
                f"Lux.Dense({num_inputs} => {num_outputs}, Lux.{activation_function.value})"
            )

    def __repr__(self):
        return capture_julia_display(self._jl_dense)
    
    def __str__(self):
        return self.__repr__()


class Chain:
    __module__ = "GenerativeModelProtocols.Auxiliary"

    def __init__(
            self, 
            *layers: Dense, 
            _jl_chain = None
        ):
        if _jl_chain is not None:
            self._jl_chain = _jl_chain
        else:
            if not layers:
                raise ValueError("Chain requires at least one layer.")

            julia_layers = [
                layer._jl_dense if isinstance(layer, Dense) else layer._jl_chain
                for layer in layers
            ]

            self._jl_chain = jl.Lux.Chain(*julia_layers)

    def __repr__(self):
        return capture_julia_display(self._jl_chain)
        
    def __str__(self):
        return self.__repr__()

