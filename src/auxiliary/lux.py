from enum import Enum
from juliacall import Main as jl

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
    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        activation_function: ActivationFunction = ActivationFunction.SWISH,
    ):
        self._jl_dense = jl.seval(
            f"Lux.Dense({num_inputs} => {num_outputs}, Lux.{activation_function.value})"
        )

    def __repr__(self):
        return str(self._jl_dense)

    def __str__(self):
        return repr(self)


class Chain:
    def __init__(self, *layers):
        if not layers:
            raise ValueError("Chain requires at least one layer.")

        julia_layers = [
            layer._jl_dense if isinstance(layer, Dense) else layer._jl_chain
            for layer in layers
        ]

        self._jl_chain = jl.Lux.Chain(*julia_layers)

    def __repr__(self):
        return str(self._jl_chain)

    def __str__(self):
        return repr(self._jl_chain)

