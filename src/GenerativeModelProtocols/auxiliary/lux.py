import io
from contextlib import redirect_stdout

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
        output = io.StringIO()
        with redirect_stdout(output):
            jl.display(self._jl_dense)
        return output.getvalue().rstrip()
    
    def __str__(self):
        return self.__repr__()


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
        output = io.StringIO()
        with redirect_stdout(output):
            jl.display(self._jl_chain)
        return output.getvalue().rstrip()
        
    def __str__(self):
        return self.__repr__()

