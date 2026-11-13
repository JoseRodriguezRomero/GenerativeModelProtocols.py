from juliacall import Main as jl
from .lux import ActivationFunction, Chain, Dense
from .capture_julia_display import capture_julia_display


_max_period = 10000.0


class TabularDenoiser:
    """
    A time-conditioned residual neural network for vector-based diffusion 
    models.

    Attributes:
        T (`int`): The number of time steps.
        time_embedding_mlp (`Chain`): The MLP used for time embedding.
        input_projection (`Dense`): The input projection layer.
        residual_layers (`tuple[Dense, ...]`): The residual layers of the network.
        time_projection_layers (`tuple[Dense, ...]`): The time projection layers.
        output_projection (`Dense`): The output projection layer.
        max_period (`float`): The maximum period for the time embedding.
        _jl_denoiser: The underlying Julia model instance. User should not modify this directly.
    """
    __module__ = "GenerativeModelProtocols.Auxiliary"
    __slots__ = ("_jl_denoiser",)

    def __new__(
        cls,
        T: int = None,
        time_embedding_mlp: Chain = None,
        input_projection: Dense = None,
        residual_layers: tuple[Dense, ...] = None,
        time_projection_layers: tuple[Dense, ...] = None,
        output_projection: Dense = None,
        max_period: float = _max_period,
        _jl_denoiser = None,
    ):
        self = super().__new__(cls)

        if _jl_denoiser is not None:
            self._jl_denoiser = _jl_denoiser
        else:
            residual_layers_jl = jl.seval(
                "layers -> NamedTuple{Tuple(Symbol.(:layer_, eachindex(layers)))}(layers)"
            )(tuple(layer._jl_dense for layer in residual_layers))

            time_projection_layers_jl = jl.seval(
                "layers -> NamedTuple{Tuple(Symbol.(:layer_, eachindex(layers)))}(layers)"
            )(tuple(layer._jl_dense for layer in time_projection_layers))

            self._jl_denoiser = jl.GenerativeModelProtocols.TabularDenoiser(
                T=T,
                time_embedding_mlp=time_embedding_mlp._jl_chain,
                input_projection=input_projection._jl_dense,
                residual_layers=residual_layers_jl,
                time_projection_layers=time_projection_layers_jl,
                output_projection=output_projection._jl_dense,
                max_period=max_period,
            )

        return self

    @property
    def T(self) -> int:
        return self._jl_denoiser.T

    @property
    def time_embedding_mlp(self) -> Chain:
        return Chain(_jl_chain = self._jl_denoiser.time_embedding_mlp)

    @property
    def input_projection(self) -> Dense:
        return Dense(_jl_dense = self._jl_denoiser.input_projection)

    @property
    def residual_layers(self) -> Dense:
        return Dense(_jl_dense = self._jl_denoiser.residual_layers)

    @property
    def residual_layers(self) -> tuple[Dense, ...]:
        layers = self._jl_denoiser.residual_layers
        return tuple(
            Dense(_jl_dense=getattr(layers, str(name)))
            for name in jl.propertynames(layers)
        )

    @property
    def time_projection_layers(self) -> tuple[Dense, ...]:
        layers = self._jl_denoiser.time_projection_layers
        return tuple(
            Dense(_jl_dense=getattr(layers, str(name)))
            for name in jl.propertynames(layers)
        )

    @property
    def output_projection(self) -> Dense:
        return Dense(_jl_dense=self._jl_denoiser.output_projection)

    @property
    def max_period(self) -> float:
        return self._jl_denoiser.max_period

    def __setattr__(self, name, value):
        if name == "_jl_denoiser" and hasattr(self, "_jl_denoiser"):
            raise AttributeError("_jl_denoiser is read-only")
        super().__setattr__(name, value)

    def __repr__(self):
        return capture_julia_display(self._jl_denoiser)
    
    def __str__(self):
        return self.__repr__()


def default_tabular_denoiser(
    num_inputs: int,
    T:int = 32,
    res_net_layers:int = 3,
    hidden_layers:int = 5,
    hidden_layer_size:int = 32,
    activation_function: ActivationFunction = ActivationFunction.SWISH,
    max_period: float = _max_period,
):
    """
    Creates a default Tabular Denoiser with the specified number of inputs, 
    hidden layer size, number of time steps, activation function, and maximum 
    period.

    Args:
        num_inputs (`int`): The number of input features.
        T (`int`, optional): The number of time steps. Defaults to 32.
        max_period (`float`, optional): The maximum period for the time embedding. Defaults to 10000.0 (defined by `_max_period`).
        res_net_layers (`int`, optional): The number of residual layers. Defaults to 3.
        hidden_layers (`int`, optional): The number of hidden layers in the time embedding mlp. Defaults to 5.
        hidden_layer_size (`int`, optional): The size of the hidden layers. Defaults to 32.
        activation_function (`ActivationFunction`, optional): The activation function to use. Defaults to `ActivationFunction.SWISH`.
        max_period (`float`, optional): The maximum period for the time embedding. Defaults to 10000.0 (defined by `_max_period`).

    Returns:
        `TabularDenoiser`: The default Tabular Denoiser instance.
    """

    _jl_denoiser = jl.seval(f"GenerativeModelProtocols.TabularDenoiser(\
        {num_inputs},\
        T = {T},\
        res_net_layers = {res_net_layers},\
        hidden_layers = {hidden_layers},\
        hidden_layer_size = {hidden_layer_size},\
        activation_function = {activation_function.value},\
        max_period = {max_period}\
        )"
    )
    return TabularDenoiser(_jl_denoiser = _jl_denoiser)


__all__ = [
    "TabularDenoiser",
    "default_tabular_denoiser",
    "_max_period"
]

