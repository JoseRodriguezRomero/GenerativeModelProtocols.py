from juliacall import Main as jl
from .lux import Chain, Dense
from .capture_julia_display import capture_julia_display

class TabularDenoiser:
    __module__ = "GenerativeModelProtocols.Auxiliary"

    def __new__(
        cls,
        T: int = None,
        time_embedding_mlp: Chain = None,
        input_projection: Dense = None,
        residual_layers: tuple[Dense, ...] = None,
        time_projection_layers: tuple[Dense, ...] = None,
        output_projection: Dense = None,
        max_period: float = 10000.0,
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

    def __repr__(self):
        return capture_julia_display(self._jl_denoiser)
    
    def __str__(self):
        return self.__repr__()


def default_tabular_denoiser(
    num_inputs: int,
    hidden_layer_size: int = 64,
    T: int = 32,
    activation_function=jl.GenerativeModelProtocols.swish,
    max_period: float = 10000.0,
):
    _jl_denoiser = jl.GenerativeModelProtocols.TabularDenoiser(
        num_inputs,
        hidden_layer_size=hidden_layer_size,
        T=T,
        activation_function=activation_function,
        max_period=max_period,
    )
    return TabularDenoiser(_jl_denoiser = _jl_denoiser)

