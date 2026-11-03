from juliacall import Main as jl
from .capture_julia_display import capture_julia_display


class AbstractOptimiser:
    def __init__(self):
        raise TypeError("Cannot instantiate AbstractOptimiser")

    def __setattr__(self, name, value):
        if name == "_jl_optimiser" and hasattr(self, "_jl_optimiser"):
            raise AttributeError("_jl_optimiser is read-only")
        super().__setattr__(name, value)

    def __repr__(self):
        return capture_julia_display(self._jl_optimiser)

    def __str__(self):
        return self.__repr__()


class Descent(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self, 
        eta: float = 1.0E-1, 
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Descent(
                **{
                    "eta": eta
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")


class Momentum(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.01,
        rho: float = 0.9,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Momentum(
                **{
                    "eta": eta,
                    "rho": rho
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def rho(self) -> float:
        return getattr(self._jl_optimiser, "rho")


class Nesterov(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        rho: float = 0.9,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Nesterov(
                **{
                    "eta": eta,
                    "rho": rho
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def rho(self) -> float:
        return getattr(self._jl_optimiser, "rho")


class Rprop(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 1E-3,
        ell: tuple[float, float] = (0.5, 1.2),
        gamma: tuple[float, float] = (1e-6, 50.0),
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Rprop(
                **{
                    "eta": eta,
                    "ell": ell,
                    "gamma": gamma,
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def ell(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "ell")

    @property
    def gamma(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "gamma")


class RMSProp(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        centred: bool = False,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.RMSProp(
                **{
                    "eta": eta,
                    "rho": rho,
                    "epsilon": epsilon,
                    "centred": centred
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def rho(self) -> float:
        return getattr(self._jl_optimiser, "rho")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")

    @property
    def centred(self) -> bool:
        return self._jl_optimiser.centred


class Adam(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Adam(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class RAdam(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.RAdam(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class AdaMax(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.AdaMax(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class OAdam(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.5, 0.9),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.OAdam(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class AdaGrad(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.OAdam(
                **{
                    "eta": eta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class AdaDelta(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        rho: float = 0.9,
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.AdaDelta(
                **{
                    "rho": rho,
                    "epsilon": epsilon
                }
            )

    @property
    def rho(self) -> float:
        return getattr(self._jl_optimiser, "rho")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class AMSGrad(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.AMSGrad(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class NAdam(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.NAdam(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class AdamW(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        lambda_: float = 0.0,
        epsilon: float = 1e-8,
        couple: bool = True,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.AdamW(
                **{
                    "eta": eta,
                    "beta": beta,
                    "lambda": lambda_,
                    "epsilon": epsilon,
                    "couple": couple
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def lambda_(self) -> float:
        return getattr(self._jl_optimiser, "lambda")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")

    @property
    def couple(self) -> bool:
        return self._jl_optimiser.couple


class AdaBelief(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-16,
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.AdaBelief(
                **{
                    "eta": eta,
                    "beta": beta,
                    "epsilon": epsilon
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")

    @property
    def epsilon(self) -> float:
        return getattr(self._jl_optimiser, "epsilon")


class Lion(AbstractOptimiser):
    __module__ = "GenerativeModelProtocols.Auxiliary.Optimisers"
    __slots__ = ("_jl_optimiser",)

    def __init__(
        self,
        eta: float = 0.001,
        beta: tuple[float, float] = (0.9, 0.999),
        _jl_optimiser = None
    ):
        if _jl_optimiser is not None:
            self._jl_optimiser = _jl_optimiser
        else:
            self._jl_optimiser = jl.Lion(
                **{
                    "eta": eta,
                    "beta": beta,
                }
            )

    @property
    def eta(self) -> float:
        return getattr(self._jl_optimiser, "eta")

    @property
    def beta(self) -> tuple[float, float]:
        return getattr(self._jl_optimiser, "beta")


__all__ = [
    "Descent",
    "Momentum",
    "Nesterov",
    "Rprop",
    "RMSProp",
    "Adam",
    "RAdam",
    "AdaMax",
    "OAdam",
    "AdaGrad",
    "AdaDelta",
    "AMSGrad",
    "NAdam",
    "AdamW",
    "AdaBelief",
    "Lion"
]

