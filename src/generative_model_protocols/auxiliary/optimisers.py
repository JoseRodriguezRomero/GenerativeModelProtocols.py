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
    """
    Classic gradient descent optimiser with learning rate `η`. For each
    parameter `p` and its gradient `dp`, this runs `p -= η*dp`.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
    """

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
    """
    Gradient descent optimizer with learning rate `η` and momentum `ρ`.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        rho (`float`): Momentum. Controls the acceleration of gradient descent in the prominent direction, in effect dampening oscillations.
    """

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
    """
    Gradient descent optimizer with learning rate `η` and Nesterov momentum `ρ`.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        rho (`float`): Nesterov momentum. Controls the acceleration of gradient descent in the prominent direction, in effect dampening oscillations.
    """

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
    """
    Optimizer using the Rprop algorithm. A full-batch learning algorithm that
    depends only on the sign of the gradient.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        ell (`tuple[float, float]`): Scaling factors. Multiplicative increase and decrease factors.
        gamma (`tuple[float, float]`): Step sizes. Minimal and maximal allowed step sizes.
    """

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
    """
    Optimizer using the RMSProp algorithm. Often a good choice for recurrent
    networks. Parameters other than learning rate generally don't need tuning.

    Centred RMSProp is a variant which normalises gradients by an estimate their
    variance, instead of their second moment.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        rho (`float`): Momentum. Controls the acceleration of gradient descent in the prominent direction, in effect dampening oscillations.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
        centred (`bool`): Keyword centred (or centered). Indicates whether to use centred variant of the algorithm.
    """

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
    """
    Adam optimiser.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    Rectified Adam optimizer.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    AdaMax is a variant of Adam based on the ∞-norm.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    OAdam (Optimistic Adam) is a variant of Adam adding an "optimistic" term
    suitable for adversarial training.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    AdaGrad optimizer. It has parameter specific learning rates based on how
    frequently it is updated. Parameters don't need tuning.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
            self._jl_optimiser = jl.AdaGrad(
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
    """
    AdaDelta is a version of AdaGrad adapting its learning rate based on a
    window of past gradient updates. Parameters don't need tuning.

    Attributes:
            rho (`float`): Factor by which the gradient is decayed at each time step.
            epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    The AMSGrad version of the Adam optimiser. Parameters don't need tuning.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    NAdam is a Nesterov variant of Adam. Parameters don't need tuning.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    AdamW is a variant of Adam fixing (as in repairing) its weight decay
    regularization. Implemented as an OptimiserChain of Adam and WeightDecay`.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        lambda_ (`float`): Weight decay. Controls the strength of L2 regularisation.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
        couple (`bool`): If true, the weight decay is coupled with the learning rate, as in pytorch's `AdamW`. This corresponds to an update of the form `x = x - η * (dx + λ * x)`, where dx is the update from Adam with learning rate `1`. If false, the weight decay is decoupled from the learning rate, in the spirit of the original paper. This corresponds to an update of the form `x = x - η * dx - λ * x`. Default is `true`.
    """

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
    """
    The AdaBelief optimiser is a variant of the well-known Adam optimiser.

    Attributes:
        eta (`float`): Learning rate. Amount by which gradients are discounted before updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
        epsilon (`float`): Constant to prevent division by zero (no need to change default)
    """

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
    """
    Lion optimiser.

    Attributes:
        eta (`float`): Learning rate. Magnitude by which gradients are updating the weights.
        beta (`tuple[float, float]`): Decay of momentums. Exponential decay for the first (β1) and the second (β2) momentum estimate.
    """

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
