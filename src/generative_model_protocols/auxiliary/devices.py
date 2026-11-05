from enum import Enum


class Device(Enum):
    """
    An enumeration of devices available to use to train a generative model.

    Attributes:
        CPU
        REACTANT_CPU
        REACTANT_GPU
        REACTANT_TPU
    """

    CPU          = "cpu"
    REACTANT_CPU = "reactant_cpu"
    REACTANT_GPU = "reactant_gpu"
    REACTANT_TPU = "reactant_tpu"


__all__ = [
    "Device"
]
