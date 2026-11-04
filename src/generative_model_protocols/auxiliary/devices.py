from enum import Enum


class Device(Enum):
    CPU          = "cpu"
    REACTANT_CPU = "reactant_cpu"
    REACTANT_GPU = "reactant_gpu"
    REACTANT_TPU = "reactant_tpu"


__all__ = [
    "Device"
]
