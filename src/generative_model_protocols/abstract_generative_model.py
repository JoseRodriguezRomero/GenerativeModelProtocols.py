from juliacall import Main as jl
from .auxiliary.capture_julia_display import capture_julia_display


class AbstractGenerativeModel:
    def __init__(self):
        raise TypeError("Cannot instantiate AbstractGenerativeModel")

    def __setattr__(self, name, value):
        if name == "_jl_model" and hasattr(self, "_jl_model"):
            raise AttributeError("_jl_model is read-only")
        super().__setattr__(name, value)

    def __repr__(self):
        return capture_julia_display(self._jl_model)

    def __str__(self):
        return self.__repr__()

