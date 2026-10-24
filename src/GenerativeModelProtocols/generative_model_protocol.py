from juliacall import Main as jl
from .auxiliary.capture_julia_display import capture_julia_display

class AbstractGenerativeModel:
    def __new__(cls, *args, **kwargs):
        if cls is AbstractGenerativeModel:
            raise TypeError("Cannot instantiate AbstractGenerativeModel")
        return super().__new__(cls)

    def __setattr__(self, name, value):
        if name == "_jl_model" and hasattr(self, "_jl_model"):
            raise AttributeError("_jl_model is read-only")
        super().__setattr__(name, value)

    def __repr__(self):
        return capture_julia_display(self._jl_model)

    def __str__(self):
        return self.__repr__()


# class GenerativeModelProtocol:
#     def __new__(self, model : AbstractGenerativeModel, train_data = None, normalize_data : bool = True):
#         self.model = model
#         self.train_data = train_data

#     def __call__(self, *args, **kwargs):
#         raise NotImplementedError("This method should be implemented in subclasses.")