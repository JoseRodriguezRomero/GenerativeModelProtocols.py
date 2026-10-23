import io
from contextlib import redirect_stdout

from juliacall import Main as jl


class AbstractGenerativeModel:
    def __init__(self, _jl_model):
        self._jl_model = _jl_model

    def __repr__(self):
        output = io.StringIO()
        with redirect_stdout(output):
            jl.display(self._jl_model)
        return output.getvalue().rstrip()

    def __str__(self):
        return self.__repr__()


# class GenerativeModelProtocol:
#     def __new__(self, model : AbstractGenerativeModel, train_data = None, normalize_data : bool = True):
#         self.model = model
#         self.train_data = train_data

#     def __call__(self, *args, **kwargs):
#         raise NotImplementedError("This method should be implemented in subclasses.")