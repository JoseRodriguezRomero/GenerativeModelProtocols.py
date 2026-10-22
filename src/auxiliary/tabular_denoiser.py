from juliacall import Main as jl

class TabularDenoiser:
    def __init__(self, _jl_model):
        self._jl_model = _jl_model

    def __repr__(self):
        jl.display(self._jl_model)
        return ""

    def __str__(self):
            jl.display(self._jl_model)
            return ""

