import unittest
import GenerativeModelProtocols as gen_prot


def default_tabular_denoiser_test():
    return gen_prot.TabularDenoiser(
        T=4,
        time_embedding_mlp=gen_prot.Chain(
            gen_prot.Dense(4, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        ),
        input_projection=gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
        residual_layers=(
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        ),
        time_projection_layers=(
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        ),
        output_projection=gen_prot.Dense(5, 2),
    )


class TestTabularDenoiser(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(default_tabular_denoiser_test(), gen_prot.TabularDenoiser)
        self.assertIsInstance(gen_prot.default_tabular_denoiser(2), gen_prot.TabularDenoiser)

    def test_printing(self):
        self.assertIsNone(print(default_tabular_denoiser_test()))
        self.assertTrue(len(str(default_tabular_denoiser_test())) > 0)

    def test_getters(self):
        model = default_tabular_denoiser_test()
        self.assertIsInstance(model.T, int)
        self.assertIsInstance(model.time_embedding_mlp, gen_prot.Chain)
        self.assertIsInstance(model.input_projection, gen_prot.Dense)
        self.assertIsInstance(model.residual_layers, tuple)
        self.assertTrue(
            all(isinstance(layer, gen_prot.Dense) for layer in model.residual_layers)
        )
        self.assertIsInstance(model.time_projection_layers, tuple)
        self.assertTrue(
            all(isinstance(layer, gen_prot.Dense) for layer in model.time_projection_layers)
        )
        self.assertIsInstance(model.output_projection, gen_prot.Dense)
        self.assertIsInstance(model.max_period, float)

    def test_reassign(self):
        dense = default_tabular_denoiser_test()
        with self.assertRaises(BaseException):
            dense._jl_denoiser = 0

