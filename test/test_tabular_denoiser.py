import unittest
import GenerativeModelProtocols as gen_prot


def test_default_tabular_denoiser():
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
        model = test_default_tabular_denoiser()
        self.assertIsNotNone(model)

        model = gen_prot.default_tabular_denoiser(2)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_tabular_denoiser())
        self.assertEqual(True, True)

