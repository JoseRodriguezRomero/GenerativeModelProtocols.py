import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot


def test_default_gaussian_mixture_model():
    return gen_prot.GaussianMixtureModel(
        log_variance = np.zeros((2, 2)),
        means = np.zeros((2, 2)),
        predictor_network = gen_prot.Chain(
            gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 2, gen_prot.ActivationFunction.RELU),
        ),
        weights=np.array([0.5, 0.5]),
    )


class TestGaussianMixtureModel(unittest.TestCase):
    def test_constructor(self):
        model = test_default_gaussian_mixture_model()
        self.assertIsNotNone(model)

        model = gen_prot.default_gaussian_mixture_model(2, 2)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_gaussian_mixture_model())
        self.assertEqual(True, True)


