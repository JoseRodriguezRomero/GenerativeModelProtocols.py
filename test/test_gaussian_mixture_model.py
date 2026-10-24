import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot


def default_gaussian_mixture_model_test():
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
        self.assertIsInstance(default_gaussian_mixture_model_test(), gen_prot.GaussianMixtureModel)
        self.assertIsInstance(gen_prot.default_gaussian_mixture_model(2, 2), gen_prot.GaussianMixtureModel)

    def test_printing(self):
        self.assertIsNone(print(default_gaussian_mixture_model_test()))
        self.assertTrue(len(str(default_gaussian_mixture_model_test())) > 0)

    def test_getters(self):
        model = default_gaussian_mixture_model_test()
        self.assertIsInstance(model.log_variance, np.ndarray)
        self.assertIsInstance(model.means, np.ndarray)
        self.assertIsInstance(model.predictor_network, gen_prot.Chain)
        self.assertIsInstance(model.weights, np.ndarray)

    def test_reassign(self):
        model = default_gaussian_mixture_model_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0