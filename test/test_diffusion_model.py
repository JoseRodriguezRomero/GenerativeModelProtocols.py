import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot

from test_tabular_denoiser import default_tabular_denoiser_test


def default_diffusion_model_test():
    return gen_prot.DiffusionModel(
        noise_schedule = np.array([0.01, 0.02, 0.03, 0.04]),
        denoiser_model = default_tabular_denoiser_test(),
    )


class TestDiffusionModel(unittest.TestCase):
    def test_constructor(self):
        model = default_diffusion_model_test()
        self.assertIsNotNone(model)

        model = gen_prot.default_diffusion_model(2, 5)
        self.assertIsNotNone(model)

    def test_printing(self):
        self.assertIsNone(print(default_diffusion_model_test()))
        self.assertTrue(len(str(default_diffusion_model_test())) > 0)

    def test_getters(self):
        model = default_diffusion_model_test()

        self.assertIsInstance(model.denoiser_model, gen_prot.TabularDenoiser)
        self.assertIsInstance(model.noise_schedule, np.ndarray)

