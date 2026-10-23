import unittest
import GenerativeModelProtocols as gen_prot

from test_tabular_denoiser import test_default_tabular_denoiser


def test_default_diffusion_model():
    return gen_prot.DiffusionModel(
        noise_schedule=(0.01, 0.02, 0.03, 0.04),
        denoiser_model=test_default_tabular_denoiser(),
    )


class TestDiffusionModel(unittest.TestCase):
    def test_constructor(self):
        model = test_default_diffusion_model()
        self.assertIsNotNone(model)

        model = gen_prot.default_diffusion_model(2, 5)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_diffusion_model())
        self.assertEqual(True, True)

