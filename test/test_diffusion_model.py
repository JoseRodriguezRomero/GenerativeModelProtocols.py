import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot

from test_tabular_denoiser import default_tabular_denoiser_test
from auxiliary_tests import model_train_test
from auxiliary_tests import print_model_test_assert


def default_diffusion_model_test() -> gen_prot.DiffusionModel:
    return gen_prot.DiffusionModel(
        noise_schedule = np.array([0.01, 0.02, 0.03, 0.04]),
        denoiser_model = default_tabular_denoiser_test(),
    )


class TestDiffusionModel(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(default_diffusion_model_test(), gen_prot.DiffusionModel)
        self.assertIsInstance(gen_prot.default_diffusion_model(2, 5), gen_prot.DiffusionModel)

    def test_printing(self):
        print_model_test_assert(self, default_diffusion_model_test())

    def test_getters(self):
        model = default_diffusion_model_test()
        self.assertIsInstance(model.denoiser_model, gen_prot.TabularDenoiser)
        self.assertIsInstance(model.noise_schedule, np.ndarray)

        protocol = gen_prot.GenerativeModelProtocol(model = model)
        self.assertIsInstance(protocol.input_size, int)
        self.assertIsInstance(protocol.latent_size, int)
        self.assertIsInstance(protocol.normalize_data, bool)
        self.assertIsInstance(protocol.training_data, np.ndarray)

    def test_reassign(self):
        model = default_diffusion_model_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0

    def test_train(self):
        model_train_test(self, gen_prot.default_diffusion_model(2, 50))

