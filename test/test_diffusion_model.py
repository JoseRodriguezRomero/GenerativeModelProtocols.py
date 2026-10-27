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
        self.assertTrue(
            all(isinstance(normalize_data, bool) for normalize_data in protocol.normalize_data)
        )
        self.assertIsInstance(protocol.training_data, np.ndarray)
        self.assertIsInstance(protocol.model, gen_prot.DiffusionModel)

    def test_reassign(self):
        model = default_diffusion_model_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0

    def test_train(self):
        model_train_test(self, gen_prot.default_diffusion_model(2, 50))

    def test_data_synthetization(self):
        data_dims = 2
        latent_dims = data_dims

        model = gen_prot.default_diffusion_model(data_dims, 20)
        protocol = gen_prot.GenerativeModelProtocol(model = model, normalize_data = False)

        x_test = np.random.randn(data_dims, 10)
        z_test = np.random.randn(latent_dims, 10)

        self.assertIsInstance(protocol(20), np.ndarray)
        self.assertTrue(x_test.shape == protocol.decode(z_test).shape)
        self.assertTrue(z_test.shape == protocol.encode(x_test).shape)

