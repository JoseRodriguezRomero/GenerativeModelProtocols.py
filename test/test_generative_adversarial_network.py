import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot

from test_variational_autoencoder import default_variational_autoencoder_test
from auxiliary_tests import print_model_test_assert
from auxiliary_tests import model_train_test


def default_discriminator_test():
    return gen_prot.Chain(
        gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 1, gen_prot.ActivationFunction.RELU),
    )


def default_generative_adversarial_network_test():
    return gen_prot.GenerativeAdversarialNetwork(
        vae_model     = default_variational_autoencoder_test(),
        discriminator = default_discriminator_test()
    )


class TestGenerativeAdversarialNetwork(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(default_generative_adversarial_network_test(), gen_prot.GenerativeAdversarialNetwork)
        self.assertIsInstance(gen_prot.default_generative_adversarial_network(2, 2, 2), gen_prot.GenerativeAdversarialNetwork)

    def test_printing(self):
        print_model_test_assert(self, default_generative_adversarial_network_test())

    def test_getters(self):
        model = default_generative_adversarial_network_test()
        self.assertIsInstance(model.discriminator, gen_prot.Chain)
        self.assertIsInstance(model.vae_model, gen_prot.VariationalAutoencoder)

        protocol = gen_prot.GenerativeModelProtocol(model = model)
        self.assertIsInstance(protocol.input_size, int)
        self.assertIsInstance(protocol.latent_size, int)
        self.assertTrue(
            all(isinstance(normalize_data, bool) for normalize_data in protocol.normalize_data)
        )
        self.assertIsInstance(protocol.training_data, np.ndarray)
        self.assertIsInstance(protocol.model, gen_prot.GenerativeAdversarialNetwork)

    def test_reassign(self):
        model = default_generative_adversarial_network_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0

    def test_train(self):
        model_train_test(self, gen_prot.default_generative_adversarial_network(2, 2, 2), beta = 0.1)

    def test_data_synthetization(self):
        data_dims = 2
        latent_dims = 3
        num_latent_layers = 2

        model = gen_prot.default_generative_adversarial_network(data_dims, latent_dims, num_latent_layers)
        protocol = gen_prot.GenerativeModelProtocol(model = model, normalize_data = False)

        x_test = np.random.randn(data_dims, 10)
        z_test = np.random.randn(latent_dims, 10)

        self.assertIsInstance(protocol(20), np.ndarray)
        self.assertTrue(x_test.shape == protocol.decode(z_test).shape)
        self.assertTrue(z_test.shape == protocol.encode(x_test).shape)

