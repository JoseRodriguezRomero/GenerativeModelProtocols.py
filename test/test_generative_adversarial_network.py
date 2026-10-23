import unittest
import GenerativeModelProtocols as gen_prot

from test_variational_autoencoder import default_variational_autoencoder_test


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
        self.assertIsNone(print(default_generative_adversarial_network_test()))
        self.assertTrue(len(str(default_generative_adversarial_network_test())) > 0)

    def test_getters(self):
        model = default_generative_adversarial_network_test()
        self.assertIsInstance(model.discriminator, gen_prot.Chain)
        self.assertIsInstance(model.vae_model, gen_prot.VariationalAutoencoder)