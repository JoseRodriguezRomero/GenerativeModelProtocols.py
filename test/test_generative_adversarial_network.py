import unittest
import GenerativeModelProtocols as gen_prot

from test_variational_autoencoder import test_default_vae


def test_default_discriminator():
    return gen_prot.Chain(
        gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 1, gen_prot.ActivationFunction.RELU),
    )


def test_default_generative_adversarial_network():
    return gen_prot.GenerativeAdversarialNetwork(
        vae_model     = test_default_vae(),
        discriminator = test_default_discriminator()
    )


class TestGenerativeAdversarialNetwork(unittest.TestCase):
    def test_constructor(self):
        model = test_default_generative_adversarial_network()
        self.assertIsNotNone(model)

        model = gen_prot.default_generative_adversarial_network(2, 2, 2)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_generative_adversarial_network())
        self.assertEqual(True, True)

