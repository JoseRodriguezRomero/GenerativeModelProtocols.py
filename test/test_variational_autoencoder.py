import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot

from auxiliary_tests import print_model_test_assert
from auxiliary_tests import model_train_test


def default_variational_autoencoder_test():
    encoders = (
        gen_prot.Chain(
            gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 6, gen_prot.ActivationFunction.IDENTITY)
        ),
        gen_prot.Chain(
            gen_prot.Dense(3, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 4, gen_prot.ActivationFunction.IDENTITY)
        )
    )

    decoders = (
        gen_prot.Chain(
            gen_prot.Dense(3, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 2, gen_prot.ActivationFunction.IDENTITY)
        ),
        gen_prot.Chain(
            gen_prot.Dense(2, 5, gen_prot.ActivationFunction.RELU),
            gen_prot.Dense(5, 3, gen_prot.ActivationFunction.IDENTITY)
        )
    )

    return gen_prot.VariationalAutoencoder(
        encoders = encoders,
        decoders = decoders
    )


class TestVariationalAutoencoder(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(default_variational_autoencoder_test(), gen_prot.VariationalAutoencoder)
        self.assertIsInstance(gen_prot.default_variational_autoencoder(2, 2, 3), gen_prot.VariationalAutoencoder)

    def test_printing(self):
        print_model_test_assert(self, default_variational_autoencoder_test())

    def test_getters(self):
        model = default_variational_autoencoder_test()
        self.assertIsInstance(model.encoders, tuple)
        self.assertTrue(
            all(isinstance(encoder, gen_prot.Chain) for encoder in model.encoders)
        )
        self.assertIsInstance(model.decoders, tuple)
        self.assertTrue(
            all(isinstance(decoder, gen_prot.Chain) for decoder in model.decoders)
        )

        protocol = gen_prot.GenerativeModelProtocol(model = model)
        self.assertIsInstance(protocol.input_size, int)
        self.assertIsInstance(protocol.latent_size, int)
        self.assertIsInstance(protocol.normalize_data, bool)
        self.assertIsInstance(protocol.training_data, np.ndarray)
        self.assertIsInstance(protocol.model, gen_prot.VariationalAutoencoder)

    def test_reassign(self):
        model = default_variational_autoencoder_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0

    def test_train(self):
        model_train_test(self, gen_prot.default_variational_autoencoder(2, 2, 2), beta = 0.2)

