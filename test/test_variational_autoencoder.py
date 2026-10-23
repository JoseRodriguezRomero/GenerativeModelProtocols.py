import unittest
import GenerativeModelProtocols as gen_prot


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
        model = default_variational_autoencoder_test()
        self.assertIsNotNone(model)

        model = gen_prot.default_variational_autoencoder(2, 2, 3)
        self.assertIsNotNone(model)

    def test_printing(self):
        self.assertIsNone(print(default_variational_autoencoder_test()))
        self.assertTrue(len(str(default_variational_autoencoder_test)) > 0)

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