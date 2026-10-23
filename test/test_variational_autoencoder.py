import unittest
import GenerativeModelProtocols as gen_prot


def test_default_vae():
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
        model = test_default_vae()
        self.assertIsNotNone(model)

        model = gen_prot.default_variational_autoencoder(2, 2, 3)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_vae())
        self.assertEqual(True, True)

