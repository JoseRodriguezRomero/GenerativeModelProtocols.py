import unittest
import GenerativeModelProtocols as gen_prot


def test_default_velocity_field():
    return gen_prot.Chain(
        gen_prot.Dense(3, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 2),
    )


def test_default_normalizing_flow():
    return gen_prot.NormalizingFlow(
        velocity_field=test_default_velocity_field()
    )


class TestNormalizingFlow(unittest.TestCase):
    def test_constructor(self):
        model = test_default_normalizing_flow()
        self.assertIsNotNone(model)

        model = gen_prot.default_normalizing_flow(2)
        self.assertIsNotNone(model)

    def test_printing(self):
        print(test_default_normalizing_flow())
        self.assertEqual(True, True)

