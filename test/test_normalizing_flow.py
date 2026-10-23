import unittest
import GenerativeModelProtocols as gen_prot


def default_velocity_field_test():
    return gen_prot.Chain(
        gen_prot.Dense(3, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 5, gen_prot.ActivationFunction.RELU),
        gen_prot.Dense(5, 2),
    )


def default_normalizing_flow_test():
    return gen_prot.NormalizingFlow(
        velocity_field = default_velocity_field_test()
    )


class TestNormalizingFlow(unittest.TestCase):
    def test_constructor(self):
        model = default_normalizing_flow_test()
        self.assertIsNotNone(model)

        model = gen_prot.default_normalizing_flow(2)
        self.assertIsNotNone(model)

    def test_printing(self):
        self.assertIsNone(print(default_normalizing_flow_test()))
        self.assertTrue(len(str(default_normalizing_flow_test())) > 0)

    def test_getters(self):
        model = default_normalizing_flow_test()

        self.assertIsInstance(model.velocity_field, gen_prot.Chain)

