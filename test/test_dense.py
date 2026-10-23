import unittest
import GenerativeModelProtocols as gen_prot


def make_default_test_layer():
    return gen_prot.Dense(5, 5, gen_prot.ActivationFunction.ELU)


class TestDense(unittest.TestCase):
    def test_constructor(self):
        layer = make_default_test_layer()
        self.assertIsNotNone(layer)

    def test_printing(self):
        print(make_default_test_layer(), flush=True)
        self.assertTrue(True)

