import unittest
import GenerativeModelProtocols as gen_prot


def make_default_test_layer() -> gen_prot.Dense:
    return gen_prot.Dense(5, 5, gen_prot.ActivationFunction.ELU)


class TestDense(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(make_default_test_layer(), gen_prot.Dense)

    def test_printing(self):
        self.assertIsNone(print(make_default_test_layer()))
        self.assertTrue(len(str(make_default_test_layer())) > 0)

    def test_reassign(self):
        dense = make_default_test_layer()
        with self.assertRaises(BaseException):
            dense._jl_dense = 0

