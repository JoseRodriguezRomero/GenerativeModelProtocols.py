import unittest
import GenerativeModelProtocols as gen_prot


def make_default_test_chain() -> gen_prot.Chain:
    layer_1 = gen_prot.Dense(2, 5, gen_prot.ActivationFunction.ELU)
    layer_2 = gen_prot.Dense(5, 5, gen_prot.ActivationFunction.ELU)
    layer_3 = gen_prot.Dense(5, 2, gen_prot.ActivationFunction.ELU)
    chain = gen_prot.Chain(layer_1, layer_2, layer_3)
    return chain


class TestChain(unittest.TestCase):
    def test_constructor(self):
        self.assertIsInstance(make_default_test_chain(), gen_prot.Chain)

    def test_printing(self):
        self.assertIsNone(print(make_default_test_chain()))
        self.assertTrue(len(str(make_default_test_chain())) > 0)

    def test_reassign(self):
        chain = make_default_test_chain()
        with self.assertRaises(BaseException):
            chain._jl_chain = 0

