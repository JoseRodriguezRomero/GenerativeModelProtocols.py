import unittest
import GenerativeModelProtocols as gen_prot


def make_default_test_chain():
    layer_1 = gen_prot.Dense(2, 5, gen_prot.ActivationFunction.ELU)
    layer_2 = gen_prot.Dense(5, 5, gen_prot.ActivationFunction.ELU)
    layer_3 = gen_prot.Dense(5, 2, gen_prot.ActivationFunction.ELU)
    chain = gen_prot.Chain(layer_1, layer_2, layer_3)
    return chain


class TestChain(unittest.TestCase):
    def test_constructor(self):
        chain = make_default_test_chain()
        self.assertIsNotNone(chain)

    def test_printing(self):
        print(make_default_test_chain())
        self.assertEqual(True, True)

