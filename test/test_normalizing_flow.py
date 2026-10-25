import unittest
import numpy as np
import GenerativeModelProtocols as gen_prot

from auxiliary_tests import print_model_test_assert
from auxiliary_tests import model_train_test


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
        self.assertIsInstance(default_normalizing_flow_test(), gen_prot.NormalizingFlow)
        self.assertIsInstance(gen_prot.default_normalizing_flow(2), gen_prot.NormalizingFlow)

    def test_printing(self):
        print_model_test_assert(self, default_normalizing_flow_test())

    def test_getters(self):
        model = default_normalizing_flow_test()
        self.assertIsInstance(model.velocity_field, gen_prot.Chain)

        protocol = gen_prot.GenerativeModelProtocol(model = model)
        self.assertIsInstance(protocol.input_size, int)
        self.assertIsInstance(protocol.latent_size, int)
        self.assertIsInstance(protocol.normalize_data, bool)
        self.assertIsInstance(protocol.training_data, np.ndarray)
        self.assertIsInstance(protocol.model, gen_prot.NormalizingFlow)

    def test_reassign(self):
        model = default_normalizing_flow_test()
        with self.assertRaises(BaseException):
            model._jl_model = 0

    def test_train(self):
        model_train_test(self, gen_prot.default_normalizing_flow(2))

