import unittest
import GenerativeModelProtocols as gen_prot

class TestOptimisers(unittest.TestCase):
    def test_descent(self):
        optimiser = gen_prot.Descent()
        self.assertIsInstance(optimiser.eta, float)

    def test_momentum(self):
        optimiser = gen_prot.Momentum()
        self.assertIsInstance(optimiser.eta, float)
        self.assertIsInstance(optimiser.rho, float)

    def test_nesterov(self):
        optimiser = gen_prot.Nesterov()
        self.assertIsInstance(optimiser.eta, float)
        self.assertIsInstance(optimiser.rho, float)

    def test_rprop(self):
        optimiser = gen_prot.Rprop()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(ell, float) for ell in optimiser.ell)
        )
        self.assertTrue(
            all(isinstance(gamma, float) for gamma in optimiser.gamma)
        )

    def test_rmsprop(self):
        optimiser = gen_prot.RMSProp()
        self.assertIsInstance(optimiser.eta, float)
        self.assertIsInstance(optimiser.rho, float)
        self.assertIsInstance(optimiser.epsilon, float)
        self.assertIsInstance(optimiser.centred, bool)

    def test_adam(self):
        optimiser = gen_prot.Adam()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_radam(self):
        optimiser = gen_prot.RAdam()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_adamax(self):
        optimiser = gen_prot.AdaMax()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_oadam(self):
        optimiser = gen_prot.OAdam()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_adagrad(self):
        optimiser = gen_prot.AdaGrad()
        self.assertIsInstance(optimiser.eta, float)
        self.assertIsInstance(optimiser.epsilon, float)

    def test_amsgrad(self):
        optimiser = gen_prot.AMSGrad()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_nadam(self):
        optimiser = gen_prot.NAdam()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_adamw(self):
        optimiser = gen_prot.AdamW()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.lambda_, float)
        self.assertIsInstance(optimiser.epsilon, float)
        self.assertIsInstance(optimiser.couple, bool)

    def test_adabelief(self):
        optimiser = gen_prot.AdaBelief()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )
        self.assertIsInstance(optimiser.epsilon, float)

    def test_lion(self):
        optimiser = gen_prot.Lion()
        self.assertIsInstance(optimiser.eta, float)
        self.assertTrue(
            all(isinstance(beta, float) for beta in optimiser.beta)
        )

