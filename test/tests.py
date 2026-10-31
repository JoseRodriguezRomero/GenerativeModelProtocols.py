import unittest

from test_dense import *
from test_chain import *

from test_tabular_denoiser import *
from test_optimisers import *

from test_variational_autoencoder import *
from test_generative_adversarial_network import *
from test_gaussian_mixture_model import *
from test_diffusion_model import *
from test_normalizing_flow import *

loader = unittest.defaultTestLoader
loader.sortTestMethodsUsing = None

runner = unittest.TextTestRunner(verbosity=0, buffer=False)

if __name__ == "__main__":
    # lux components tests
    print("\nTesting GenerativeModelProtocols.Dense", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestDense))

    print("\nTesting GenerativeModelProtocols.Chain", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestChain))

    # auxiliary tests
    print("\nTesting GenerativeModelProtocols.TabularDenoiser", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestTabularDenoiser))

    print("\nTesting GenerativeModelProtocols.Auxiliary.Optimisers", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestOptimisers))

    # generative model tests
    print("\nTesting GenerativeModelProtocols.VariationalAutoencoder", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestVariationalAutoencoder))

    print("\nTesting GenerativeModelProtocols.GenerativeAdversarialNetwork", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestGenerativeAdversarialNetwork))

    print("\nTesting GenerativeModelProtocols.GaussianMixtureModel", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestGaussianMixtureModel))

    print("\nTesting GenerativeModelProtocols.DiffusionModel", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestDiffusionModel))

    print("\nTesting GenerativeModelProtocols.NormalizingFlow", flush=True)
    runner.run(loader.loadTestsFromTestCase(TestNormalizingFlow))

