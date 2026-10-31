import numpy as np
import GenerativeModelProtocols as gen_prot

def make_data(num_samples):
    t = (2.0 * np.pi) * np.random.rand(num_samples)
    
    x_noise = 0.05 * np.random.randn(num_samples)
    y_noise = 0.05 * np.random.randn(num_samples)
    
    x = np.cos(1.0 * t) + x_noise
    y = np.sin(2.0 * t) + y_noise
    
    return np.vstack((x, y))


def _print_model_test_assert(env, obj):
    env.assertIsNone(print(obj))
    env.assertTrue(len(str(obj)) > 0)


def print_model_test_assert(env, model):
    protocol = gen_prot.GenerativeModelProtocol(
        model = model,
        training_data = make_data(50)
    )
    _print_model_test_assert(env, model)
    _print_model_test_assert(env, protocol)


def model_train_test(env, model, **kwargs):
    training_data = make_data(50)
    protocol = gen_prot.GenerativeModelProtocol(
        model         = model,
        training_data = training_data
    )
    optimiser = gen_prot.default_optimiser(protocol)
    env.assertIsInstance(protocol, gen_prot.GenerativeModelProtocol)
    train_log = protocol.train(epochs = 20, optimiser = optimiser, **kwargs)
    env.assertIsNotNone(train_log)

