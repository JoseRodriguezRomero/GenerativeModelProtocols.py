import os
import numpy as np
import generative_model_protocols as gen_prot

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


def model_train_test(env, model, device, **kwargs):
    training_data = make_data(50)
    protocol = gen_prot.GenerativeModelProtocol(
        model         = model,
        training_data = training_data
    )
    optimiser = gen_prot.default_optimiser(protocol)
    env.assertIsInstance(protocol, gen_prot.GenerativeModelProtocol)
    train_log = protocol.train(
        epochs    = 20, 
        optimiser = optimiser,
        device    = device,
        **kwargs
    )
    env.assertIsNotNone(train_log)


def model_train_test_cpu(env, model, **kwargs):
    print("Testing model on CPU")
    model_train_test(env, model, device = gen_prot.Device.CPU, **kwargs)


def model_train_test_reactant(env, model, **kwargs):
    print("Testing model on Reactant (CPU backend)")
    model_train_test(env, model, device = gen_prot.Device.REACTANT_CPU, **kwargs)


def save_load_protocol_test(env, model, file_name):
    training_data = make_data(50)
    protocol = gen_prot.GenerativeModelProtocol(
        model         = model,
        training_data = training_data
    )

    gen_prot.load_save_load_exension()
    gen_prot.save(protocol, file_name)

    loaded_protocol = gen_prot.load(file_name, training_data = training_data)
    env.assertIsInstance(loaded_protocol, gen_prot.GenerativeModelProtocol)
    env.assertTrue(type(protocol.model) == type(loaded_protocol.model))

    loaded_protocol = gen_prot.load(file_name)
    env.assertIsInstance(loaded_protocol, gen_prot.GenerativeModelProtocol)
    env.assertTrue(type(protocol.model) == type(loaded_protocol.model))

    os.remove(file_name)

