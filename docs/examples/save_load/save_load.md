# Saving and Loading Models

This module provides convenient functionality for saving and loading trained
generative models in the [HDF5](https://hdfgroup.org) file format.
[HDF5](https://hdfgroup.org) was specifically chosen to maximize
interoperability. This ensures your trained models remain accessible and usable
across other software solutions, even if those systems do not use this module or
the python programming language at all.

To access this functionality, one only needs to load the Julia modules
[`FileIO.jl`](https://github.com/juliaio/fileio.jl) and
[`HDF5.jl`](https://github.com/JuliaIO/HDF5.jl), which can be done via this
python wrapper with
```pycon
>>> import generative_model_protocols as gen_prot
>>> gen_prot.load_save_load_extension()
```

finally, we can save an already trained model
```pycon
>>> protocol
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.DiffusionModel
>>> protocol.model
GenerativeModelProtocols.DiffusionModel:
β              = NTuple{250, Float64}
denoiser_model = GenerativeModelProtocols.TabularDenoiser
```
to a file named `file_name.h5` with a single function call
```pycon
>>> gen_prot.save(protocol, "file_name.h5")
```

```{image} ../../assets/examples/save_load/hdfview_screenshot.png
:alt: Description
:class: transparent-bg
```

Likewise, we can load our saved model again with a single function call
```pycon
>>> loaded_protocol, loaded_metadata = gen_prot.load("file_name.h5")
>>> loaded_protocol
GenerativeModelProtocol:
training_data      = 2×0 Matrix{Float32}
mean_training_data = Tuple{Float32, Float32}
var_training_data  = Tuple{Float32, Float32}
model              = GenerativeModelProtocols.DiffusionModel
>>> loaded_protocol.model
GenerativeModelProtocols.DiffusionModel:
β              = NTuple{250, Float64}
denoiser_model = GenerativeModelProtocols.TabularDenoiser
```
It is worth pointing out that saved models/protocols are stripped of their
training data. Thus, to continue training a saved model, `train_data` must be
supplied separately by the user with
```pycon
>>> loaded_protocol, loaded_metadata = gen_prot.load("file_name.h5", train_data)
>>> loaded_protocol
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.DiffusionModel
>>> _ = loaded_protocol.train(epochs = 50)
Training Diffusion Model...
Epoch       1 | Mean MSE Loss:   4.56868857E-01
Epoch       5 | Mean MSE Loss:   4.59564358E-01
Epoch      10 | Mean MSE Loss:   4.72298145E-01
Epoch      15 | Mean MSE Loss:   4.59782302E-01
Epoch      20 | Mean MSE Loss:   4.61032003E-01
Epoch      25 | Mean MSE Loss:   4.63341594E-01
Epoch      30 | Mean MSE Loss:   4.40896928E-01
Epoch      35 | Mean MSE Loss:   4.69295681E-01
Epoch      40 | Mean MSE Loss:   4.48327363E-01
Epoch      45 | Mean MSE Loss:   4.60119516E-01
Epoch      50 | Mean MSE Loss:   4.60467845E-01
Training complete!
Training took 52 seconds.
```

## HDF5 Group Names

By default the main group name used for all saved `GenerativeModelProtocol`
instances is `generative_model_protocol`, the default group name for the trained
parameters of its generative model is `generative_model` and the default group
name for all its metadata is `metadata`. However, users can easily modify all or
some of these names by simply using the corresponding key word arguments.

For example, we can save our example protocol with customized HDF5 group names
as
```pycon
>>> gen_prot.save(protocol, "saved_protocol.h5",
...    main_group_name             = "customized_protocol",
...    metadata_group_name         = "customized_metadata",
...    generative_model_group_name = "customized_generative_model"
... )
```

```{image} ../../assets/examples/save_load/hdfview_screenshot_group_names.png
:alt: Description
:class: transparent-bg
```

## User Metadata

Users can easily include additional metadata to any saved instance of a
`GenerativeModelProtocol` via the following keyword argument
```pycon
>>> metadata = {
...     "A": 5.0,
...     "B": "32",
...     "C": [1,2]
... }
>>> gen_prot.save(protocol, "saved_protocol.h5",
...    main_group_name             = "customized_protocol",
...    metadata_group_name         = "customized_metadata",
...    generative_model_group_name = "customized_generative_model",
...    metadata                    = metadata
... )
```

```{image} ../../assets/examples/save_load/hdfview_screenshot_metadata.png
:alt: Description
:class: transparent-bg
```

Finally, users can retrieve the saved protocol and metadata with

```pycon
>>> saved_protocol, metadata = gen_prot.load("saved_protocol.h5",
...    main_group_name             = "customized_protocol",
...    metadata_group_name         = "customized_metadata",
...    generative_model_group_name = "customized_generative_model"
... )
>>> saved_protocol
GenerativeModelProtocol:
training_data      = 2×0 Matrix{Float32}
mean_training_data = Tuple{Float32, Float32}
var_training_data  = Tuple{Float32, Float32}
model              = GenerativeModelProtocols.DiffusionModel
>>> metadata
{'A': 5.0, 'B': '32', 'C': [1, 2]}
```
or, to continue training, with
```pycon
>>> saved_protocol, metadata = gen_prot.load("saved_protocol.h5", train_data,
...    main_group_name             = "customized_protocol",
...    metadata_group_name         = "customized_metadata",
...    generative_model_group_name = "customized_generative_model"
... )
>>> saved_protocol
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.DiffusionModel
>>> metadata
{'A': 5.0, 'B': '32', 'C': [1, 2]}
```