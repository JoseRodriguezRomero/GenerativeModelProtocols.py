# Beta Annealing

Recall that the objective function used for training is an Evidence Lower
Bound (ELBO) of the log-likelihood of the observed training data, obtained using
Jensen's inequality. The objective function used for training a
Variational Autoencoder (VAE) consists of two components: a reconstruction
loss term and a Kullback-Leibler (KL) divergence term. In other words:
```{math}
    \text{ELBO}_{\phi, \theta} = \mathbb{E}_{q_\phi \left( z \left| x \right.
        \right)} \left[ \log \left( p_\theta \left( x | z \right) \right)
        \right] - D_\text{KL} \left( q_\phi \left( z \left| x \right. \right)
        \ \lVert \ p(z) \right).
```

A common issue encountered when training VAEs with this ELBO is posterior
collapse, where the decoder ignores the latent variables and the encoder ignores
the input, leading to $q_\phi \left( z \left| x \right. \right) \approx p(z)$.
A widely adopted solution is to introduce a hyperparameter $\beta$ that balances
the reconstruction loss against the KL divergence term, modifying the objective
to:
```{math}
    \text{ELBO}_{\phi, \theta} = \mathbb{E}_{q_\phi \left( z \left| x \right.
        \right)} \left[ \log \left( p_\theta \left( x | z \right) \right)
        \right] - \beta \ D_\text{KL} \left( q_\phi \left( z \left| x \right.
        \right) \ \lVert \ p(z) \right).
```

Moreover, another strategy often used when training VAEs is to use an annealing
schedule for the values of $\beta$ {cite:p}`Chunyuan2019`, where, for example,
one could begin with $\beta = 0$ and gradually increase it to a target value,
such as $\beta = 0.1$.

This module offers both the option to train a VAE with either a fixed value of
$\beta$, or establish some user-defined schedule. Suppose then that we have some
VAE model
```pycon
>>> protocol
GenerativeModelProtocol:
training_data      = 2×5000 Matrix{Float64}
mean_training_data = Tuple{Float64, Float64}
var_training_data  = Tuple{Float64, Float64}
model              = GenerativeModelProtocols.VariationalAutoencoder
```
Since the `protocol.model` is a `gen_prot.VariationalAutoencoder`, we can,
for example, train our VAE model with $\beta = 0.1$ by invoking
```
>>> _ = protocol.train(beta = 0.1, batchsize = 256, epochs = 50)
Training VAE... (β = 0.1)
Epoch       1 | Mean ELBO:  -2.93258399E-01
Epoch       5 | Mean ELBO:  -2.88559973E-01
Epoch      10 | Mean ELBO:  -2.89316356E-01
Epoch      15 | Mean ELBO:  -2.91302025E-01
Epoch      20 | Mean ELBO:  -2.90796936E-01
Epoch      25 | Mean ELBO:  -2.92370260E-01
Epoch      30 | Mean ELBO:  -2.89844930E-01
Epoch      35 | Mean ELBO:  -2.89488167E-01
Epoch      40 | Mean ELBO:  -2.90597558E-01
Epoch      45 | Mean ELBO:  -2.87822068E-01
Epoch      50 | Mean ELBO:  -2.90508896E-01
Training complete!
Training took 47 seconds.
```
Likewise, we can continue training our VAE model with a different $\beta$, in
this example $\beta = 0.5$, for another 80 epochs by simply invoking
```pycon
>>> _ = protocol.train(beta = 0.5, batchsize = 256, epochs = 80)
Training VAE... (β = 0.5)
Epoch       1 | Mean ELBO:  -1.08569384E+00
Epoch       5 | Mean ELBO:  -9.01392102E-01
Epoch      10 | Mean ELBO:  -8.83189201E-01
Epoch      15 | Mean ELBO:  -8.65078568E-01
Epoch      20 | Mean ELBO:  -8.50186944E-01
Epoch      25 | Mean ELBO:  -8.44886959E-01
Epoch      30 | Mean ELBO:  -8.34774792E-01
Epoch      35 | Mean ELBO:  -8.32411468E-01
Epoch      40 | Mean ELBO:  -8.34093928E-01
Epoch      45 | Mean ELBO:  -8.31691921E-01
Epoch      50 | Mean ELBO:  -8.29780698E-01
Epoch      55 | Mean ELBO:  -8.25851440E-01
Epoch      60 | Mean ELBO:  -8.37692380E-01
Epoch      65 | Mean ELBO:  -8.28985393E-01
Epoch      70 | Mean ELBO:  -8.30140412E-01
Epoch      75 | Mean ELBO:  -8.21664155E-01
Epoch      80 | Mean ELBO:  -8.23872685E-01
Training complete!
Training took 49 seconds.
```