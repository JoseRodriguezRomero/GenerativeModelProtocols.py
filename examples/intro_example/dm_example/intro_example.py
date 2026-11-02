import GenerativeModelProtocols as gen_prot
import matplotlib.pyplot as plt
import numpy as np


def make_data(num_samples):
    t = (2.0 * np.pi) * np.random.rand(num_samples)
    
    x_noise = 0.05 * np.random.randn(num_samples)
    y_noise = 0.05 * np.random.randn(num_samples)
    
    x = np.cos(1.0 * t) + x_noise
    y = np.sin(2.0 * t) + y_noise
    
    return np.vstack((x, y))


num_samples = 5000
train_data = make_data(num_samples)

model = gen_prot.default_diffusion_model(2,250)
protocol = gen_prot.GenerativeModelProtocol(model, train_data)
protocol.train(
    batchsize = 256,
    epochs    = 1500,
    optimiser = gen_prot.Adam(eta = 1.0E-3, beta = (0.95,0.999))
)

## Compare Distributions

synthetic_data = protocol(num_samples)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows   = 2, 
    ncols   = 2, 
    figsize = (8, 8),
    dpi     = 100
)


def scatter_data_comp(data, ax, title):
    ax.scatter(
        data[0,:], data[1,:], 
        edgecolors = 'black',
        linewidths = 0.8
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)


def hist2d_data_comp(data, ax):
    ax.hist2d(
        data[0, :], data[1, :],
        bins=100,
        cmap="viridis",
        rasterized=True
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)


hist2d_data_size = 50000

scatter_data_comp(train_data, ax1, "Training Data")
scatter_data_comp(synthetic_data, ax2, "Diffusion Model")
hist2d_data_comp(make_data(hist2d_data_size), ax3)
hist2d_data_comp(protocol(hist2d_data_size), ax4)

plt.tight_layout(w_pad = 2.0)
fig.savefig("intro_example.svg", dpi=300)
plt.show()

## Reconstruction test

test_data = make_data(800)
z_test_data = protocol.encode(test_data)
recon_test_data = protocol.decode(test_data)

fig, ax = plt.subplots()
ax.scatter(
    test_data[0,:], test_data[1,:],
    edgecolors = 'black',
    linewidths = 0.8,
    label = "input"
)

ax.scatter(
    recon_test_data[0,:], recon_test_data[1,:],
    edgecolors = 'black',
    linewidths = 0.8,
    label = "reconstruction"
)

ax.legend(loc='upper center')
plt.tight_layout(w_pad = 2.0)
fig.savefig("intro_example_recon.svg", dpi=300)
plt.show()

## Test latent space

def reference_gaussian(x, mean, var):
    return np.exp(-((x - mean) ** 2.0)/(2.0 * var))/np.sqrt(2.0 * np.pi * var)


def plot_ref_gaussian(ax):
    x = np.linspace(-10.0, 10.0, 1000)
    y = reference_gaussian(x, 0.0, 1.0)
    ax.plot(x, y, lw = 3.5, label = r"$\mathcal{N} (0, 1)$")

test_data = make_data(10000)
z_test_data = protocol.encode(test_data)

fig, (ax1, ax2) = plt.subplots(
    nrows   = 2, 
    ncols   = 1, 
    figsize = (8, 5),
    dpi     = 100, 
    sharex  = True
)

ax1.hist(
    z_test_data[0,:], 
    bins = 25,
    density = True,
    edgecolor='black',
    label = r"$z_1$"
)
plot_ref_gaussian(ax1)
ax1.legend(loc='upper right', ncol = 1)
ax1.set_ylabel("Probability Density")

ax2.hist(
    z_test_data[1,:], 
    bins = 25,
    density = True,
    edgecolor='black',
    label = r"$z_2$"
)
plot_ref_gaussian(ax2)
ax2.legend(loc='upper right', ncol = 1)
ax2.set_ylabel("Probability Density")
ax2.set_xlabel("Latent Variable")

fig.savefig("intro_example_latents.svg", dpi=300)
plt.tight_layout(w_pad = 2.0)
plt.show()

## Test latent space independence
from scipy import stats

test_data = make_data(10000)
z_test_data = protocol.encode(test_data)

pearson_corr, _ = stats.pearsonr(z_test_data[0,:], z_test_data[1,:])
spearman_corr, _ = stats.spearmanr(z_test_data[0,:], z_test_data[1,:])

fig, ax = plt.subplots()
ax.scatter(
    z_test_data[0,:], z_test_data[1,:],
    linewidths = 0.8,
    edgecolors = 'black',
)
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title(f"pearson_corr = {pearson_corr:.4f}\nspearman_corr = {spearman_corr:.4f}")

fig.savefig("intro_example_latents_corr.svg", dpi=300)
plt.tight_layout(w_pad = 2.0)
plt.show()

