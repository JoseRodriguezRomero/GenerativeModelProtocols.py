import GenerativeModelProtocols as gen_prot
import matplotlib.pyplot as plt
import numpy as np


def make_data(num_samples):
    t = (2.0 * np.pi * np.random.rand(num_samples))
    
    x_noise = (0.05 * np.random.randn(num_samples))
    y_noise = (0.05 * np.random.randn(num_samples))
    
    x = np.cos(1.0 * t) + x_noise
    y = np.sin(2.0 * t) + y_noise
    
    return np.vstack((x, y))


num_samples = 5000
train_data = make_data(num_samples)

k = [15, 65, 120]
models = [gen_prot.default_gaussian_mixture_model(2, ki) for ki in k]
protocols = [gen_prot.GenerativeModelProtocol(model, train_data) for model in models]

for i in range(len(protocols)):
    protocols[i].train(
        batchsize = 256,
        epochs    = 3500,
        optimiser = gen_prot.Adam(eta = 1.0E-3, beta = (0.95,0.999))
    )

## Compare Synthetic Data

synthetic_data_1 = protocols[0](num_samples)
synthetic_data_2 = protocols[1](num_samples)
synthetic_data_3 = protocols[2](num_samples)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows   = 2, 
    ncols   = 2, 
    figsize = (8, 5),
    dpi     = 100
)


def plot_scatter_data(ax, data, title):
    ax.scatter(
        data[0,:], data[1,:],
        linewidths = 0.8,
        edgecolors = 'black',
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)


plot_scatter_data(ax1, train_data, "Training data")
plot_scatter_data(ax2, synthetic_data_1, f"Synthetic data (k = {k[0]:1})")
plot_scatter_data(ax3, synthetic_data_2, f"Synthetic data (k = {k[1]:1})")
plot_scatter_data(ax4, synthetic_data_3, f"Synthetic data (k = {k[2]:1})")

plt.tight_layout(w_pad = 2.0)
fig.savefig("intro_example.svg", dpi=300)
plt.show()

## Compare Distributions

n_data_hist2d = 100000
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    nrows   = 2,
    ncols   = 2, 
    figsize = (8, 5),
    dpi     = 100
)


def plot_hist2d_dat(ax, data, title):
    H, xedges, yedges = np.histogram2d(
        data[0, :], data[1, :],
        bins=100
    )
    ax.pcolormesh(
        xedges, yedges, H.T,
        shading="flat",
        cmap="viridis",
        rasterized=True
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)



plot_hist2d_dat(ax1, make_data(n_data_hist2d), "Exact Data PDF\n2D Histogram")
plot_hist2d_dat(ax2, protocols[0](n_data_hist2d), f"GMM PDF (k = {k[0]:1})\n2D Histogram")
plot_hist2d_dat(ax3, protocols[1](n_data_hist2d), f"GMM PDF (k = {k[1]:1})\n2D Histogram")
plot_hist2d_dat(ax4, protocols[2](n_data_hist2d), f"GMM PDF (k = {k[2]:1})\n2D Histogram")

plt.tight_layout(w_pad = 2.0)
fig.savefig("intro_example_density_comp.svg", dpi=300)
plt.show()

