from .data import load_mlati
from matplotlib import pyplot as plt
import numpy as np

def plot_velocity_waveforms(filename=None, xyz=None, t=None, n_samples=30, figsize=(8, 2.5), xticks=(-100, -50, 0, 50, 100)):
    """
    """

    if xyz is None:
        X, y, z = load_mlati(filename)
    else:
        X, y, z = xyz

    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
    if t is None:
        t = np.arange(y.shape[1])

    for ax, label, color in zip(axs, [0, -1, 1], ['0.5', 'C0', 'C1']):
        samples = y[z == label, :]
        index = np.random.choice(np.arange(samples.shape[0]), size=n_samples)
        for yi in samples[index, :]:
            v = yi / 0.005
            ax.plot(t, v, color=color, alpha=0.15, lw=0.8)
        ax.plot(t, samples[index, :].mean(0) / 0.005, color='k', lw=1)

    xyz = (X, y, z)
    axs[0].set_ylabel('Eye velocity (deg/s)')
    for ax in axs:
        ax.set_xlabel('Time (ms)')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    ylim = (
        max(axs[0].get_ylim()) * -1,
        max(axs[0].get_ylim())
    )
    for ax in axs:
        ax.set_ylim(ylim)
        ax.set_xticks(xticks)
    for title, ax in zip(['Not a saccade (z=0)', 'Temporal (z=-1)', 'Nasal (z=1)'], axs):
        ax.set_title(title, fontsize=10)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()

    return fig, axs, xyz