from .data import load_mlati
from .mlp import PyTorchMLPRegressor, PyTorchMLPClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def visualize_Xyz(
    filename,
    n_cols=11,
    figsize=(9, 4),
    cmap='pink_r',
    X_vrange=(-7, 7),
    y_vrange=(-1500, 1500),
    xyz=None
    ):
    """
    """

    if xyz is None:
        X, y, z = load_mlati(filename, X_bincounts=(20, 20))
    else:
        X, y, z = xyz
    n_bins = 40
    n_splits = X.shape[1] // n_bins
    splits = np.split(X, n_splits, axis=1)
    height_ratios = (
        np.sum(z == 0),
        np.sum(z == 1),
        np.sum(z == 2)
    )
    if n_cols is None:
        n_cols = n_splits
    fig, axs = plt.subplots(
        ncols=n_cols + 2,
        nrows=3,
        gridspec_kw={'height_ratios': height_ratios}
    )
    index = np.argsort([
        sp[z != 0].max(1).mean() for sp in splits
    ])[::-1]
    splits = [splits[i] for i in index]
    for j, sp in enumerate(splits[:n_cols]):
        for i, l in enumerate([0, 1, 2]):
            m = sp[z == l, :]
            m = gaussian_filter(m, 1.5)
            vmin, vmax = X_vrange
            axs[i, j].pcolor(
                m,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
    cmap_ = plt.get_cmap(cmap, 3)
    for i, l in enumerate([0, 1, 2]):
        m = y[z == l] / 0.002
        vmin, vmax = y_vrange
        axs[i, n_cols].pcolor(m, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[i, n_cols + 1].add_patch(
            plt.Rectangle([0, 0], 1, 1, color=cmap_(i))
        )
        axs[i, n_cols + 1].set_xlim([0, 1])
        axs[i, n_cols + 1].set_ylim([0, 1])
    for ax in axs.flatten():
        for sp in ('top', 'right', 'bottom', 'left'):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0, 0].set_ylabel('0', rotation=0, labelpad=15)
    axs[1, 0].set_ylabel('1', rotation=0, labelpad=15)
    axs[2, 0].set_ylabel('2', rotation=0, labelpad=15)
    fig.supylabel('Label', fontsize=10)
    axs[0, 5].set_title('X', fontsize=10)
    axs[0, -2].set_title('y', fontsize=10)
    axs[0, -1].set_title('z', fontsize=10)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.1)

    return fig, axs, (X, y, z)

def visualize_mlp_classifier_performance(
    data,
    cmap='Blues',
    figsize=(6.5, 3)
    ):
    """
    """

    #
    if type(data) == str:
        X, y, z = load_mlati(data)
    elif type(data) == tuple:
        X, y, z = data

    #
    clf_pt = PyTorchMLPClassifier()
    clf_sk = MLPClassifier(solver='adam', max_iter=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2)
    clf_pt.fit(X_train, y_train)
    clf_sk.fit(X_train, y_train)
    y_pred_pt = clf_pt.predict(X_test)
    y_pred_sk = clf_sk.predict(X_test)
    cm_pt = confusion_matrix(y_test, y_pred_pt)
    cm_sk = confusion_matrix(y_test, y_pred_sk)

    #
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    im_pt = axs[0].imshow(cm_pt / cm_pt.sum(0).reshape(1, -1), cmap=cmap, vmin=0, vmax=1)
    im_sk = axs[1].imshow(cm_sk / cm_pt.sum(0).reshape(1, -1), cmap=cmap, vmin=0, vmax=1)

    #
    for (i, j), v in np.ndenumerate(cm_pt):
        axs[0].text(j, i, v, color='k', fontsize=10, horizontalalignment='center',
        verticalalignment='center')
    for (i, j), v in np.ndenumerate(cm_sk):
        axs[1].text(j, i, v, color='k', fontsize=10, horizontalalignment='center',
        verticalalignment='center')   

    #
    axs[0].set_ylabel('True labels')
    for ax in axs:
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['0', '1', '2'])
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(['2', '1', '0'])
        ax.set_xlabel('Predicted labels')
    axs[0].set_title('PyTorch', fontsize=10)
    axs[1].set_title('scikit-learn', fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.3, 0.03, 0.5])
    fig.colorbar(im_pt, cax=cax)
    cax.set_ylabel('Frac. of predictions (per class)', rotation=270, labelpad=20)
    cax.yaxis.set_label_position('right')  

    return fig, axs, clf_pt, clf_sk