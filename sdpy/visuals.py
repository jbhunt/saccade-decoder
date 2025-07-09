from .data import load_mlati
from .mlp import PyTorchMLPRegressor, PyTorchMLPClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, root_mean_squared_error, r2_score, make_scorer
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def visualize_Xyz(
    filename,
    n_cols=11,
    figsize=(9, 4),
    cmap='pink_r',
    X_vrange=(-8, 8),
    y_vrange=(-1500, 1500),
    X_sigma=1.5,
    y_sigma=1.5,
    xyz=None,
    **load_mlati_kwargs
    ):
    """
    """

    _load_mlati_kwargs = {
        'X_bincounts': (20, 20),
        'X_binsize': 0.01,
        'y_bincounts': (25, 45),
        'y_binsize': 0.002,
    }
    _load_mlati_kwargs.update(load_mlati_kwargs)
    if xyz is None:
        X, y, z = load_mlati(filename, **load_mlati_kwargs)
    else:
        X, y, z = xyz

    # Initialize plot
    n_bins = np.sum(_load_mlati_kwargs['X_bincounts'])
    n_units = X.shape[1] // n_bins
    height_ratios = (
        np.sum(z == 0),
        np.sum(z == 1),
        np.sum(z == 2)
    )
    width_ratios = (
        n_bins * n_cols, # * _load_mlati_kwargs['X_binsize'],
        np.sum(_load_mlati_kwargs['y_bincounts']), # * _load_mlati_kwargs['y_binsize'],
        5 # 0.05,
    )
    fig, axs = plt.subplots(
        ncols=3,
        nrows=3,
        gridspec_kw={'height_ratios': height_ratios, 'width_ratios': width_ratios}
    )

    # Reorder X
    splits = np.split(X, n_units, axis=1)
    index = np.argsort([
        sp[z != 0].max(1).mean() for sp in splits
    ])[::-1]
    splits_reordered = [splits[i] for i in index][:n_cols]
    X_reordered = np.hstack(splits_reordered)

    # Plot X
    vmin, vmax = X_vrange
    for i, l in enumerate([0, 1, 2]):
        axs[i, 0].pcolor(
            gaussian_filter(X_reordered[z == l, :], X_sigma),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

    # Plot y 
    cmap_fn = plt.get_cmap(cmap, 3)
    vmin, vmax = y_vrange
    for i, (c, l) in enumerate(zip([1, 2, 0], [0, 1, 2])):
        axs[i, 1].pcolor(
            gaussian_filter(y[z == l], y_sigma),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        axs[i, 2].add_patch(
            plt.Rectangle([0, 0], 1, 1, color=cmap_fn(c))
        )
        axs[i, 2].set_xlim([0, 1])
        axs[i, 2].set_ylim([0, 1])

    #
    for i in range(3):
        y1, y2 = axs[i, 0].get_ylim()
        for x0 in range(0, n_cols * n_bins, n_bins)[:-1]:
            x1 = x0 + n_bins
            axs[i, 0].vlines(x1, y1, y2, color='k', alpha=0.5, linestyle=':', lw=1)
        axs[i, 0].set_ylim([y1, y2])

    # Clean up
    for ax in axs.flatten():
        # for sp in ('top', 'right', 'bottom', 'left'):
        #     ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0, 0].set_ylabel('0', rotation=0, labelpad=15)
    axs[1, 0].set_ylabel('1', rotation=0, labelpad=15)
    axs[2, 0].set_ylabel('2', rotation=0, labelpad=15)
    fig.supylabel('Label', fontsize=10)
    axs[0, 0].set_title('X', fontsize=10)
    axs[0, 1].set_title('y', fontsize=10)
    axs[0, 2].set_title('z', fontsize=10)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05, wspace=0.025)

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

def visualize_mlp_regressor_performance(
    data,
    cmap='Blues',
    vrange=(-1500, 1500),
    subplot_height_ratio=80,
    figsize=(7.5, 4)
    ):
    """
    """

    #
    if type(data) == str:
        X, y, z = load_mlati(data)
    elif type(data) == tuple:
        X, y, z = data

    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
        X, y, z,
        test_size=0.2
    )

    #
    reg_pt = PyTorchMLPRegressor(max_epochs=100)
    reg_pt.fit(X_train, y_train)
    y_predicted_pt = reg_pt.predict(X_test)
    reg_sk = MLPRegressor(solver='adam', max_iter=100)
    reg_sk.fit(X_train, y_train)
    y_predicted_sk = reg_sk.predict(X_test)

    #
    height_ratios = [
        subplot_height_ratio,
        np.sum(z_test == 0),
        subplot_height_ratio,
        np.sum(z_test == 1),
        subplot_height_ratio,
        np.sum(z_test == 2)
    ]
    fig, axs = plt.subplots(
        nrows=6,
        ncols=5,
        sharex=True,
        gridspec_kw={'height_ratios': height_ratios}
    )
    cmap_fn = plt.get_cmap(cmap, 3)
    for i, l in zip([1, 3, 5], [0, 1, 2]):
        residuals_sk = y_test[z_test == l] - y_predicted_sk[z_test == l]
        residuals_pt = y_test[z_test == l] - y_predicted_pt[z_test == l]
        # index = np.argsort(np.abs(residuals_pt).sum(1))
        index = np.arange(residuals_pt.shape[0])
        axs[i, 0].pcolor(y_test[z_test == l][index], vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        axs[i, 1].pcolor(y_predicted_sk[z_test == l][index], vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        axs[i, 2].pcolor(y_predicted_pt[z_test == l][index], vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        axs[i, 3].pcolor(
            residuals_sk,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap
        )
        axs[i, 4].pcolor(
            residuals_pt,
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap
        )
        axs[i - 1, 0].plot(y_test[z_test == l].mean(0), color=cmap_fn(2))
        axs[i - 1, 1].plot(y_predicted_sk[z_test == l].mean(0), color=cmap_fn(2))
        axs[i - 1, 2].plot(y_predicted_pt[z_test == l].mean(0), color=cmap_fn(2))
        axs[i - 1, 3].plot(residuals_sk.mean(0), color=cmap_fn(2))
        axs[i - 1, 4].plot(residuals_pt.mean(0), color=cmap_fn(2))
    
    #
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[::2].flatten():
        ax.set_ylim([-1000, 2000])
        for sp in ('top', 'right', 'bottom', 'left'):
            ax.spines[sp].set_visible(False)
    axs[1, 0].set_ylabel('0', rotation=0, labelpad=15)
    axs[3, 0].set_ylabel('1', rotation=0, labelpad=15)
    axs[5, 0].set_ylabel('2', rotation=0, labelpad=15)
    fig.supylabel('Label', fontsize=10)
    titles = (
        r'$y_{\text{test}}$',
        r'$y_{pred}\text{ }\text{(scikit-learn)}$',
        r'$y_{pred}\text{ }\text{(PyTorch)}$',
        r'$\text{Res. (scikit-learn)}$',
        r'$\text{Res. (PyTorch)}$'
    )
    for j, t in enumerate(titles):
        axs[0, j].set_title(t, fontsize=10)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    return fig, axs