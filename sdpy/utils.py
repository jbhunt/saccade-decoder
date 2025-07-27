import numpy as np

def estimate_recording_epoch(
    spike_timestamps,
    spike_clusters=None,
    target_cluster=None,
    pad=1,
    ):
    """
    Estimate the start and stop of global spiking given the spike timestamps for one or all units
    """

    if target_cluster is None:
        spike_indices = np.arange(spike_timestamps.size)
    else:
        spike_indices = np.where(spike_clusters == target_cluster)[0]
    t1 = np.floor(spike_timestamps[spike_indices].min()) + pad
    t2 = np.ceil(spike_timestamps[spike_indices].max()) - pad
    epoch = (t1, t2)

    return epoch

def split_time_series(X, y, test_fraction=0.2):
    """
    """

    i_split = int(round(X.shape[0] * (1 - test_fraction)))
    X_train, X_test = X[:i_split], X[i_split:]
    y_train, y_test = y[:i_split], y[i_split:]

    return X_train, X_test, y_train, y_test