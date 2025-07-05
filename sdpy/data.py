import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal

def psth(
    reference_event,
    relative_event,
    window=(-1, 1),
    binsize=0.05,
    return_shape=False
    ):
    """
    """

    # Case of a single bin
    if binsize is None:
        nBins = 1
        binEdges = window
        t = window[0] + np.diff(window).item() / 2

    # Check that the time window is evenly divisible by the binsize
    else:
        start, stop = np.around(window, 3)
        residual = (Decimal(str(stop)) - Decimal(str(start))) % Decimal(str(binsize))
        if residual != 0:
            raise ValueError('Window must be evenly divisible by binsize')

        # Compute the bin edges
        window_length = float(Decimal(str(stop)) - Decimal(str(start)))
        nBins = int(round(window_length / binsize))
        bin_edges = np.linspace(start, stop, nBins + 1)
        t = bin_edges[:-1] + binsize / 2

    # Return early if all you need is the shape of the PSTH
    if return_shape:
        return t, reference_event.size, nBins

    # Compute relative timestamps and histograms
    M = np.full([reference_event.size, nBins], np.nan)
    relative_timestamps = list()
    for row_index, timestamp in enumerate(reference_event):
        relative_timestamps_ = relative_event - timestamp
        m = np.logical_and(
            relative_timestamps_ >= window[0],
            relative_timestamps_ <= window[1]
        )
        bin_counts, bin_edges_ = np.histogram(relative_timestamps_[m], bins=bin_edges)
        M[row_index, :] = bin_counts
        relative_timestamps.append(relative_timestamps_[m])

    #
    return t, M, relative_timestamps

def _estimate_recording_epoch(
    spike_timestamps,
    spike_clusters=None,
    target_cluster=None,
    pad=1,
    ):
    """
    Estimate the start and stop of global spiking given the spike timestamps for one or all units
    """

    if target_cluster is None:
        spike_indices = np.arange(spike_timestamps)
    else:
        spike_indices = np.where(spike_clusters == target_cluster)[0]
    t1 = np.floor(spike_clusters[spike_indices].min()) + pad
    t2 = np.ceil(spike_timestamps[spike_indices].max()) - pad
    epoch = (t1, t2)

    return epoch

def load_mlati(
    filename,
    window=(-0.1, 0.1),
    binsize_y=0.005,
    binsize_x=0.05,
    n_lagging_bins=(30, 0),
    p_max=0.05,
    fr_min=0.2,
    spike_timestamps=None,
    ):
    """
    """

    # Load all the required datasets from the h5 file
    with h5py.File(filename, 'r') as stream:
        eye_position = np.array(stream['pose/filtered'])[:, 0]
        n_frames_recorded = len(eye_position)
        frame_timestamps = np.array(stream['frames/left/timestamps'])[:n_frames_recorded]
        if spike_timestamps is None:
            spike_timestamps = np.array(stream[f'spikes/timestamps'])
        spike_clusters = np.array(stream[f'spikes/clusters'])
        p_values = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)
        saccade_timestamps = np.array(stream['saccades/predicted/left/timestamps'])
        saccade_labels = np.array(stream['saccades/predicted/left/labels'])

    # NOTE: Sometimes there are different numbers of frames and timestamps which will preclude this analysis
    if frame_timestamps.size != eye_position.size:
        raise Exception(f'Different number of frames ({eye_position.size}) and timestamps ({frame_timestamps.size})')

    # Create the eye position time series
    t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
    v_raw = np.diff(eye_position)
    v_raw[np.isnan(v_raw)] = np.interp(t_raw[np.isnan(v_raw)], t_raw, v_raw) # Impute with interpolation

    #
    event_timestamps = np.concatenate([
        saccade_timestamps[:, 0],
        saccade_timestamps[:-1, 0] + (np.diff(saccade_timestamps[:, 0]) / 2)
    ])
    z = np.concatenate([
        saccade_labels,
        np.zeros(saccade_timestamps.size - 1)
    ])

    # Collect the eye velocity waveforms for saccades 
    y = list()
    t_eval = np.arange(window[0], window[1], binsize_y) + (binsize_y / 2)
    for event_timestamp in event_timestamps:
        wf = np.interp(
            t_eval + event_timestamp,
            t_raw,
            v_raw
        )
        y.append(wf)
    y = np.array(y)

    # Exclude units without event-related activity
    unique_clusters = np.unique(spike_clusters)
    if p_max is None:
        target_clusters = unique_clusters
    else:
        cluster_indices = np.arange(len(unique_clusters))[p_values <= p_max]
        target_clusters = unique_clusters[cluster_indices] 

    # Exclude units with too low of a firing rate (probabily partial units)
    if fr_min is not None:
        unit_indices = list()
        for i_unit, target_cluster in enumerate(target_clusters):
            t1, t2 = _estimate_recording_epoch(
                spike_timestamps,
                spike_clusters,
                target_cluster,
                pad=5   
            )
            spike_indices = np.where(spike_clusters == target_cluster)[0]
            n_bins = int((t2 - t1) / binsize_x)
            n_spikes, bin_edges_ = np.histogram(
                spike_timestamps[spike_indices],
                range=(t1, t2),
                bins=n_bins
            )
            fr = n_spikes.mean() / binsize_x
            if fr < fr_min:
                unit_indices.append(i_unit)
        target_clusters = np.delete(target_clusters, unit_indices)

    # Compute the edges of the time bins centered on the saccade
    bin_offsets = np.arange(-1 * n_lagging_bins[0], n_lagging_bins[1] + 1, 1)
    bin_centers = bin_offsets * binsize_x
    left_edges = np.around(bin_centers - (binsize_x / 2), 5)
    right_edges = left_edges + binsize_x
    bin_edges = np.concatenate([left_edges, right_edges[-1:]])

    # Compute histograms and store in response matrix of shape N units x M saccades x P time bins
    n_units = len(target_clusters)
    R = list()
    for i_unit, target_cluster in enumerate(target_clusters):
        end = '\r' if i_unit + 1 != n_units else '\n'
        print(f'Computing histograms for unit {i_unit + 1} out of {n_units} ...', end=end)
        spike_indices = np.where(spike_clusters == target_cluster)[0]
        sample = list()
        for event_timestamp in event_timestamps:
            n_spikes, bin_edges_ = np.histogram(
                spike_timestamps[spike_indices],
                bins=np.around(bin_edges + event_timestamp, 3)
            )
            fr = n_spikes / binsize_x
            sample.append(fr)
        R.append(sample)
    R = np.array(R)

    # Populate response matrix
    X = list()
    n_events = R.shape[1]
    for i_event in range(n_events):
        sample = list()
        for bin_offset in bin_offsets:  # Outer loop over lags
            for i_unit in range(n_units):  # Inner loop over neurons
                bin_index = int((R.shape[2] - 1) / 2) + bin_offset
                fr = R[i_unit, i_event, bin_index]
                sample.append(fr)
        X.append(sample)
    X = np.array(X)

    # Sort by time
    index = np.argsort(event_timestamps)
    X = X[index, :]
    y = y[index]
    z = z[index]

    # Normalize firing rate
    X = MinMaxScaler().fit_transform(X)

    # Remove samples with NaN values
    mask = np.vstack([
        np.isnan(X).any(1),
        np.isnan(y).any(1),
        np.isnan(z)
    ]).any(0)
    X = np.delete(X, mask, axis=0)
    y = np.delete(y, mask, axis=0)
    z = np.delete(z, mask)

    return X, y, z

def jitter_spike_timestamps(
    filename,
    j_min=-3,
    j_max=3,
    ):
    """
    Apply temporal jitter to spike timestamps
    """

    with h5py.File(filename, 'r') as stream:
        spike_timestamps = np.array(stream[f'spikes/timestamps'])
    spike_timestamps_jittered = spike_timestamps + np.random.uniform(low=j_min, high=j_max, size=spike_timestamps.size)

    return spike_timestamps_jittered