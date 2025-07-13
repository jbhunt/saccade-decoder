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
    X_binsize=0.01,
    X_bincounts=(20, 20),
    y_binsize=0.002,
    y_bincounts=(25, 45),
    p_max=1e-3,
    fr_min=1,
    standardize_firing_rate=True,
    add_null_events=True,
    resample_spike_timestamps=False,
    shuffle_labels=False,
    random_seed=42,
    ):
    """
    """

    #
    np.random.seed(random_seed)

    # Load all the required datasets from the h5 file
    with h5py.File(filename, 'r') as stream:
        eye_position = np.array(stream['pose/filtered'])[:, 0]
        n_frames_recorded = len(eye_position)
        frame_timestamps = np.array(stream['frames/left/timestamps'])[:n_frames_recorded]
        spike_timestamps = np.array(stream[f'spikes/timestamps'])
        spike_clusters = np.array(stream[f'spikes/clusters'])
        p_values = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)
        saccade_timestamps = np.array(stream['saccades/predicted/left/timestamps'])
        saccade_labels = np.array(stream['saccades/predicted/left/labels'])

    # For some experiments there are different numbers of frames and timestamps which will preclude further processing
    if frame_timestamps.size != eye_position.size:
        raise Exception(f'Different number of frames ({eye_position.size}) and frame timestamps ({frame_timestamps.size})')
    
    # Drop saccades without timestamps
    invalid_saccades = np.isnan(saccade_timestamps[:, 0])
    saccade_labels = np.delete(saccade_labels, invalid_saccades)
    saccade_timestamps = np.delete(saccade_timestamps[:, 0], invalid_saccades)

    # Resample spike timestamps using a uniform distribution (optional)
    if resample_spike_timestamps == True:
        spike_timestamps = np.around(np.random.uniform(
            low=spike_timestamps.min(),
            high=spike_timestamps.max(),
            size=spike_timestamps.size
        ), 3).astype(spike_timestamps.dtype)

    # Mix in "null" events (optional)
    if add_null_events:
        n_null_events = round(np.mean([np.sum(saccade_labels == -1), np.sum(saccade_labels ==1)]))
        null_event_timestamps = np.random.uniform(
            low=saccade_timestamps.min(),
            high=saccade_timestamps.max(),
            size=n_null_events
        )
        event_timestamps = np.concatenate([
            saccade_timestamps,
            null_event_timestamps
        ])
    else:
        event_timestamps = saccade_timestamps
    
    # Re-code temporal saccades as 2 (instead of -1)
    z = np.concatenate([
        saccade_labels,
        np.zeros(n_null_events)
    ])
    z[z == -1] = 2

    # Sort by time
    index = np.argsort(event_timestamps)
    event_timestamps = event_timestamps[index]
    z = z[index]

    # Shuffle saccade labels (optional)
    if shuffle_labels:
        np.random.shuffle(z)

    # Create the eye position time series
    t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
    v_raw = np.diff(eye_position)
    v_raw[np.isnan(v_raw)] = np.interp(t_raw[np.isnan(v_raw)], t_raw, v_raw) # Impute with interpolation

    # Collect the eye velocity waveforms for saccades 
    y = list()
    t_eval = y_binsize * (np.arange(-1 * y_bincounts[0], y_bincounts[1], 1) + 0.5)
    for event_timestamp in event_timestamps:
        wf = np.interp(
            t_eval + event_timestamp,
            t_raw,
            v_raw
        )
        wf_scaled =  wf / y_binsize
        y.append(wf_scaled)
    y = np.array(y)

    # Exclude units without event-related activity
    unique_clusters = np.unique(spike_clusters)
    if p_max is None:
        target_clusters = unique_clusters
    else:
        cluster_indices = np.arange(len(unique_clusters))[p_values <= p_max]
        target_clusters = unique_clusters[cluster_indices] 

    # Exclude units with too low of a firing rate
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
            n_bins = int((t2 - t1) / X_binsize)
            n_spikes, bin_edges_ = np.histogram(
                spike_timestamps[spike_indices],
                range=(t1, t2),
                bins=n_bins
            )
            fr = n_spikes / X_binsize
            
            #
            if fr.mean() < fr_min:
                unit_indices.append(i_unit)
        target_clusters = np.delete(target_clusters, unit_indices)

    # Compute the edges of the time bins centered on the saccade
    left_edges = np.arange(-1 * X_bincounts[0], X_bincounts[1], 1)
    right_edges = left_edges + 1
    all_edges = np.concatenate([left_edges, [right_edges[-1],]]) * X_binsize

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
                bins=np.around(all_edges + event_timestamp, 3)
            )
            fr = n_spikes / X_binsize
            sample.append(fr)
        R.append(sample)
    R = np.array(R)

    # Populate X (unit major)
    X = list()
    n_units, n_events, n_bins = R.shape
    for i_event in range(n_events):
        sample = list()
        for i_unit in range(n_units):
            for i_bin in range(n_bins):
                sample.append(R[i_unit, i_event, i_bin])
        X.append(sample)
    X = np.array(X)

    # Standardize firing rate
    if standardize_firing_rate:
        n_samples = X.shape[0]
        splits = np.split(X, n_units, axis=1)
        splits_standardized = list()
        for i_unit, split in enumerate(splits):
            fr_mean = split[z == 0, :].mean()
            fr_std = split[z == 0, :].std()
            split_standardized = (split - fr_mean) / fr_std
            split_standardized = split_standardized.reshape(n_samples, -1)
            splits_standardized.append(split_standardized)
        X = np.hstack(splits_standardized)

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

def load_mlati_continuous(
    filename,
    trange=(0, 120),
    Xy_binsize=0.01,
    p_max=1e-3,
    fr_min=1,
    standardize_firing_rate=True,
    resample_spike_timestamps=False,
    random_seed=42
    ):
    """
    """

    #
    np.random.seed(random_seed)

    #
    trange_rounded = (
        np.floor(trange[0]),
        np.ceil(trange[1])
    )

    # Load all the required datasets from the h5 file
    with h5py.File(filename, 'r') as stream:
        eye_position = np.array(stream['pose/filtered'])[:, 0]
        n_frames_recorded = len(eye_position)
        frame_timestamps = np.array(stream['frames/left/timestamps'])[:n_frames_recorded]
        spike_timestamps = np.array(stream[f'spikes/timestamps'])
        spike_clusters = np.array(stream[f'spikes/clusters'])
        p_values = np.vstack([
            np.array(stream['zeta/saccade/nasal/p']),
            np.array(stream['zeta/saccade/temporal/p'])
        ]).min(0)

    # For some experiments there are different numbers of frames and timestamps which will preclude further processing
    if frame_timestamps.size != eye_position.size:
        raise Exception(f'Different number of frames ({eye_position.size}) and frame timestamps ({frame_timestamps.size})')
    
    # Resample spike timestamps using a uniform distribution (optional)
    if resample_spike_timestamps == True:
        spike_timestamps = np.around(np.random.uniform(
            low=spike_timestamps.min(),
            high=spike_timestamps.max(),
            size=spike_timestamps.size
        ), 3).astype(spike_timestamps.dtype)

    # Create the eye position time series
    t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
    v_raw = np.diff(eye_position)
    v_raw[np.isnan(v_raw)] = np.interp(t_raw[np.isnan(v_raw)], t_raw, v_raw) # Impute with interpolation
    y = np.interp(
        np.arange(*trange_rounded, Xy_binsize) + (Xy_binsize / 2),
        t_raw,
        v_raw
    )
    y[np.isnan(y)] = np.nanmedian(y)
    y = y / Xy_binsize

    # Exclude units without event-related activity
    unique_clusters = np.unique(spike_clusters)
    if p_max is None:
        target_clusters = unique_clusters
    else:
        cluster_indices = np.arange(len(unique_clusters))[p_values <= p_max]
        target_clusters = unique_clusters[cluster_indices] 

    # Exclude units with too low of a firing rate
    X = list()
    for i_unit, target_cluster in enumerate(target_clusters):

        #
        spike_indices = np.where(spike_clusters == target_cluster)[0]

        # Compute whole-recording parameters
        t1, t2 = _estimate_recording_epoch(
            spike_timestamps,
            spike_clusters,
            target_cluster,
            pad=5   
        )
        n_bins = int((t2 - t1) / Xy_binsize)
        n_spikes, bin_edges_ = np.histogram(
            spike_timestamps[spike_indices],
            range=(t1, t2),
            bins=n_bins
        )
        fr = n_spikes / Xy_binsize
        fr_mean = fr.mean()
        if fr_mean < fr_min:
            continue
        fr_std = fr.std()

        # Compute firing rate within target window
        n_bins = int((trange_rounded[1] - trange_rounded[0]) / Xy_binsize)
        n_spikes, bin_edges_ = np.histogram(
            spike_timestamps[spike_indices],
            range=trange_rounded,
            bins=n_bins
        )
        fr = n_spikes / Xy_binsize
        fr_scaled = (fr - fr_mean) / fr_std
        if standardize_firing_rate:
            X.append(fr_scaled)
        else:
            X.append(fr)
    X = np.array(X).T

    return X, y