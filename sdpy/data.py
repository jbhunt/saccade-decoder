import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from decimal import Decimal
from sdpy.utils import estimate_recording_epoch

def load_mlati_discrete(
    filename,
    derivative=0,
    X_binsize=0.01,
    X_bincounts=(20, 20),
    y_binsize=0.002,
    y_bincounts=(25, 45),
    p_max=1e-3,
    fr_min=1,
    standardize_firing_rate=False,
    add_null_events=True,
    resample_spike_timestamps=False,
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

    # Create the eye position time series
    t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
    if derivative == 0:
        t_raw = frame_timestamps
        y_raw = eye_position
    elif derivative == 1:
        t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
        y_raw = np.diff(eye_position) / y_binsize
    else:
        raise Exception(f'Derivatives > 1 not supported')
    y_raw[np.isnan(y_raw)] = np.interp(t_raw[np.isnan(y_raw)], t_raw, y_raw) # Impute with interpolation

    # Collect the eye velocity waveforms for saccades 
    y = list()
    t_eval = y_binsize * (np.arange(-1 * y_bincounts[0], y_bincounts[1], 1) + 0.5)
    for event_timestamp in event_timestamps:
        wf = np.interp(
            t_eval + event_timestamp,
            t_raw,
            y_raw
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
            t1, t2 = estimate_recording_epoch(
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
    derivative=0,
    t_range=None,
    xy_binsize=0.01,
    p_max=1e-3,
    fr_min=1,
    standardize_firing_rate=False,
    resample_spike_timestamps=False,
    random_seed=42
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

    #
    if t_range is None:
        t_range = estimate_recording_epoch(spike_timestamps)
    else:
        t_range = (
            np.floor(t_range[0]),
            np.ceil(t_range[1])
        )

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
    if derivative == 0:
        t_raw = frame_timestamps
        y_raw = eye_position
    elif derivative == 1:
        t_raw = frame_timestamps[:-1] + (np.diff(frame_timestamps) / 2)
        y_raw = np.diff(eye_position) / xy_binsize
    else:
        raise Exception(f'Derivatives > 1 not supported')
    y_raw[np.isnan(y_raw)] = np.interp(t_raw[np.isnan(y_raw)], t_raw, y_raw) # Impute with interpolation
    y = np.interp(
        np.arange(*t_range, xy_binsize) + (xy_binsize / 2),
        t_raw,
        y_raw
    )
    y[np.isnan(y)] = np.nanmedian(y)

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
        t1, t2 = estimate_recording_epoch(
            spike_timestamps,
            spike_clusters,
            target_cluster,
            pad=5   
        )
        n_bins = int((t2 - t1) / xy_binsize)
        n_spikes, bin_edges_ = np.histogram(
            spike_timestamps[spike_indices],
            range=(t1, t2),
            bins=n_bins
        )
        fr = n_spikes / xy_binsize
        fr_mean = fr.mean()
        if fr_mean < fr_min:
            continue
        fr_std = fr.std()

        # Compute firing rate within target window
        n_bins = int((t_range[1] - t_range[0]) / xy_binsize)
        n_spikes, bin_edges_ = np.histogram(
            spike_timestamps[spike_indices],
            range=t_range,
            bins=n_bins
        )
        fr = n_spikes / xy_binsize
        fr_scaled = (fr - fr_mean) / fr_std
        if standardize_firing_rate:
            X.append(fr_scaled)
        else:
            X.append(fr)
    X = np.array(X).T

    return X, y.reshape(-1, 1)

class SlidingWindowDataset(Dataset):
    """
    Dataset that returns (X_window, y_value, y_index) tuples for
    time-series learning, supporting both positive and negative lags.
    """

    def __init__(self, X, y=None, window_size=1, lag=0):
        self.X = torch.as_tensor(X, dtype=torch.float) if not isinstance(X, torch.Tensor) else X
        self.y = torch.as_tensor(y, dtype=torch.float) if y is not None and not isinstance(y, torch.Tensor) else y
        self.window_size = window_size
        self.lag = lag
        if self.lag < 0:
            if abs(self.lag) > self.window_size:
                self.element_length = abs(self.lag)
            else:
                self.element_length = self.window_size
        else:
            self.element_length = self.window_size + self.lag + 1
        if self.element_length > self.X.shape[0]:
            raise Exception('Not enough samples to create window')
        return

    def __len__(self):
        return self.X.shape[0] - self.element_length + 1

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError("Dataset index out of range")
        if i < 0:
            raise Exception('Negative indexing not supported')
        X_start = i
        X_end = i + self.window_size
        X_slice = self.X[X_start:X_end]
        y_index = X_end + self.lag
        y_value = self.y[y_index]
        return X_slice, y_value, torch.tensor(y_index, dtype=torch.long)

    @property
    def offset(self):
        return self.window_size + self.lag
    
def load_mlati_windowed(
    filename,
    **kwargs_,
    ):
    """
    """

    kwargs = {
        'xy_binsize': 0.01,
        't_range': None,
        'derivative': 0,
        'window_size': 1,
        'lag': 0
    }
    kwargs.update(kwargs_)

    X_series, y_series = load_mlati_continuous(
        filename,
        xy_binsize=kwargs['xy_binsize'],
        t_range=kwargs['t_range'],
        derivative=kwargs['derivative']
    )
    ds = SlidingWindowDataset(X_series, y_series,
        window_size=kwargs['window_size'],
        lag=kwargs['lag']
    )
    X, y = list(), list()
    for X_seq, y_seq, y_index in ds:
        X.append(X_seq.flatten())
        y.append(y_seq)
    X, y = map(np.array, [X, y])

    return X, y.reshape(-1, 1)

class Mlati():
    """
    """

    def __init__(
        self,
        filename,
        form='D',
        **kwargs,
        ):
        """
        """

        #
        if form in ['discrete', 'd', 'D', 1]:
            self._X, self._y, self._z = load_mlati_discrete(filename, **kwargs)
        elif form in ['continuous', 'c', 'C', 2]:
            self._X, self._y = load_mlati_continuous(filename, **kwargs)
            self._z = None
        elif form in ['windowed', 'w', 'W', 3]:
            self._X, self._y = load_mlati_windowed(filename, **kwargs)
            self._z = None
        else:
            raise Exception(f'{format} is not a valid form')

        return

    @property
    def X(self):
        return self._X
    
    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z