from pathlib import Path
from sdpy.data import Mlati
from sdpy.utils import split_time_series
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor

def collect_h5_files(root):
    """
    """

    if type(root) != Path:
        root = Path(root)
    files = list()
    for folder in root.iterdir():
        if folder.is_dir() == False:
            continue
        for animal in folder.iterdir():
            if animal.is_dir() == False:
                continue
            metadata_file = animal.joinpath('metadata.txt')
            if metadata_file.exists() == False:
                continue
            with open(metadata_file, 'r') as stream:
                lines = stream.readlines()
            metadata = {}
            for ln in lines:
                key, value = ln.rstrip('\n').split(': ')
                metadata[key.lower()] = value
            experiment = metadata['experiment']
            if experiment.lower() == 'mlati':
                output_file = animal.joinpath('output.hdf')
                if output_file.exists() == False:
                    continue
                files.append(output_file)

    return files

class DecodingAnalysis():
    """
    """

    def __init__(self, files):
        self.files = files
        return
    
    def run(
        self,
        window_size=10,
        lags=[0,],
        xy_binsize=0.02,
        derivative=1,
        t_range=None,
        skip_to_index=None
        ):
        """
        """

        r2_scores = np.full([len(self.files), len(lags)], np.nan)
        reg = MLPRegressor(hidden_layer_sizes=[256, 128], early_stopping=True)
        n_files = len(self.files)
        n_lags = len(lags)
        for i_file, file in enumerate(self.files):
            if skip_to_index is not None and i_file < skip_to_index:
                continue
            try:
                for i_lag, lag in enumerate(lags):
                    end = '\r' if i_lag + 1 < n_lags else '\n'
                    print(f'Working on lag {i_lag + 1} out of {n_lags} lags for file {i_file + 1} out of {n_files} files', end=end)
                    ds = Mlati(file, form='W', window_size=window_size,
                        lag=lag, xy_binsize=xy_binsize, derivative=derivative,
                        t_range=t_range
                    )
                    X = MinMaxScaler().fit_transform(ds.X)
                    y = StandardScaler().fit_transform(ds.y).reshape(-1, 1)
                    X_train, X_test, y_train, y_test = split_time_series(X, y, test_fraction=0.2)
                    reg.fit(X_train, y_train.flatten())
                    r2_scores[i_file, i_lag] = reg.score(X_test, y_test.flatten())
            except:
                print(f'Exception for file: {file}')
                continue
            np.save(f'/home/josh/Desktop/decoding_curves/curve_{i_file}.npy', r2_scores[i_file, :])

        return r2_scores