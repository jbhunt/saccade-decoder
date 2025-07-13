import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import cross_validate, TimeSeriesSplit
import numpy as np
from .data import load_mlati_continuous

def split_time_series(X, y, test_fraction=0.2):
    """
    """

    i_split = int(round(X.shape[0] * test_fraction))
    X_train, X_test = X[:i_split], X[i_split:]
    y_train, y_test = y[:i_split], y[i_split:]

    return X_train, X_test, y_train, y_test

class SlidingWindowDataset(Dataset):
    """
    """

    def __init__(self, X, y=None, window_size=1, lag=0):
        """
        """

        if lag < 0:
            raise Exception('Lag must be positive')

        if type(X) != torch.Tensor:
            self.X = torch.tensor(X, dtype=torch.float)
        else:
            self.X = X
        if y is None:
            self.y = y
        else:
            if type(y) != torch.Tensor:
                self.y = torch.tensor(y, dtype=torch.float)
            else:
                self.y = y
        self.window_size = window_size
        self.lag = lag

        return
    
    def __len__(self):
        t_range = (self.window_size + self.lag) # Distance from start of sequence to prediction
        return self.X.shape[0] - t_range
    
    def __getitem__(self, i):
        """
        """

        if i >= len(self):
            raise IndexError('Dataset index out of range')
        
        # Handle negative indices
        n_steps = self.X.shape[0]
        if i < 0:
            i = n_steps + i

        # Slice out X data
        X_stop_index = i + self.window_size
        X_start_index = i
        X_slice = self.X[X_start_index: X_stop_index]

        # Index y value
        # y_index = X_stop_index + self.lag
        y_index = X_stop_index + self.lag
        if self.y is None:
            y_value = torch.tensor(np.nan, dtype=torch.float)
        else:
            if y_index >= n_steps:
                y_value = torch.tensor(np.nan, dtype=torch.float)
            else:
                y_value = self.y[y_index]
        y_index = torch.tensor(y_index, dtype=torch.long)

        return X_slice, y_value, y_index
    
    @property
    def offset(self):
        return self.window_size + self.lag

class RecurrentNeuralNetwork(nn.Module):
    """
    """

    def __init__(
        self,
        n_features=1,
        n_units=1, # Memory "width"
        n_layers=1, # Model "depth"
        dropout=0.0
        ):
        """
        """

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_units,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.fc = nn.Linear(n_units, 1)

        return

    def forward(self, X):
        """
        """

        _, (h_n, _) = self.lstm(X)
        last_hidden = h_n[-1]
        y_value = self.fc(last_hidden)

        return y_value

class PyTorchRNNRegressor(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(
        self,
        n_steps=10,
        n_units=16,
        n_layers=1,
        lag=0,
        batch_size=64,
        max_iter=30,
        dropout=0.0,
        lr=1e-3,
        tolerance=1e-4,
        patience=30,
        early_stopping=False,
        hold_out_fraction=0.1,
        f_report=100,
        device=None
        ):
        """
        """

        self.n_steps = n_steps
        self.n_units = n_units
        self.n_layers = n_layers
        self.lag = lag
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.model = None
        self.dropout = dropout
        self.lr = lr
        self.tolerance = tolerance
        self.patience = patience
        self.early_stopping = early_stopping
        self.hold_out_fraction = hold_out_fraction
        self.f_report = f_report
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_epoch = None

        return
    
    def fit(self, X, y):
        """
        """

        #
        if self.early_stopping == True:
            split_index = int(round(X.shape[0] * (1 - self.hold_out_fraction)))
            X_train, y_train = X[:split_index], y[:split_index]
            X_test, y_test = X[split_index:], y[split_index:]
            data_loader_test = DataLoader(
                SlidingWindowDataset(X_test, y_test, self.n_steps, self.lag),
                self.batch_size,
                shuffle=False
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
            data_loader_test = None
        data_loader_train = DataLoader(
            SlidingWindowDataset(X_train, y_train, self.n_steps, self.lag),
            self.batch_size,
            shuffle=True
        )

        #
        self.model = RecurrentNeuralNetwork(
            X.shape[1],
            self.n_units,
            self.n_layers,
            self.dropout
        ).to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.performance = {
            'train': np.full(self.max_iter, np.nan),
            'test': np.full(self.max_iter, np.nan)
        }
        loss_minimum = np.inf
        best_epoch = None
        n_epochs_without_improvement = 0
        for i_epoch in range(self.max_iter):

            # Training phase
            self.model.train()
            loss_current_epoch_train = 0.0
            n_samples = 0
            for X_batch, y_batch, y_indices in data_loader_train:
                optimizer.zero_grad()
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).view(-1, 1)
                y_predicted = self.model(X_batch)
                loss = loss_fn(y_predicted, y_batch)
                loss.backward()
                optimizer.step()
                loss_current_epoch_train += loss.item() * X_batch.shape[0]
                n_samples += X_batch.shape[0]
            loss_current_epoch_train /= n_samples

            # Validation phase
            loss_current_epoch_test = 0.0
            if self.early_stopping:
                self.model.eval()
                with torch.no_grad():
                    n_samples = 0
                    for X_batch, y_batch, y_indices in data_loader_test:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device).view(-1, 1)
                        y_predicted = self.model(X_batch)
                        loss = loss_fn(y_predicted, y_batch)
                        loss_current_epoch_test += loss.item() * X.shape[0]
                        n_samples += X_batch.shape[0]
                loss_current_epoch_test /= n_samples

            #
            if (i_epoch == 0) or ((i_epoch + 1) % self.f_report == 0):
                print(f'Epoch {i_epoch + 1}: training loss={loss_current_epoch_train:.5f}, test loss={loss_current_epoch_test:.5f}')
            self.performance['train'][i_epoch] = loss_current_epoch_train
            self.performance['test'][i_epoch] = loss_current_epoch_test

            # If measuring test performance for early stopping
            if self.early_stopping:
                loss_current_epoch = loss_current_epoch_test
            else:
                loss_current_epoch = loss_current_epoch_train

            # Save the best state
            if loss_current_epoch < loss_minimum:
                best_epoch = i_epoch
                state_dict = self.model.state_dict()
                loss_minimum = loss_current_epoch

            # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
            if loss_current_epoch - loss_minimum < self.tolerance:
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1
            if self.early_stopping and (n_epochs_without_improvement > self.patience):
                break

        #
        self.model.load_state_dict(state_dict)
        self.best_epoch = best_epoch

        return
    
    def predict(self, X, return_indices=False):
        """
        """

        self.model.eval()
        data_loader = DataLoader(
            SlidingWindowDataset(X, y=None, window_size=self.n_steps, lag=self.lag),
            batch_size=self.batch_size,
            shuffle=False
        )
        predictions = list()
        target_indices = list()
        with torch.no_grad():
            for X_batch, y_batch, i_batch in data_loader:
                X_batch = X_batch.to(self.device)
                predictions.append(self.model(X_batch).cpu())
                target_indices.append(i_batch)
        if return_indices:
            return torch.cat(predictions).squeeze(1).numpy(), torch.cat(target_indices).cpu().numpy().astype(int)
        else:
            return torch.cat(predictions).squeeze(1).numpy()
    
    def score(self, X, y):
        """
        Compute the root mean squared error
        """

        offset = self.n_steps + self.lag
        y_pred = self.predict(X)
        y_shortened = y[offset:]
        rmse = np.sqrt(np.mean(np.power(y_shortened - y_pred, 2)))

        return rmse
    
    def shorten_target_series(self, y):
        """
        """

        return y[self.n_steps + self.lag:]
    
def measure_rnn_regressor_performance(
    filename,
    n_splits=5,
    Xy=None,
    **kwargs   
    ):
    """
    """

    if Xy is None:
        X, y = load_mlati_continuous(filename)
    else:
        X, y = Xy
    offset = 10 + 1
    reg = PyTorchRNNRegressor(**kwargs)
    cv = TimeSeriesSplit(n_splits)
    scores = {
        'r2_test': np.full(n_splits, np.nan),
        'rmse_test': np.full(n_splits, np.nan)
    }
    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_test_shortened = y_test[offset:]
        r2 = 1 - (np.sum(np.power(y_test_shortened - y_pred, 2)) / np.sum(np.power(y_test_shortened - y_test_shortened.mean(), 2)))
        rmse = np.sqrt(np.mean(np.power(y_test_shortened - y_pred, 2)))
        scores['r2_test'][k] = r2
        scores['rmse_test'][k] = rmse

    return scores
