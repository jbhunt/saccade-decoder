import torch
import numpy as np
from torch import nn
from torch import optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    
class _MultiLayerPerceptron(nn.Module):
    """
    Implmentation of an articial neural network (Multi-layer perceptron)
    """

    def __init__(self, input_layer_size=1, output_layer_size=1, hidden_layer_sizes=[1,]):
        """
        """

        super().__init__()
        layers = list()
        layers.append(nn.Linear(input_layer_size, hidden_layer_sizes[0]))
        layers.append(nn.ReLU())
        n_layers = len(hidden_layer_sizes)
        for i_layer in range(n_layers):
            s1 = hidden_layer_sizes[i_layer]
            if i_layer + 1 < n_layers:
                s2 = hidden_layer_sizes[n_layers + 1]
                layers.append(nn.Linear(s1, s2))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(s1, output_layer_size))
        self.seq = nn.Sequential(*layers)
            
        return
    
    def forward(self, x):
        """
        """

        return self.seq(x)
    
class PyTorchMLPRegressor(BaseEstimator, RegressorMixin):
    """
    """

    def __init__(
        self,
        hidden_layer_sizes=[10,],
        lr=0.0001,
        max_epochs=3000,
        tolerance=0,
        patience=50,
        early_stopping=True,
        hold_out_fraction=0.1,
        device=None
        ):
        """
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.lr = lr
        self.tolerance = tolerance
        self.patience = patience
        self.ann = None
        self.performance = None
        self.early_stopping = early_stopping
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.hold_out_fraction = hold_out_fraction

        return
    
    def _fit_with_early_stopping(
        self,
        X,
        y,
        ):
        """
        """

        #
        if len(y.shape) == 1 or y.shape[0] != X.shape[0]:
            raise Exception('y must have the same size as X along the first dimension')

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.ann = _MultiLayerPerceptron(
            Xt.shape[1],
            yt.shape[1],
            self.hidden_layer_sizes
        ).to(self.device)

        # Declare training and test indices
        n_samples = X.shape[0]
        training_index = np.random.choice(
            np.arange(n_samples),
            size=int(round(n_samples * (1 - self.hold_out_fraction)))
        )
        test_index = np.array([i for i in np.arange(n_samples) if i not in training_index])

        #
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.ann.parameters(), lr=self.lr)
        n_epochs_without_improvement = 0
        self.performance = np.full(self.max_epochs, np.nan)
        loss_minimum = np.inf
        state_dict = None
        best_epoch = None

        # Main training loop
        for i_epoch in range(self.max_epochs):

            # Training step
            self.ann.train()
            predictions = self.ann(Xt[training_index])
            loss_obj = loss_function(predictions, yt[training_index])
            optimizer.zero_grad()
            loss_obj.backward()
            optimizer.step()

            # Validation step
            self.ann.eval()
            with torch.no_grad():
                predictions = self.ann(Xt[test_index])
                loss_obj = loss_function(predictions, yt[test_index])
                loss_current_epoch = loss_obj.item()
            self.performance[i_epoch] = loss_current_epoch

            # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
            if loss_current_epoch - loss_minimum < self.tolerance:
                best_epoch = i_epoch
                state_dict = self.ann.state_dict()
                loss_minimum = loss_current_epoch
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            #
            if n_epochs_without_improvement > self.patience:
                break

        # Load the best cross-validated model
        self.ann.load_state_dict(state_dict)
        self.best_epoch = best_epoch

        return self
    

    def _fit_without_early_stopping(
        self,
        X,
        y
        ):
        """
        """

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.device)
        self.ann = _MultiLayerPerceptron(
            Xt.shape[1],
            yt.shape[1],
            self.hidden_layer_sizes
        ).to(self.device)
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.ann.parameters(), lr=self.lr)
        self.performance = np.full(self.max_epochs, np.nan)
        loss_minimum = np.inf
        loss_current_epoch = None
        best_epoch = None

        # Main training loop
        for i_epoch in range(self.max_epochs):

            # Training step
            self.ann.train()
            predictions = self.ann(Xt)
            loss_obj = loss_function(predictions, yt)
            optimizer.zero_grad()
            loss_obj.backward()
            optimizer.step()

            # Evaluation
            self.ann.eval()
            with torch.no_grad():
                predictions = self.ann(Xt)
                loss_obj = loss_function(predictions, yt)
                loss_current_epoch = loss_obj.item()
            self.performance[i_epoch] = loss_current_epoch

            #
            if loss_current_epoch < loss_minimum:
                loss_minimum = loss_current_epoch
                best_epoch = i_epoch

        self.best_epoch = best_epoch

        return self
    
    def fit(self, X, y):
        """
        """

        if self.early_stopping:
            self._fit_with_early_stopping(X, y)
        else:
            self._fit_without_early_stopping(X, y)

        return self
    
    def predict(self, X):
        """
        """

        tensor = self.ann(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
        return np.array(tensor.detach())
    
def train(X, y, n_splits=5, **kwargs):
    """
    """

    regressor = PyTorchMLPRegressor(**kwargs)
    cv = TimeSeriesSplit(n_splits=n_splits)
    params = {
        'hidden_layer_sizes': [[10,],
                               [100,],
                               [1000,],
                               [10000,],
                               [100000],
                              ],
    }
    gs = GridSearchCV(
        regressor,
        params,
        cv=cv,
        verbose=True,
        scoring='neg_mean_squared_error'
    )
    gs.fit(X, y)
    regressor = gs.best_estimator_
    y_predicted = regressor.predict(X)
    mse = np.mean(np.power(y_predicted - y, 2))
    print(f'Best estimator identified with MSE of {mse:.3f}')

    return gs.best_estimator_