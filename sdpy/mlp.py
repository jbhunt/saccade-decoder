import torch
import numpy as np
from torch import nn
from torch import optim
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor
    
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
                s2 = hidden_layer_sizes[i_layer + 1]
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
    
class _BasePyTorchModel():
    """
    """

    def __init__(
        self,
        hidden_layer_sizes=[100,],
        lr=1e-3,
        alpha=1e-4,
        max_epochs=1000,
        tolerance=1e-4,
        patience=10,
        early_stopping=False,
        hold_out_fraction=0.1,
        device=None,
        ):
        """
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_epochs = max_epochs
        self.lr = lr
        self.alpha = alpha
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
    
    def fit(self, X, y):
        """
        """

        return self._fit(X, y)
    
class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin, _BasePyTorchModel):
    """
    """

    def _fit(self, X, y):
        """
        """

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y.ravel(), dtype=torch.long).to(self.device)

        #
        output_layer_size = np.unique(yt).size
        self.ann = _MultiLayerPerceptron(
            Xt.shape[1],
            output_layer_size,
            self.hidden_layer_sizes
        ).to(self.device)

        # Declare train and test indices
        n_samples = X.shape[0]
        if self.early_stopping:
            train_index = np.random.choice(
                np.arange(n_samples),
                size=int(round(n_samples * (1 - self.hold_out_fraction))),
                replace=False
            )
            test_index = np.array([i for i in np.arange(n_samples) if i not in train_index])
        else:
            train_index = np.arange(n_samples)
            test_index = None

        #
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.ann.parameters(),
            lr=self.lr,
            weight_decay=self.alpha
        )
        self.performance = {
            'train': np.full(self.max_epochs, np.nan),
            'test': np.full(self.max_epochs, np.nan),
        }
        loss_minimum = np.inf
        loss_current_epoch_train = None
        loss_current_epoch_test = None
        best_epoch = None

        # Main training loop
        for i_epoch in range(self.max_epochs):

            # Training step
            self.ann.train()
            predictions = self.ann(Xt[train_index])
            loss_obj = loss_function(predictions, yt[train_index])
            optimizer.zero_grad()
            loss_obj.backward()
            optimizer.step()

            # Monitor performance
            self.ann.eval()
            with torch.no_grad():

                # Train dataset
                predictions = self.ann(Xt[train_index])
                loss_obj = loss_function(predictions, yt[train_index])
                loss_current_epoch_train = loss_obj.item()
                self.performance['train'][i_epoch] = loss_current_epoch_train

                # Test dataset
                if self.early_stopping:
                    predictions = self.ann(Xt[test_index])
                    loss_obj = loss_function(predictions, yt[test_index])
                    loss_current_epoch_test = loss_obj.item()
                    self.performance['test'][i_epoch] = loss_current_epoch_test

            # If measuring test performance for early stopping
            if self.early_stopping:
                loss_current_epoch = loss_current_epoch_test
            else:
                loss_current_epoch = loss_current_epoch_train

            # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
            if loss_current_epoch - loss_minimum < self.tolerance:
                best_epoch = i_epoch
                state_dict = self.ann.state_dict()
                loss_minimum = loss_current_epoch
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            #
            if self.early_stopping and (n_epochs_without_improvement > self.patience):
                break

        # Load the best model
        self.ann.load_state_dict(state_dict)
        self.best_epoch = best_epoch

        return self
    
    def predict(self, X):
        """
        """

        logits = self.ann(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
        probability = logits.softmax(dim=1)
        most_likely_class = np.array(probability.argmax(1).detach())

        return most_likely_class
    
    def predict_proba(self, X):
        """
        """

        logits = self.ann(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu()
        probability = np.array(logits.softmax(dim=1).detach())

        return probability
    
class PyTorchMLPRegressor(BaseEstimator, RegressorMixin, _BasePyTorchModel):
    """
    """
    
    def _fit(self, X, y):
        """
        """

        Xt = torch.tensor(X, dtype=torch.float32).to(self.device)
        yt = torch.tensor(y, dtype=torch.float32).to(self.device)

        #
        output_layer_size = yt.shape[1]
        self.ann = _MultiLayerPerceptron(
            Xt.shape[1],
            output_layer_size,
            self.hidden_layer_sizes
        ).to(self.device)

        # Declare train and test indices
        n_samples = X.shape[0]
        if self.early_stopping:
            train_index = np.random.choice(
                np.arange(n_samples),
                size=int(round(n_samples * (1 - self.hold_out_fraction))),
                replace=False
            )
            test_index = np.array([i for i in np.arange(n_samples) if i not in train_index])
        else:
            train_index = np.arange(n_samples)
            test_index = None

        #
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(
            self.ann.parameters(),
            lr=self.lr,
            weight_decay=self.alpha
        )
        self.performance = {
            'train': np.full(self.max_epochs, np.nan),
            'test': np.full(self.max_epochs, np.nan),
        }
        loss_minimum = np.inf
        loss_current_epoch_train = None
        loss_current_epoch_test = None
        best_epoch = None

        # Main training loop
        for i_epoch in range(self.max_epochs):

            # Training step
            self.ann.train()
            predictions = self.ann(Xt[train_index])
            loss_obj = loss_function(predictions, yt[train_index])
            optimizer.zero_grad()
            loss_obj.backward()
            optimizer.step()

            # Monitor performance
            self.ann.eval()
            with torch.no_grad():

                # Train dataset
                predictions = self.ann(Xt[train_index])
                loss_obj = loss_function(predictions, yt[train_index])
                loss_current_epoch_train = loss_obj.item()
                self.performance['train'][i_epoch] = loss_current_epoch_train

                # Test dataset
                if self.early_stopping:
                    predictions = self.ann(Xt[test_index])
                    loss_obj = loss_function(predictions, yt[test_index])
                    loss_current_epoch_test = loss_obj.item()
                    self.performance['test'][i_epoch] = loss_current_epoch_test

            # If measuring test performance for early stopping
            if self.early_stopping:
                loss_current_epoch = loss_current_epoch_test
            else:
                loss_current_epoch = loss_current_epoch_train

            # Stop early if change in loss is slowing down or worsening (to prevent overfitting)
            if loss_current_epoch - loss_minimum < self.tolerance:
                best_epoch = i_epoch
                state_dict = self.ann.state_dict()
                loss_minimum = loss_current_epoch
                n_epochs_without_improvement = 0
            else:
                n_epochs_without_improvement += 1

            #
            if self.early_stopping and (n_epochs_without_improvement > self.patience):
                break

        # Load the best model
        self.ann.load_state_dict(state_dict)
        self.best_epoch = best_epoch

        return self
    
    def predict(self, X):
        """
        """

        y_predicted = np.array(self.ann(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().detach())

        return y_predicted
    
def measure_regressor_performance(X, y):
    """
    Benchmark the PyTorch MLP regressor with R2 and RMSE using scikit-learn's MLPRegressor class as a comparison
    """

    reg_pt = PyTorchMLPRegressor()
    reg_sk = MLPRegressor(solver='adam', max_iter=1000)
    r2_scorer = make_scorer(
        lambda y_t, y_p: 1 - (np.sum(np.power(y_t.ravel() - y_p.ravel(), 2)) / np.sum(np.power(y_t.ravel() - y_t.ravel().mean(), 2)))
    )
    rmse_scorer = make_scorer(
        lambda y_t, y_p: np.sqrt(np.mean(np.power(y_t.flatten() - y_p.ravel(), 2))) 
    )
    scoring = {
        'r2': r2_scorer,
        'rmse': rmse_scorer
    }
    scores = {
        'sk': None,
        'pt': None
    }
    for k, reg in zip(scores.keys(), [reg_sk, reg_pt]):
        scores[k] = cross_validate(
            reg,
            X,
            y,
            scoring=scoring,
            cv=5
        )

    return scores