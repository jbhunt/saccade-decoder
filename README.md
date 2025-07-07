# Saccade decoder
This is a repository that showcases a set of neural decoders implemented with PyTorch. The models are being used to decode eye movement kinematics (velocity) using neural data (single-unit spiking).

# Data
For these analyses I'm using a dataset I collected during my PhD which can be downloaded [here](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqcfn). I call this dataset the "Mlati" dataset, which stands for medial-lateral axis tangential insertion. This refers to the geometry of how a high density linear electrode array (Neuropixels 1.0) was inserted into the superior colliculus of mice to collect extracellular recordings of neural activity. Briefly, the data for each session in this dataset is stored in an h5 file with the filenameing convention "\<date>_\<animal>_store.hdf." I created the `data` module to interface with the Mlati dataset using these h5 files.

### Loading the training data
Here is an example that demonstrates how to load the neural activity and eye velocity signals for a single experiment.
```Python
from sdpy import data
X, y, z = data.load_mlati(<path to h5 file>)
```
The `load_mlati` function extracts spiking activity and eye velocity signals from the h5 files and organizes them into these ML-ready variables:
- `X` (N saccades x M features) - NumPy array where N is the number of saccades in the recording and M is the firing rate for each neuron in a window of time around the onset of saccades
- `y` (N saccades x P features) - NumPy array where N is the number of saccades and P is the eye velocity for each saccade in a window of time around the saccade
- `z` (N saccades x 1) - NumPy array where N is the number of saccades and each entry indicates the direction of the saccade (0, 1, or 2): 0 is not a saccade (i.e., null event), 1 is a nasal saccade, and 2 is a temporal saccade

### Predictor variables
The `X` variable returned by the `load_mlati` function contains neural activity in a window of time around the onset of saccades. The matrix is organized neuron-major/bin-minor such that each row is formed by the concatenation of multiple single-trial peri-stimulus time histograms (PETHs); therefore, the size of the second dimension of `X` is the number of time bins in the PETHs x the number of neurons in the recording. Here is a plot that visualizes the first 1200 columns of `X` (i.e., the first 30 neurons) grouped by unit along the x-axis and the type of saccade along the y-axis. 

<p align="center">
  <img src="docs/imgs/X.png" width="700" alt="Animated demo">
</p>

It seems like there are many units with a response to nasal and temporal saccades and some units exhibit a preference for one type of saccade. There is no response to the null events.

### Target variables
The `y` variable returned by the `load_mlati` function contains the velocity waveforms for each saccade in the training dataset, and the `z` variable indicates the type of saccade. These data will be used as the target variables for regression and classification problems, respectively. In the figure below, I'm showing 30 samples (and the mean) of each "type" of saccade: "Not a saccade," "Temporal" saccades, and "Nasal" saccades.

<p align="center">
  <img src="docs/imgs/saccade_waveforms.png" width="700" alt="Animated demo">
</p>

# Modeling
Below are some examples showcases various implementations of machine learning models and techniques.

### Multi-layer perceptron (for regression)
I implemented a simple Multi-layer perceptron regressor and classifier using PyTorch. The class definition for each of these models follows the design and interfacing conventions used by Scikit-learn. For example, each model has a `fit` method which accepts arguments `X` (predictive factors) and `y` (target variable) and a `predict` method. Just a reminder, in my case I'm using neural activity (`X`) to predict eye movement kinematics (`y`).
```Python
from sdpy import mlp, data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X, y, z = data.load_mlati(<path to h5 file>)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
reg = mlp.PyTorchMLPRegressor()
reg.fit(X_train, y_train)
y_predicted = reg.predict(X_test)
```

As a sanity check, I compared the performance of my implementation against Scikit-learn's `MLPRegressor`.
```Python
from sklearn.neural_network import MLPRegressor
reg_sk = MLPRegressor(solver='adam', max_iter=1000) # Change hyperparameters to match my implementation for a fair comparison
reg_sk.fit(X_train, y_train)
y_predicted_sk = reg_sk.predict(X_test)
mse_sk = mean_squared_error(y_test, y_predicted_sk) # ~0.52
mse_pt = mean_squared_error(y_test, y_predicted) # ~0.51
```

On my machine, the Scikit-learn implementation produces a mean squared error of ~0.52 and my implementation with PyTorch produces a mean squared error of ~0.51, so I think it's fair to say my implementation is acheiving a similar level of performance.

### Multi-layer perceptron (for classification)
I also implemented an MLP-based classifier analgous to Scikit-learn's `MLPClassifier` class.
```Python
from sdpy import mlp, data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X, y, z = data.load_mlati(<path to h5 file>)
X_train, X_test, y_train, y_test = train_test_split(X, z, test_size=0.2) # I'm using the saccade type here (z) instead of the velocity waveforms (y)
clf = mlp.PyTorchMLPClassifier()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
```
And again for my own sanity I compared the performance of my implementation against Scikit-learn's `MPLClassifier`.
```
from sklearn.neural_network import MLPClassifier
clf_sk = MLPClassifier(solver='adam', max_iter=1000) # Set hyperparameters to match my implementation for a fair comparison
clf`_sk.fit(X_train, y_train)
y_predicted_sk = clf_sk.predict(X_test)
acc_sk = accuracy_score(y_test, y_predicted_sk) # ~0.75
acc_pt = accuracy_score(y_test, y_predicted) # ~0.83
```
On my machine, the Scikit-learn implementation is ~75% accuracte and my implementation with PyTorch is ~83% accurate, so again fairly similar performance. Here is a visualization that shows the confusion matrices (CM) for each model. The CMs are normalized to the sum of the columns such that the color of each cell indicates the fraction of predictions for each class that were correct (along the diagonal) or incorrect (above or below the diagonal). The raw frequencies are shown in black text within each cell.

<p align="center">
  <img src="docs/imgs/mlp_classifier_performance.png" width="700" alt="Animated demo">
</p>

Both models do pretty well decoding the type of saccade using neural activity.