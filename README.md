# Introduction
This is a repository that showcases a set of neural decoders implemented with PyTorch. The models are being used to decode eye movement kinematics (velocity) using neural data (single-unit spiking).

# Data
### Interfacing with the Mlati dataset
For these analyses I'm using a dataset I collected during my PhD which can be downloaded [here](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqcfn). Briefly, the data for each experiment in this dataset is stored in an h5 file with the filenameing convention "\<date>_\<animal>_store.hdf." I created the `sdpy.data` module to interface with the Mlati dataset using these h5 files. Here is an example that demonstrates how to load the neural activity and eye velocity signals for a single experiment.
```Python
from sdpy import data
X, y, z = data.load_mlati(<path to h5 file>)
```
The `sdpy.data.load_mlati` function returns 3 variables:
- `X` - The `X` variable contains neural activity in a window of time around the onset of saccades. The matrix is organized neuron-major/bin-minor such that each row is formed by the concatenation of multiple single-trial peri-stimulus time histograms (PETHs). Therefore, the size of the second dimension of `X` is the number of time bins in the PETHs x the number of neurons in the recording.
- `y` - The `y` variable contains the velocity waveforms (in deg/s) for each saccade.
- `z` - The `z` variable indicates the type of saccade: 0 (Null), 1 (Nasal), or 2 (Temporal).

Here is a plot that visualizes the first 500 columns of `X` (i.e, the first 25 neurons), all values of `y`, and all values of `z`. Samples are grouped by the type of saccade along the y-axis, and for `X`, boundaries between individual neurons are indicated with a dotted black line.

<p align="center">
  <img src="docs/imgs/Xyz.png" width="700" alt="Animated demo">
</p>

# Modeling
Below are some examples showcasing various implementations of machine learning models and techniques applied to this dataset.

### Predicting saccade type with a multi-layer perceptron
I wanted to see if I could use peri-saccadic single-unit spiking to predict 1.) if the animal made a saccade and 2.) the direction of each saccade that was made. To do this, I implemented an Multi-layer perceptron (MLP) classifier with PyTorch. You can find the implementation [here](https://github.com/jbhunt/saccade-decoder/blob/7556287ea31d8e364d5fe4fc0428b654b108bed8/sdpy/mlp.py#L197). My implementationd - `sdpy.mpl.PyTorchMLPClassifier` - emulates the design and interface used by scikit-learn. For example, this class has both a `fit` and `predict` method.
```Python
from sdpy import mlp, data
from sklearn.model_selection import train_test_split
X, y, z = data.load_mlati(<path to h5 file>)
X_train, X_test, y_train, y_test = test_train_split(X, z, test_size=0.2)
clf_pt = mlp.PyTorchMLPClassifier()
clf_pt.fit(X_train, y_train)
y_predicted = clf_pt.predict(X_test)
```
And as a sanity check, I compared the cross-validated performance of my implementation with scikit-learn's `sklearn.neural_network.MPLClassifier` class.
```Python
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, accuracy_score
from sklearn.metrics import make_scorer
clf_sk = MLPClassifier(solver='adam', max_iter=1000) # Set hyperparameters to match my implementation for a fair comparison
acc_score_pt = cross_val_score(clf_pt, X, z, scoring=make_scorer(accuracy_score), cv=5).mean() # returns 0.85
acc_score_sk = cross_val_score(clf_sk, X, z, scoring=make_scorer(accuracy_score), cv=5).mean() # returns 0.75
```
On my machine, the Scikit-learn implementation is ~75% accuracte and my implementation with PyTorch is ~83% accurate. Here is a visualization that shows the confusion matrices (CM) for each model. The CMs are normalized to the sum of the columns such that the color of each cell indicates the fraction of predictions for each class that were correct (along the diagonal) or incorrect (above or below the diagonal). The raw frequencies are shown in black text within each cell.

<p align="center">
  <img src="docs/imgs/mlp_classifier_performance.png" width="700" alt="Animated demo">
</p>

Both models do pretty well decoding the type of saccade using neural activity with sklearn's MLP struggling a little bit more to predict null events (z=0).