# demo2_data_poisoning.py
# Requirements: numpy, scikit-learn
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create synthetic binary classification dataset
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=0)
# Split
idx = np.arange(X.shape[0])
np.random.shuffle(idx)
train_idx, test_idx = idx[:1400], idx[1400:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Train baseline classifier
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Clean test accuracy:", accuracy_score(y_test, clf.predict(X_test)))

# Create label-flip poisoning on a small fraction of training data
poison_fraction = 0.05
n_poison = int(poison_fraction * X_train.shape[0])
poison_idx = np.random.choice(X_train.shape[0], n_poison, replace=False)
X_train_poison = X_train.copy()
y_train_poison = y_train.copy()
# Flip labels for selected indices
y_train_poison[poison_idx] = 1 - y_train_poison[poison_idx]

# Retrain on poisoned data
clf_poison = LogisticRegression(max_iter=1000).fit(X_train_poison, y_train_poison)
print("Poisoned test accuracy:", accuracy_score(y_test, clf_poison.predict(X_test)))

# Quantify targeted degradation: measure accuracy on a subpopulation
# Define subpopulation where feature0 > median
mask = X_test[:,0] > np.median(X[:,0])
print("Clean subpopulation accuracy:", accuracy_score(y_test[mask], clf.predict(X_test[mask])))
print("Poisoned subpopulation accuracy:", accuracy_score(y_test[mask], clf_poison.predict(X_test[mask])))

# Simple detection via label distribution shift on training set
from collections import Counter
orig_label_dist = Counter(y_train.tolist())
poison_label_dist = Counter(y_train_poison.tolist())
print("Original label distribution:", orig_label_dist)
print("Poisoned label distribution:", poison_label_dist)
