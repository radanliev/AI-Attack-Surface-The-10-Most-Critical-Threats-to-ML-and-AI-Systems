# demo4_federated_byzantine.py
# Requirements: numpy, scikit-learn
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from copy import deepcopy

# Create synthetic data and split into clients
X, y = make_classification(n_samples=2000, n_features=20, n_informative=10, random_state=0)
n_clients = 5
client_data = []
indices = np.array_split(np.arange(X.shape[0]), n_clients)
for inds in indices:
    client_data.append((X[inds], y[inds]))

# Initialise global model parameters (SGDClassifier for partial_fit)
classes = np.unique(y)
global_model = SGDClassifier(max_iter=1, tol=None, learning_rate='constant', eta0=0.01, random_state=0)

# Warm start using a small sample
sample_X = np.vstack([d[0][:20] for d in client_data])
sample_y = np.hstack([d[1][:20] for d in client_data])
global_model.partial_fit(sample_X, sample_y, classes=classes)

def get_model_params(model):
    return deepcopy((model.coef_.copy(), model.intercept_.copy()))

def set_model_params(model, params):
    coef, intercept = params
    model.coef_ = coef.copy()
    model.intercept_ = intercept.copy()

# Federated averaging routine
def federated_round(global_model, client_data, byzantine_client=None, scale_attack=10.0):
    global_params = get_model_params(global_model)
    client_params = []
    for i, (Xi, yi) in enumerate(client_data):
        local_model = SGDClassifier(max_iter=1, tol=None, learning_rate='constant', eta0=0.01)
        set_model_params(local_model, global_params)
        local_model.partial_fit(Xi, yi, classes=classes)
        params = get_model_params(local_model)
        if i == byzantine_client:
            # Byzantine behavior: scale parameter update to poison aggregation
            params = (params[0] * scale_attack, params[1] * scale_attack)
        client_params.append(params)
    # Average
    avg_coef = np.mean([p[0] for p in client_params], axis=0)
    avg_intercept = np.mean([p[1] for p in client_params], axis=0)
    set_model_params(global_model, (avg_coef, avg_intercept))

# Evaluate baseline and with Byzantine client
X_test, y_test = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)

# Baseline training without Byzantine
global_model_clean = deepcopy(global_model)
for rnd in range(5):
    federated_round(global_model_clean, client_data, byzantine_client=None)
preds_clean = global_model_clean.predict(X_test)
print("Clean federated accuracy:", accuracy_score(y_test, preds_clean))

# With Byzantine client 0
global_model_byz = deepcopy(global_model)
for rnd in range(5):
    federated_round(global_model_byz, client_data, byzantine_client=0, scale_attack=8.0)
preds_byz = global_model_byz.predict(X_test)
print("Byzantine federated accuracy (client 0 attack):", accuracy_score(y_test, preds_byz))

# Simple robust aggregation: median of client parameter updates
def federated_round_median(global_model, client_data, byzantine_client=None, scale_attack=10.0):
    global_params = get_model_params(global_model)
    client_updates = []
    for i, (Xi, yi) in enumerate(client_data):
        local_model = SGDClassifier(max_iter=1, tol=None, learning_rate='constant', eta0=0.01)
        set_model_params(local_model, global_params)
        local_model.partial_fit(Xi, yi, classes=classes)
        params = get_model_params(local_model)
        update_coef = params[0] - global_params[0]
        update_intercept = params[1] - global_params[1]
        if i == byzantine_client:
            update_coef = update_coef * scale_attack
            update_intercept = update_intercept * scale_attack
        client_updates.append((update_coef, update_intercept))
    median_coef = np.median(np.array([u[0] for u in client_updates]), axis=0)
    median_intercept = np.median(np.array([u[1] for u in client_updates]), axis=0)
    new_coef = global_params[0] + median_coef
    new_intercept = global_params[1] + median_intercept
    set_model_params(global_model, (new_coef, new_intercept))

# Evaluate with median aggregation
global_model_median = deepcopy(global_model)
for rnd in range(5):
    federated_round_median(global_model_median, client_data, byzantine_client=0, scale_attack=8.0)
preds_median = global_model_median.predict(X_test)
print("Median-aggregated federated accuracy (with Byzantine):", accuracy_score(y_test, preds_median))
