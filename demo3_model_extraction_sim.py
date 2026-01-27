# demo3_model_extraction_sim.py
# Requirements: numpy, scikit-learn
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Create small target model trained on a secret dataset
X_secret, y_secret = make_classification(n_samples=2000, n_features=10, n_informative=6, random_state=1)
target = RandomForestClassifier(n_estimators=50, random_state=1)
target.fit(X_secret, y_secret)

# Simulate black-box API: function that returns predicted label (no probabilities)
def target_api(query_batch):
    return target.predict(query_batch)

# Attacker builds a surrogate by querying the target with inputs drawn from a public distribution
# Public distribution: noisy samples around the secret data mean
mu = X_secret.mean(axis=0)
cov = np.cov(X_secret.T)
np.random.seed(2)
queries = np.random.multivariate_normal(mu, cov + 1e-1*np.eye(mu.shape[0]), size=1000)
responses = target_api(queries)

# Train surrogate model on (queries, responses)
surrogate = DecisionTreeClassifier(max_depth=10, random_state=2)
surrogate.fit(queries, responses)

# Evaluate fidelity on a held-out set sampled from secret distribution
X_hold, y_hold = make_classification(n_samples=500, n_features=10, n_informative=6, random_state=3)
target_preds = target_api(X_hold)
surrogate_preds = surrogate.predict(X_hold)
fidelity = (surrogate_preds == target_preds).mean()
target_acc = (target_preds == y_hold).mean()
surrogate_acc = (surrogate_preds == y_hold).mean()
print("Target accuracy on holdout:", target_acc)
print("Surrogate accuracy on holdout:", surrogate_acc)
print("Surrogate fidelity to target:", fidelity)

# Defensive countermeasure: simple output randomisation simulation (probabilistic labelling)
def target_api_noisy(query_batch, flip_prob=0.05):
    preds = target.predict(query_batch)
    flip = np.random.rand(preds.shape[0]) < flip_prob
    preds[flip] = 1 - preds[flip]
    return preds

noisy_responses = target_api_noisy(queries, flip_prob=0.1)
surrogate_noisy = DecisionTreeClassifier(max_depth=10, random_state=2)
surrogate_noisy.fit(queries, noisy_responses)
surrogate_noisy_preds = surrogate_noisy.predict(X_hold)
fidelity_noisy = (surrogate_noisy_preds == target_preds).mean()
print("Surrogate fidelity when target adds noise:", fidelity_noisy)
