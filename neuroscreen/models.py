"""
Classification models for diabetic-neuropathy screening.

Reproduces the two learning strategies from the source papers and adds the
cross-modal fusion that is the novel contribution of this project:

  * Paper 1 -> ``ComponentEnsemble``: one kNN per COP component (APc / MLc /
    STKc) combined by majority voting. Each component sees only its own
    feature block, mirroring the paper's design that the components carry
    complementary rather than redundant information.
  * Paper 2 -> a Gradient Boosting classifier over the fused feature vector
    (COP + wearable gait / pressure / HRV), with feature importances used for
    SHAP-style explanations.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class ComponentEnsemble:
    """Majority-voting ensemble of per-COP-component kNN classifiers (Paper 1)."""

    def __init__(self, feature_names, k=3, components=("APc", "MLc", "STKc")):
        self.components = components
        self.k = k
        self.feature_names = list(feature_names)
        self.blocks = {
            c: [i for i, n in enumerate(self.feature_names) if n.startswith(c + "_")]
            for c in components
        }
        self.scalers, self.models, self.classes_ = {}, {}, None

    def fit(self, X, y):
        self.classes_ = sorted(set(y))
        for c, idx in self.blocks.items():
            sc = StandardScaler().fit(X[:, idx])
            self.scalers[c] = sc
            self.models[c] = KNeighborsClassifier(n_neighbors=self.k).fit(
                sc.transform(X[:, idx]), y)
        return self

    def predict(self, X):
        votes = []
        for c, idx in self.blocks.items():
            Xs = self.scalers[c].transform(X[:, idx])
            votes.append(self.models[c].predict(Xs))
        votes = np.array(votes)                 # (n_components, n_samples)
        out = []
        for col in votes.T:
            out.append(Counter(col).most_common(1)[0][0])
        return np.array(out)

    def predict_proba(self, X):
        """Soft vote: average per-component class probabilities."""
        acc = None
        for c, idx in self.blocks.items():
            Xs = self.scalers[c].transform(X[:, idx])
            p = self.models[c].predict_proba(Xs)
            acc = p if acc is None else acc + p
        return acc / len(self.blocks)


def build_gradient_boosting():
    """Paper-2 style gradient boosting over the fused feature vector."""
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=3, subsample=0.9,
        random_state=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def per_class_metrics(y_true, y_pred, labels):
    """Accuracy plus per-class precision / sensitivity / specificity."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float(np.mean(y_true == y_pred))
    rows = {}
    for lab in labels:
        tp = int(np.sum((y_pred == lab) & (y_true == lab)))
        fp = int(np.sum((y_pred == lab) & (y_true != lab)))
        fn = int(np.sum((y_pred != lab) & (y_true == lab)))
        tn = int(np.sum((y_pred != lab) & (y_true != lab)))
        rows[lab] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "support": int(np.sum(y_true == lab)),
        }
    return acc, rows


def confusion(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M
