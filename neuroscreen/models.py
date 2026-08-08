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

    def __init__(self, feature_names, k=3, components=("APc", "MLc", "STKc"),
                 bfs=False, n_runs=50, seed=0):
        self.components = components
        self.k = k
        self.feature_names = list(feature_names)
        self.bfs, self.n_runs, self.seed = bfs, n_runs, seed
        self.blocks = {
            c: [i for i, n in enumerate(self.feature_names) if n.startswith(c + "_")]
            for c in components
        }
        # Columns actually used per component; narrowed by B-FS when enabled.
        self.selected = {c: list(idx) for c, idx in self.blocks.items()}
        self.scalers, self.models, self.classes_ = {}, {}, None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes_ = sorted(set(y))
        for c, idx in self.blocks.items():
            if not idx:
                continue
            if self.bfs:
                # Select within this component's block, then map back to
                # absolute column indices.
                local = backward_feature_selection(
                    X[:, idx], y, k=self.k, n_runs=self.n_runs, seed=self.seed)
                idx = [idx[i] for i in local]
            self.selected[c] = list(idx)
            sc = StandardScaler().fit(X[:, idx])
            self.scalers[c] = sc
            self.models[c] = KNeighborsClassifier(
                n_neighbors=min(self.k, len(y))).fit(sc.transform(X[:, idx]), y)
        return self

    def predict(self, X):
        votes = []
        for c, idx in self.selected.items():
            if c not in self.models:
                continue
            Xs = self.scalers[c].transform(X[:, idx])
            votes.append(self.models[c].predict(Xs))
        votes = np.array(votes)                 # (n_components, n_samples)
        out = []
        for col in votes.T:
            out.append(Counter(col).most_common(1)[0][0])
        return np.array(out)

    def predict_proba(self, X):
        """Soft vote: average per-component class probabilities."""
        acc, n = None, 0
        for c, idx in self.selected.items():
            if c not in self.models:
                continue
            Xs = self.scalers[c].transform(X[:, idx])
            p = self.models[c].predict_proba(Xs)
            acc = p if acc is None else acc + p
            n += 1
        return acc / max(n, 1)


def build_gradient_boosting():
    """Paper-2 style gradient boosting over the fused feature vector."""
    return GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.08, max_depth=3, subsample=0.9,
        random_state=0)


# ---------------------------------------------------------------------------
# Backward feature selection (Paper 1, Sec. II)
# ---------------------------------------------------------------------------
def _knn_score(X, y, tr, te, cols, k):
    """Accuracy of a kNN trained on `cols` only, for one train/test split."""
    if not cols:
        return 0.0
    Xtr, Xte = X[np.ix_(tr, cols)], X[np.ix_(te, cols)]
    sc = StandardScaler().fit(Xtr)
    n_k = min(k, len(tr))
    m = KNeighborsClassifier(n_neighbors=n_k).fit(sc.transform(Xtr), y[tr])
    return float(np.mean(m.predict(sc.transform(Xte)) == y[te]))


def backward_feature_selection(X, y, k=3, n_runs=50, seed=0):
    """Paper 1's B-FS: greedy backward elimination under random sub-sampling.

    From the full d-dimensional set, d-1 candidate subsets are built (each
    dropping one feature); the feature missing from the best-scoring subset is
    eliminated. This repeats until one feature remains, and the retained subset
    is the one with the highest accuracy and -- on ties -- the fewest features.

    To limit selection bias the whole procedure is repeated `n_runs` times on
    random half-splits of the cohort. The paper does not state how the runs are
    combined; we keep features chosen by at least half of them, which is the
    natural frequency aggregation and degrades gracefully if a run is unlucky.

    Cost note: this is O(d^2) model fits per run -- ~45 s for a 21-feature COP
    block at n_runs=50. That makes it unaffordable *inside* a leave-one-subject-
    out loop (~8 h for the full cascade), and running it once over the whole
    cohort would leak test subjects into feature selection and inflate the
    reported accuracy. The headline DP-2 figure is therefore computed with
    `bfs=False`; enable it only for inspecting which features survive.

    Returns the selected column indices (never empty).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    votes = np.zeros(d, int)

    for _ in range(n_runs):
        # Stratified half-split so both classes appear on each side.
        tr, te = [], []
        for lab in np.unique(y):
            idx = rng.permutation(np.flatnonzero(y == lab))
            cut = max(1, len(idx) // 2)
            tr.extend(idx[:cut]); te.extend(idx[cut:])
        tr, te = np.array(tr), np.array(te)
        if len(te) == 0:
            continue

        current = list(range(d))
        best_score, best_set = -1.0, list(current)
        while len(current) > 1:
            scored = [(_knn_score(X, y, tr, te, [c for c in current if c != f], k), f)
                      for f in current]
            scored.sort(key=lambda t: -t[0])
            score, drop = scored[0]
            current = [c for c in current if c != drop]
            # ">=" so that, among equally accurate subsets, the smallest wins.
            if score >= best_score:
                best_score, best_set = score, list(current)
        votes[best_set] += 1

    keep = np.flatnonzero(votes >= max(1, n_runs // 2))
    if keep.size == 0:                       # nothing reached the threshold
        keep = np.array([int(np.argmax(votes))])
    return keep


# ---------------------------------------------------------------------------
# Diagnosis pathway 2 (Paper 1, Sec. II-C)
# ---------------------------------------------------------------------------
class DiagnosisPathway2:
    """Two-stage cascade for staging peripheral neuropathy.

      stage 1: (NN+AN) vs SN  -- is symptomatic neuropathy present?
      stage 2: NN vs AN       -- among the rest, is it asymptomatic neuropathy?

    Paper 1 reports this beating the single 3-class model (DP-1): 3/104
    misclassified at stage 1 and 2/72 at stage 2, with *no* asymptomatic
    patient missed -- the result that makes it viable as an early-detection
    screening tool, which DP-1's ~86% does not support.
    """

    NEG = "NN+AN"

    def __init__(self, feature_names, k=3, components=("APc", "MLc", "STKc"),
                 bfs=False, n_runs=50, seed=0):
        self.feature_names = list(feature_names)
        self.k, self.components = k, components
        self.bfs, self.n_runs, self.seed = bfs, n_runs, seed
        self.stage1 = self.stage2 = None
        self.classes_ = ["NN", "AN", "SN"]

    def _mk(self):
        return ComponentEnsemble(self.feature_names, k=self.k,
                                 components=self.components,
                                 bfs=self.bfs, n_runs=self.n_runs, seed=self.seed)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.stage1 = self._mk().fit(X, np.where(y == "SN", "SN", self.NEG))
        rest = y != "SN"
        # Stage 2 only ever sees non-symptomatic patients, matching deployment.
        self.stage2 = self._mk().fit(X[rest], y[rest]) if rest.any() else None
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        out = np.array(self.stage1.predict(X), dtype=object)
        rest = out != "SN"
        if rest.any() and self.stage2 is not None:
            out[rest] = self.stage2.predict(X[rest])
        return out


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
