import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocket


class MiniRocketClassifier:
    def __init__(self, num_kernels=10000, alphas=None, random_state=0, n_jobs=-1):
        self.num_kernels = num_kernels
        self.alphas = np.logspace(-1, 6, 15) if alphas is None else np.asarray(alphas)
        self.random_state = random_state
        self.transform = MiniRocket(num_kernels=num_kernels, random_state=random_state, n_jobs=n_jobs)
        self.scaler = StandardScaler()
        self.ridge = None
        self.alpha_scores = {}

    def features(self, x):
        return self.scaler.transform(np.asarray(self.transform.transform(x), dtype=np.float32))

    def fit(self, x, y, x_val=None, y_val=None):
        feats = self.scaler.fit_transform(np.asarray(self.transform.fit_transform(x), dtype=np.float32))
        if x_val is None:
            self.ridge = RidgeClassifier(alpha=float(np.median(self.alphas))).fit(feats, y)
            return self
        val_feats = self.features(x_val)
        best = None
        for alpha in self.alphas:
            ridge = RidgeClassifier(alpha=float(alpha)).fit(feats, y)
            score = ridge.score(val_feats, y_val)
            self.alpha_scores[float(alpha)] = float(score)
            if best is None or score > best[0]:
                best = (score, ridge)
        self.ridge = best[1]
        return self

    def decision_function(self, x):
        return self.ridge.decision_function(self.features(x))

    def predict(self, x):
        return self.ridge.predict(self.features(x))

    @property
    def n_features(self):
        return self.ridge.coef_.shape[1]

    @property
    def alpha(self):
        return self.ridge.alpha
