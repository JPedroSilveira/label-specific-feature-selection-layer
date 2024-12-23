import numpy as np
from sklearn.svm import SVC
from predictor.BasePredictor import BasePredictor


class SVCPredictor(BasePredictor):
    def __init__(self, n_features, n_labels) -> None:
        self._model = SVC()

    def get_name() -> str:
        return "SVC"

    def fit(self, X: np.ndarray, y: np.ndarray): 
        self._model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self._model.predict(X)
        return y_pred