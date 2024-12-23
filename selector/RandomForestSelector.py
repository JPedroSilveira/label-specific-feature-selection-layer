import numpy as np
from sklearn.ensemble import RandomForestClassifier
from data.Dataset import Dataset
from selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from selector.enum.PredictionMode import PredictionMode
from selector.enum.SelectionSpecificity import SelectionSpecificity


class RandomForestSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self._model = RandomForestClassifier()

    def get_name() -> str:
        return "RandomForest"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE
    
    def get_selection_specificities(self):
        return [SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset):
        self._model.fit(train_dataset.get_features(), train_dataset.get_labels())
    
    def predict(self, dataset: Dataset):
        return self._model.predict(dataset.get_features())
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return self._model.predict_proba(dataset.get_features())
    
    def get_general_weights(self) -> np.ndarray:
        return self._model.feature_importances_