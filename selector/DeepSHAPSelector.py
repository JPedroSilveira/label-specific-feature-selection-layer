from config.general_config import EPOCHS, DEEP_SHAP_K
import shap
import numpy as np
from data.Dataset import Dataset
from model.ClassifierModel import ClassifierModel
from pytorch_helpers.PyTorchFit import pytorch_fit
from pytorch_helpers.PyTorchPredict import pytorch_predict_propabilities
from pytorch_helpers.PyTorchData import convert_nparray_to_tensor
from selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from selector.enum.PredictionMode import PredictionMode
from selector.enum.SelectionSpecificity import SelectionSpecificity
from util.array_util import get_weight_per_class_from_shap
from util.device_util import get_device


class DeepSHAPSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._k = 100
        self._representative_k = 50

    def get_name() -> str:
        return "DeepSHAP"

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.AVAILABLE

    def get_selection_specificities(self):
        return [SelectionSpecificity.PER_LABEL, SelectionSpecificity.GENERAL]

    def fit(self, train_dataset: Dataset, test_dataset: Dataset): 
        k = self._k
        if k > len(train_dataset.get_features()):
            k = len(train_dataset.get_features())
        representative_k = self._representative_k
        if representative_k > len(train_dataset.get_features()):
            representative_k = len(train_dataset.get_features())
        pytorch_fit(self._model, train_dataset)
        shap_samples = shap.sample(test_dataset.get_features(), k)
        shap_representative = shap.sample(train_dataset.get_features(), representative_k)
        explainer = shap.DeepExplainer(model=self._model, data=convert_nparray_to_tensor(shap_representative))
        self._shap_values = explainer.shap_values(convert_nparray_to_tensor(shap_samples))

    def predict(self, dataset: Dataset) -> np.ndarray:
        y_pred = self.predict_probabilities(dataset)
        return np.argmax(y_pred, 1)
    
    def predict_probabilities(self, dataset: Dataset, use_softmax: bool=True) -> np.ndarray:
        return pytorch_predict_propabilities(self._model, dataset.get_features(), use_softmax)
    
    def get_general_weights(self) -> np.ndarray:
        return np.max(self.get_weights_per_class(), axis=0)
    
    def get_weights_per_class(self) -> np.ndarray:
        return get_weight_per_class_from_shap(np.abs(self._shap_values).mean(0))