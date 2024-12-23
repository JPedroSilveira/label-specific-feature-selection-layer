import numpy as np
import pandas as pd
from data.Dataset import Dataset
from sklearn.feature_selection import mutual_info_regression
from selector.BaseWeightSelectorWrapper import BaseWeightSelectorWrapper
from selector.enum.PredictionMode import PredictionMode
from selector.enum.SelectionSpecificity import SelectionSpecificity
from util.print_util import print_load_bar
from mrmr import mrmr_classif


class MRMRFeatureSelectorWrapper(BaseWeightSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        self.n_features_to_select = n_features

    def get_prediction_mode(self) -> PredictionMode:
        return PredictionMode.UNAVAILABLE
    
    def get_selection_specificities(self):
        return [SelectionSpecificity.GENERAL]
    
    def get_name() -> str:
        return "MRMR"

    def fit(self, train_dataset: Dataset, _: Dataset): 
        n_features = train_dataset.get_n_features()
        n_features_to_select = self.n_features_to_select
        selected = []
        remaining = list(range(n_features))
        selected_to_remaining_mi = np.zeros((n_features_to_select - 1, n_features))
        selected_to_class_mi = []
        features_to_class_mi = np.array(mutual_info_regression(X, y))
        first_feature = np.argmax(features_to_class_mi)
        selected.append(first_feature)
        remaining.remove(first_feature)
        highest_mi = np.max(features_to_class_mi)
        selected_to_class_mi.append(highest_mi)
        for num_selected in range(1, n_features_to_select):
            print_load_bar(num_selected, n_features_to_select)
            print("LOOP")
            remaining_features = X[:, remaining]
            last_selected_feature = X.T[selected[-1]]
            print("BEFORE MUTUAL")
            last_selected_to_remaining_mi = mutual_info_regression(remaining_features, last_selected_feature)
            print("AFTER MUTUAL")
            selected_to_remaining_mi[num_selected - 1, remaining] = last_selected_to_remaining_mi
            redundancy = np.mean(selected_to_remaining_mi[:num_selected, remaining], axis=0)
            mRMR_scores = features_to_class_mi[remaining] - redundancy
            selected_idx = remaining[np.argmax(mRMR_scores)]
            selected.append(selected_idx)
            remaining.remove(selected_idx)
            higher_score = np.max(mRMR_scores)
            selected_to_class_mi.append(higher_score)
        self._weights = np.zeros(n_features, dtype='d')
        self._weights[selected] = selected_to_class_mi

    def get_general_weights(self) -> np.ndarray:
        return self._weights