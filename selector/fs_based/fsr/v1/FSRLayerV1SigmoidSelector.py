from torch import nn
from model.ClassifierModel import ClassifierModel
from selector.fs_based.fsr.v1.BaseFSRLayerV1Wrapper import BaseFSRLayerV1Wrapper
from selector.fs_based.fsr.BaseFSRSelectorWrapper import BaseFSRSelectorWrapper
from util.device_util import get_device

    
class FSRLayerV1SigmoidSelector(BaseFSRSelectorWrapper):
    def __init__(self, n_features, n_labels):
        super().__init__(n_features, n_labels)
        device = get_device()
        self._activation = nn.Sigmoid()
        self._model = ClassifierModel(n_features, n_labels).to(device)
        self._model = BaseFSRLayerV1Wrapper(self._model, n_features, n_labels, self._activation).to(device)

    def get_name() -> str:
        return  "Output Aware - Sigmoid"