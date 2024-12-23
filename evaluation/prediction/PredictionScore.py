from selector.BaseSelectorWrapper import BaseSelectorWrapper
from predictor.BasePredictor import BasePredictor
from util.classification_report_util import ClassificationScoreReport


class BasePredictionScore():
    def __init__(self, selector: BaseSelectorWrapper, selection_size: int, report: ClassificationScoreReport) -> None:
        self.selector_name = selector.get_class_name()
        self.report: ClassificationScoreReport = report
        self.selection_size = selection_size

class SelectorPredictionScore(BasePredictionScore):
    def __init__(self, selection_size: int, selector: BaseSelectorWrapper, report: ClassificationScoreReport) -> None:
        super().__init__(selector, selection_size, report)

class PredictorPredictionScore(BasePredictionScore):
    def __init__(self, selection_size: int, selector: BaseSelectorWrapper, predictor: BasePredictor, evaluator_name: str, report: ClassificationScoreReport) -> None:
        super().__init__(selector, selection_size, report)
        self.predictor_name = predictor.get_class_name()
        self.evaluator_name = evaluator_name