from typing import Type
from predictor.BasePredictor import BasePredictor
from predictor.SequentialPredictor import SequentialPredictor
from predictor.SVCPredictor import SVCPredictor
from typing import Type
from evaluation.prediction.prediction_evaluator.BasePredictionEvaluator import BasePredictionEvaluator
from evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByRank import GeneralPredictionEvaluatorByRank
from evaluation.prediction.prediction_evaluator.GeneralPredictionEvaluatorByWeight import GeneralPredictionEvaluatorByWeight


PREDICTION_EVALUATOR_TYPES: list[Type[BasePredictionEvaluator]] = [GeneralPredictionEvaluatorByRank, GeneralPredictionEvaluatorByWeight]
PREDICTOR_TYPES: list[Type[BasePredictor]] = [SVCPredictor]#, SequentialPredictor] # [SVCPredictor] #[SequentialPredictor] # SVCPredictor
