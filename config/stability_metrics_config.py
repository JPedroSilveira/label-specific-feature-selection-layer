from typing import Type
from evaluation.stability.metric.BaseStabilityMetric import BaseStabilityMetric
from evaluation.stability.metric.JaccardMetric import JaccardMetric
from evaluation.stability.metric.SpearmanMetric import SpearmanMetric
from evaluation.stability.metric.PearsonMetric import PearsonMetric
from evaluation.stability.metric.KunchevaMetric import KunchevaMetric


STABILITY_METRICS_TYPES: list[Type[BaseStabilityMetric]] = [JaccardMetric, SpearmanMetric, PearsonMetric]