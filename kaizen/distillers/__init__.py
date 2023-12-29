from kaizen.distillers.base import base_distill_wrapper
from kaizen.distillers.contrastive import contrastive_distill_wrapper
from kaizen.distillers.decorrelative import decorrelative_distill_wrapper
from kaizen.distillers.knowledge import knowledge_distill_wrapper
from kaizen.distillers.predictive import predictive_distill_wrapper
from kaizen.distillers.predictive_mse import predictive_mse_distill_wrapper


__all__ = [
    "base_distill_wrapper",
    "contrastive_distill_wrapper",
    "decorrelative_distill_wrapper",
    "nearest_neighbor_distill_wrapper",
    "predictive_distill_wrapper",
    "predictive_mse_distill_wrapper",
]

DISTILLERS = {
    "base": base_distill_wrapper,
    "contrastive": contrastive_distill_wrapper,
    "decorrelative": decorrelative_distill_wrapper,
    "knowledge": knowledge_distill_wrapper,
    "predictive": predictive_distill_wrapper,
    "predictive_mse": predictive_mse_distill_wrapper,
}
