from kaizen.distiller_factories.base import base_frozen_model_factory
from kaizen.distiller_factories.contrastive import contrastive_distill_factory
from kaizen.distiller_factories.decorrelative import decorrelative_distill_factory
from kaizen.distiller_factories.knowledge import knowledge_distill_factory
from kaizen.distiller_factories.predictive import predictive_distill_factory
from kaizen.distiller_factories.predictive_mse import predictive_mse_distill_factory
from kaizen.distiller_factories.soft_label import soft_label_distill_factory


__all__ = [
    "base_frozen_model_factory",
    "contrastive_distill_factory",
    "decorrelative_distill_factory",
    "nearest_neighbor_distill_wrapper", # TODO: Check what this is
    "knowledge_distill_factory",
    "predictive_distill_factory",
    "predictive_mse_distill_factory",
    "soft_label_distill_factory"
]

DISTILLER_FACTORIES = {
    "base": base_frozen_model_factory,
    "contrastive": contrastive_distill_factory,
    "decorrelative": decorrelative_distill_factory,
    "knowledge": knowledge_distill_factory,
    "predictive": predictive_distill_factory,
    "predictive_mse": predictive_mse_distill_factory,
    "soft_label": soft_label_distill_factory
}
