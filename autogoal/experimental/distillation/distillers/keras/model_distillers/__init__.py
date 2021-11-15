from typing import Dict, Type
from .distiller_base import DistillerBase
from .basic_no_distiller import BasicClassifierNoDistiller
from .response_classifier_distiller import ResponseClassifierDistiller
from .relation_distiller import RelationDistiller

MODEL_DISTILLERS: Dict[str, Type[DistillerBase]] = {
    "no_kd": BasicClassifierNoDistiller,
    "hinton": ResponseClassifierDistiller,
    "park": RelationDistiller,
}
DEFAULT_DISTILLER: Type[DistillerBase] = ResponseClassifierDistiller


def get_distiller(name: str) -> Type[DistillerBase]:
    return MODEL_DISTILLERS.get(name, DEFAULT_DISTILLER)
