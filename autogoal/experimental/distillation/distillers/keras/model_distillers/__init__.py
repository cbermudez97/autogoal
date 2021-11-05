from typing import Dict, Type
from .distiller_base import DistillerBase
from .response_distiller import ResponseDistiller

MODEL_DISTILLERS: Dict[str : Type[DistillerBase]] = {
    "hinton": ResponseDistiller,
}
DEFAULT_DISTILLER: Type[DistillerBase] = ResponseDistiller


def get_distiller(name: str) -> Type[DistillerBase]:
    return MODEL_DISTILLERS.get(name, DEFAULT_DISTILLER)
