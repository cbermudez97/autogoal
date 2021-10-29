from autogoal.kb import AlgorithmBase
from autogoal.contrib.transformers import BertEmbedding
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)
from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)
from typing import List, Type

from .utils import (
    get_pre_trained_name,
    get_pre_trained_model,
    get_pre_trained_tokenizer,
)


class BertEmbeddingDistiller(AlgorithmDistillerBase):
    def __init__(self) -> None:
        super().__init__()

    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        # TODO: Add other use cases
        return self._is_seq_alg(algorithm)

    def _is_seq_alg(self, algorithm: AlgorithmBase) -> bool:
        # TODO: Improve SeqAlgorithm detection
        return "inner" in algorithm.__dict__ and isinstance(
            algorithm.__dict__["inner"], BertEmbedding
        )

    def extract_model(self, algorithm: AlgorithmBase):
        if self._is_seq_alg(algorithm):
            return algorithm.inner.model
        return None

    def extract_tokenizer(self, algorithm: AlgorithmBase):
        if self._is_seq_alg(algorithm):
            return algorithm.inner.tokenizer
        return None

    def distill(
        self,
        algorithm: AlgorithmBase,
        train_inputs: dict,
        test_inputs: dict,
        registry: List[Type[ModelCompressorBase]] = None,
    ) -> AlgorithmBase:
        if not self.can_distill(algorithm):
            raise ValueError("Param 'algorithm' must use a BertEmbedding instance")

        alg_model = self.extract_model(algorithm)
        alg_tokenizer = self.extract_tokenizer(algorithm)

        distilled_model = alg_model
        distilled_tokenizer = alg_tokenizer
        try:
            model_name = alg_model.config.name_or_path
            distilled_model_name = get_pre_trained_name(model_name)
            if distilled_model_name:
                distilled_model = get_pre_trained_model(distilled_model_name)
                distilled_tokenizer = get_pre_trained_tokenizer(distilled_model_name)
            else:
                raise Exception("No pre-trained model found.")
        except:
            # TODO: Implement custom distillation process
            pass

        distilled_algorithm = self.build_distilled(
            algorithm, distilled_model, distilled_tokenizer
        )
        return distilled_algorithm

    def build_distilled(
        self, algorithm: AlgorithmBase, new_model, new_tokenizer
    ) -> AlgorithmBase:
        if self._is_seq_alg(algorithm):
            return self._build_seq_alg(algorithm, new_model, new_tokenizer)
        return algorithm

    def _build_seq_alg(self, algorithm: AlgorithmBase, new_model, new_tokenizer):
        new_inner = BertEmbedding(
            merge_mode=algorithm.inner.merge_mode, verbose=algorithm.inner.verbose
        )
        new_inner.model = new_model
        new_inner.tokenizer = new_tokenizer
        new_algorithm = algorithm.__class__()
        new_algorithm.inner = new_inner
        return new_algorithm
