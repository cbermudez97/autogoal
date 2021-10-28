from autogoal.kb import AlgorithmBase
from autogoal.contrib.transformers import BertEmbedding
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)
from typing import List

from .utils import (
    get_pre_trained_name,
    get_pre_trained_model,
    get_pre_trained_tokenizer,
)


class BertEmbeddingDistiller(AlgorithmDistillerBase):
    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        return isinstance(algorithm, BertEmbedding)

    def distill(
        self,
        algorithm: BertEmbedding,
        train_inputs: dict,
        test_inputs: dict,
        registry: List = None,
    ) -> AlgorithmBase:
        if not self.can_distill(algorithm):
            raise ValueError("Param 'algorithm' must be a BertEmbedding instance")

        model = algorithm.model
        distilled_model = algorithm.model
        distilled_tokenizer = algorithm.tokenizer
        try:
            model_name = model.config.name_or_path
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
        self, algorithm: BertEmbedding, new_model, new_tokenizer
    ) -> BertEmbedding:
        new_algorithm = BertEmbedding(
            merge_mode=algorithm.merge_mode, verbose=algorithm.verbose
        )
        new_algorithm.model = new_model
        new_algorithm.tokenizer = new_tokenizer
        return new_algorithm
