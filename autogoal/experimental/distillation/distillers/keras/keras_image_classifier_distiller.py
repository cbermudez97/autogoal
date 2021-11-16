import abc
from typing import Any, Dict, List, Type
from autogoal.experimental.distillation.distillers.keras.keras_classifier_distiller import (
    _KerasClassifierDistiller,
)
from autogoal.grammar._graph import Graph
from autogoal.contrib.keras import KerasImageClassifier
from autogoal.experimental.distillation.compressors import find_compressors
from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)
from autogoal.experimental.distillation.distillers.keras.model_distillers import (
    DistillerBase,
)
from autogoal.kb import AlgorithmBase
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN


class _KerasImageClassifierDistiller(_KerasClassifierDistiller):
    def __init__(
        self,
        epochs: int = 10,
        early_stop: int = 3,
        distiller_alpha: float = 0.9,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            early_stop=early_stop,
            distiller_alpha=distiller_alpha,
            batch_size=batch_size,
            verbose=verbose,
        )

    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        return algorithm.__class__ == KerasImageClassifier

    def distill(
        self,
        algorithm: KerasImageClassifier,
        train_inputs: dict,
        test_inputs: dict,
        registry: List[Type[ModelCompressorBase]] = None,
        compressors_kwargs: Dict[Type[ModelCompressorBase], Dict[str, Any]] = {},
    ) -> KerasImageClassifier:
        train_x, train_y = train_inputs.values()
        test_x, test_y = test_inputs.values()

        if not self.can_distill(algorithm):
            raise ValueError("Param 'algorithm' must be a KerasClassifier algorithm")

        if algorithm._classes is None:
            raise ValueError("Param 'algorithm' must be trained before distillation")

        train_y = to_categorical([algorithm._classes[tag] for tag in train_y])
        test_y = to_categorical([algorithm._classes[tag] for tag in test_y])

        if not registry:
            registry = find_compressors()

        teacher_model: Model = algorithm.model

        compressed_model: Model = None
        evaluation_score = 0.0
        for compressor_cls in registry:
            compressor_kwargs = compressors_kwargs.get(compressor_cls, {})
            compressor: ModelCompressorBase = compressor_cls(**compressor_kwargs,)
            candidate_model: Model = None
            candidate_score = 0.0
            if not compressor.can_compress(teacher_model):
                continue
            # try:
            candidate_model = compressor.compress(teacher_model)
            distiller = self.build_distiller(candidate_model, teacher_model)
            candidate_model = distiller.student
            train_data = algorithm.preprocessor.flow(
                train_x, train_y, batch_size=self._distiller_batch_size
            )
            distiller.fit(
                train_data,
                steps_per_epoch=len(train_x) // self._distiller_batch_size,
                epochs=self._epochs,
                callbacks=[
                    EarlyStopping(
                        monitor="loss",
                        patience=self._early_stop,
                        restore_best_weights=True,
                    ),
                    TerminateOnNaN(),
                ],
                verbose=self._distiller_verbose,
            )
            candidate_score, _, _, _ = distiller.evaluate(test_x, test_y, verbose=0,)
            # except Exception as e:
            #     continue
            if candidate_score > evaluation_score:
                compressed_model = candidate_model

        if compressed_model:
            return self.build_keras_classifier_from(algorithm, compressed_model)

        return None

    @abc.abstractmethod
    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> DistillerBase:
        pass

    def build_keras_classifier_from(
        self, original: KerasImageClassifier, new_model: Model
    ) -> KerasImageClassifier:
        # TODO: Improve the algorithm construction
        classifier = KerasImageClassifier(
            original.preprocessor, original.optimizer, grammar=original._grammar,
        )
        classifier._model = new_model
        classifier._classes = original._classes
        classifier._inverse_classes = original._inverse_classes
        classifier.eval()
        graph: Graph = Graph()
        if not original._graph is None:
            original_order = original._graph.build_order()
            nodes = []
            for layer, _ in original_order:
                nodes.insert(0, new_model.get_layer(name=layer.name))
            graph.add_nodes_from(nodes)
        classifier._graph = graph
        return classifier
