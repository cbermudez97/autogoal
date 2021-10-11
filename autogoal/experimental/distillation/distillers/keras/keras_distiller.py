from typing import List

from autogoal.contrib.keras import KerasClassifier
from autogoal.experimental.distillation.compressors import find_compressors
from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)
from autogoal.kb import AlgorithmBase
from tensorflow.keras import Model
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping, TerminateOnNaN

from .distiller import _Distiller


class KerasDistiller(AlgorithmDistillerBase):
    def __init__(
        self, epochs=10, early_stop=3,
    ):
        super().__init__()
        self._epochs = epochs
        self._early_stop = early_stop

    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        return isinstance(algorithm, KerasClassifier)

    def distill(
        self,
        algorithm: KerasClassifier,
        train_x,
        train_y,
        test_x,
        test_y,
        registry: List = None,
    ) -> KerasClassifier:
        if not self.can_distill(algorithm):
            raise ValueError("Param 'algorithm' must be a KerasClassifier algorithm")

        if not registry:
            registry = find_compressors()

        teacher_model: Model = algorithm.model

        compressed_model: Model = None
        evaluation_score = 0.0
        for compressor_cls in registry:
            compressor: ModelCompressorBase = compressor_cls()
            candidate_model: Model = None
            candidate_score = 0.0
            if not compressor.can_compress(teacher_model):
                continue
            try:
                candidate_model = compressor.compress(teacher_model)
                distiller = _Distiller(student=candidate_model, teacher=teacher_model)
                candidate_model = distiller.student
                distiller.compile(
                    optimizers.RMSprop(),
                    ["accuracy"],
                    losses.categorical_crossentropy,
                    losses.categorical_crossentropy,
                    alpha=0.9,
                    temperature=1,
                )
                distiller.fit(
                    train_x,
                    train_y,
                    epochs=self._epochs,
                    callbacks=[
                        EarlyStopping(
                            monitor="loss",
                            patience=self._early_stop,
                            restore_best_weights=True,
                        ),
                        TerminateOnNaN(),
                    ],
                    verbose=0,
                )
                candidate_score, _, _, _ = distiller.evaluate(
                    test_x, test_y, verbose=0,
                )
            except Exception as e:
                continue
            if candidate_score > evaluation_score:
                compressed_model = candidate_model

        if compressed_model:
            return self.build_keras_classifier_from(algorithm, compressed_model)

        return None

    def build_keras_classifier_from(
        self, original: KerasClassifier, new_model: Model
    ) -> KerasClassifier:
        # TODO: Improve the algorithm construction
        classifier = KerasClassifier(original.optimizer, grammar=original._grammar)
        classifier._model = new_model
        classifier._classes = original._classes
        classifier._inverse_classes = original._inverse_classes
        classifier.eval()
        return classifier
