from tensorflow.keras import Model, losses, optimizers
from autogoal.experimental.distillation.distillers.keras.model_distillers import (
    BasicClassifierNoDistiller,
)
from autogoal.experimental.distillation.distillers.keras.keras_classifier_distiller import (
    _KerasClassifierDistiller,
)


class BasicKerasClassifierCompressor(_KerasClassifierDistiller):
    def __init__(
        self,
        epochs: int = 10,
        early_stop: int = 3,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        super().__init__(
            epochs=epochs,
            early_stop=early_stop,
            distiller_alpha=0,
            batch_size=batch_size,
            verbose=verbose,
        )

    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> BasicClassifierNoDistiller:
        distiller = BasicClassifierNoDistiller(
            student=student_model, teacher=teacher_model
        )
        distiller.compile(
            teacher_model.optimizer, ["accuracy"], losses.categorical_crossentropy,
        )
        return distiller
