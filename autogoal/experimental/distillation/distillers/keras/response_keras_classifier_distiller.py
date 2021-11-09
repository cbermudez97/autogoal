from tensorflow.keras import Model, losses, optimizers
from autogoal.experimental.distillation.distillers.keras.model_distillers import (
    ResponseClassifierDistiller,
)
from autogoal.experimental.distillation.distillers.keras.keras_classifier_distiller import (
    _KerasClassifierDistiller,
)


class ResponseKerasClassifierDistiller(_KerasClassifierDistiller):
    def __init__(
        self,
        epochs: int = 10,
        early_stop: int = 3,
        distiller_alpha: float = 0.9,
        distiller_temperature: float = 1,
    ):
        super().__init__(
            epochs=epochs, early_stop=early_stop, distiller_alpha=distiller_alpha
        )
        self._distiller_temperature = distiller_temperature

    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> ResponseClassifierDistiller:
        distiller = ResponseClassifierDistiller(
            student=student_model, teacher=teacher_model
        )
        distiller.compile(
            teacher_model.optimizer,
            ["accuracy"],
            losses.categorical_crossentropy,
            losses.categorical_crossentropy,
            alpha=self._distiller_alpha,
            temperature=self._distiller_temperature,
        )
        return distiller
