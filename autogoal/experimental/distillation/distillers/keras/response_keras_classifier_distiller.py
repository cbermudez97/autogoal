from tensorflow.keras import Model, losses, optimizers
from .model_distillers import ResponseClassifierDistiller
from .keras_classifier_distiller import _KerasClassifierDistiller


class ResponseKerasClassifierDistiller(_KerasClassifierDistiller):
    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> ResponseClassifierDistiller:
        distiller = ResponseClassifierDistiller(
            student=student_model, teacher=teacher_model
        )
        distiller.compile(
            optimizers.RMSprop(),
            ["accuracy"],
            losses.categorical_crossentropy,
            losses.categorical_crossentropy,
            alpha=self._distiller_alpha,
            temperature=self._distiller_temperature,
        )
        return distiller
