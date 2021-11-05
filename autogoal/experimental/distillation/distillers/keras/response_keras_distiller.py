from tensorflow.keras import Model, losses, optimizers
from .model_distillers import ResponseDistiller
from .keras_distiller import _KerasDistiller


class ResponseKerasDistiller(_KerasDistiller):
    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> ResponseDistiller:
        distiller = ResponseDistiller(student=student_model, teacher=teacher_model)
        distiller.compile(
            optimizers.RMSprop(),
            ["accuracy"],
            losses.categorical_crossentropy,
            losses.categorical_crossentropy,
            alpha=self._distiller_alpha,
            temperature=self._distiller_temperature,
        )
        return distiller
