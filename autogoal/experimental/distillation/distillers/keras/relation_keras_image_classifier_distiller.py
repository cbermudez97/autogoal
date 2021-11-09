from tensorflow.keras import Model, losses, optimizers
from autogoal.experimental.distillation.distillers.keras.model_distillers import (
    RelationDistiller,
)
from autogoal.experimental.distillation.distillers.keras.keras_image_classifier_distiller import (
    _KerasImageClassifierDistiller,
)


class RelationKerasImageClassifierDistiller(_KerasImageClassifierDistiller):
    def __init__(
        self,
        epochs: int = 10,
        early_stop: int = 3,
        distiller_alpha: float = 0.9,
        distiller_temperature: float = 1,
        distiller_psi: str = "angle",
        delta: float = 1,
        batch_size: int = 8,
    ):
        super().__init__(
            epochs=epochs,
            early_stop=early_stop,
            distiller_alpha=distiller_alpha,
            distiller_temperature=distiller_temperature,
            batch_size=batch_size,
        )
        self.delta = delta
        self._distiller_psi = distiller_psi

    def build_distiller(
        self, student_model: Model, teacher_model: Model
    ) -> RelationDistiller:
        distiller = RelationDistiller(student=student_model, teacher=teacher_model)
        distiller.compile(
            teacher_model.optimizer,
            ["accuracy"],
            losses.categorical_crossentropy,
            losses.Huber(delta=self.delta),
            alpha=self._distiller_alpha,
        )
        return distiller
