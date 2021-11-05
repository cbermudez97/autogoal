from .distiller_base import DistillerBase
from tensorflow.keras.activations import softmax


class ResponseDistiller(DistillerBase):
    def calculate_distillation_loss(self, x, y):
        teacher_predictions = self.teacher_no_act(x, training=False)
        student_predictions_no_act = self.student_no_act(x, training=True)
        distillation_loss = self.alpha * self.distillation_loss_fn(
            softmax(teacher_predictions / self.temperature, axis=1),
            softmax(student_predictions_no_act / self.temperature, axis=1),
        )
        return distillation_loss
