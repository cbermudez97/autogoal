import tensorflow as tf

from .distiller_base import DistillerBase


class RelationDistiller(DistillerBase):
    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.8,
        psi="angle",
        **kwargs
    ):
        super().compile(
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=alpha,
            **kwargs
        )
        if not psi in ["distance", "angle"]:
            raise ValueError('Invalid param "psi".')
        self.psi = psi

    def calculate_distillation_loss(self, x, y):
        teacher_predictions = self.teacher_no_act(x, training=False)
        student_predictions_no_act = self.student_no_act(x, training=True)
        psi_fn = self.calcule_distance
        if self.psi == "angle":
            psi_fn = self.calcule_angle
        teacher_relations = psi_fn(teacher_predictions)
        student_relations = psi_fn(student_predictions_no_act)
        huber_loss = self.distillation_loss_fn(teacher_relations, student_relations)
        distillation_loss = self.alpha * huber_loss
        return distillation_loss

    def calcule_distance(self, y):
        distances = tf.map_fn(
            lambda yi: tf.map_fn(
                lambda yj: tf.norm(yi - yj), y, fn_output_signature=y.dtype
            ),
            y,
            fn_output_signature=y.dtype,
        )
        plain_distances = tf.reshape(distances, (-1,))
        mu = tf.reduce_mean(plain_distances)
        return plain_distances / mu

    def calcule_angle(self, y):
        cosin = lambda yi, yj, yk: tf.tensordot(
            (yi - yj) / tf.norm(yi - yj), (yk - yj) / tf.norm(yk - yj), 1
        )
        angles = tf.map_fn(
            lambda yi: tf.map_fn(
                lambda yj: tf.map_fn(
                    lambda yk: cosin(yi, yj, yk), y, fn_output_signature=y.dtype
                ),
                y,
                fn_output_signature=y.dtype,
            ),
            y,
            fn_output_signature=y.dtype,
        )
        plain_angles = tf.reshape(angles, (-1,))
        no_nan_angles = tf.where(
            tf.math.is_nan(plain_angles), tf.zeros_like(plain_angles), plain_angles,
        )
        return no_nan_angles
