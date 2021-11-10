import tensorflow as tf

from .distiller_base import DistillerBase


@tf.custom_gradient
def norm(x):
    y = tf.norm(x)

    def grad(dy):
        return dy * (x / (y + 1e-19))

    return y, grad


class RelationDistiller(DistillerBase):
    def compile(
        self,
        optimizer,
        metrics,
        task_loss_fn,
        distillation_loss_fn,
        alpha=0.9,
        psi="angle",
        **kwargs
    ):
        super().compile(
            optimizer,
            metrics,
            task_loss_fn,
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
        item_shape = y.shape[1:]
        N = tf.cast(tf.size(y) / tf.reduce_sum(item_shape), tf.int32)
        y2 = tf.reshape(tf.repeat(y, N, axis=0), (N, -1, *item_shape))
        y2t = tf.transpose(y2, perm=[1, 0, 2])
        plain_y2 = tf.reshape(y2, (-1, *item_shape))
        plain_y2t = tf.reshape(y2t, (-1, *item_shape))
        distances = tf.map_fn(norm, plain_y2 - plain_y2t)
        mu = tf.reduce_mean(distances)
        distances = distances / mu
        return distances

    def calcule_angle(self, y):
        item_shape = y.shape[1:]
        N = tf.cast(tf.size(y) / tf.reduce_sum(item_shape), tf.int32)
        tN1 = tf.stack([(N), tf.constant(1)], 0)
        tN2 = tf.stack([N ** 2, tf.constant(1)], 0)
        yi = tf.repeat(y, N ** 2, 0)
        yj_part = tf.repeat(y, N, 0)
        yj = tf.tile(yj_part, tN1,)
        yk = tf.tile(y, tN2)
        difij = yi - yj
        ndifij = tf.map_fn(norm, difij) + 1e-19
        eij = difij / tf.expand_dims(ndifij, 1)
        difkj = yk - yj
        ndifkj = tf.map_fn(norm, difkj) + 1e-19
        ekj = difkj / tf.expand_dims(ndifkj, 1)
        angles = tf.reduce_sum(eij * ekj, 1)
        return angles
