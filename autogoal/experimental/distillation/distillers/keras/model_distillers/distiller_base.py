import abc
from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Activation

loss_metric = metrics.Mean(name="loss")
task_loss_metric = metrics.Mean(name="task_loss")
distillation_loss_metric = metrics.Mean(name="distillation_loss")


class DistillerBase(Model, abc.ABC):
    def __init__(self, student: Model, teacher: Model):
        super(DistillerBase, self).__init__()
        self.student, self.student_no_act = self.build_models(student)
        self.teacher, self.teacher_no_act = self.build_models(teacher)

    def __call__(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    def compile(
        self,
        optimizer,
        metrics,
        task_loss_fn,
        distillation_loss_fn,
        alpha=0.9,
        **kwargs,
    ):
        super(DistillerBase, self).compile(
            optimizer=optimizer, metrics=metrics, **kwargs
        )
        self.task_loss_fn = task_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha

    def train_step(self, data):
        x, y = data
        with GradientTape() as tape:
            student_predictions = self.student(x, training=True)
            student_loss = self.task_loss_fn(y, student_predictions)
            distillation_loss = self.calculate_distillation_loss(x, y)
            loss = student_loss + distillation_loss
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, student_predictions)
        loss_metric.update_state(loss)
        task_loss_metric.update_state(student_loss)
        distillation_loss_metric.update_state(distillation_loss)
        results = {m.name: m.result() for m in self.metrics}
        return results

    @abc.abstractmethod
    def calculate_distillation_loss(self, x, y):
        pass

    @property
    def metrics(self):
        return self.compiled_metrics.metrics + [
            loss_metric,
            task_loss_metric,
            distillation_loss_metric,
        ]

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.task_loss_fn(y, y_prediction)
        self.compiled_metrics.update_state(y, y_prediction)
        loss_metric.update_state(student_loss)
        task_loss_metric.update_state(student_loss)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def split_outputs_activations(self, model):
        def f(layer):
            config = layer.get_config()
            if not isinstance(layer, Activation) and layer.name in model.output_names:
                config.pop("activation", None)
            copy = layer.__class__.from_config(config)
            return copy

        copy = clone_model(model, clone_function=f)
        copy.build(model.input_shape)
        copy.set_weights(model.get_weights())
        old_outputs = [model.get_layer(name=name) for name in copy.output_names]
        new_outputs = [
            Activation(old_output.activation)(output)
            if old_output.activation
            else output
            for output, old_output in zip(copy.outputs, old_outputs)
        ]
        copy = Model(copy.inputs, new_outputs)
        return copy

    def build_models(self, model):
        copy = self.split_outputs_activations(model)
        no_act_outputs = [
            copy.get_layer(name=name).output for name in model.output_names
        ]
        return copy, Model(copy.inputs, no_act_outputs)
