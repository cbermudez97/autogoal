from .distiller_base import DistillerBase


class BasicClassifierNoDistiller(DistillerBase):
    def compile(self, optimizer, metrics, task_loss_fn, **kwargs):
        super().compile(optimizer, metrics, task_loss_fn, **kwargs)

    def calculate_distillation_loss(self, x, y):
        return 0
