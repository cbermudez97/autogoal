from .distiller_base import DistillerBase


class BasicClassifierNoDistiller(DistillerBase):
    def compile(self, optimizer, metrics, task_loss_fn, alpha=0, **kwargs):
        super().compile(optimizer, metrics, task_loss_fn, alpha=alpha, **kwargs)

    def calculate_distillation_loss(self, x, y):
        return 0
