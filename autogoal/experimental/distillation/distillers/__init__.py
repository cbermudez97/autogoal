def find_distillers(include=None, exclude=None, modules=None):
    import inspect
    import re

    from .base_distiller import AlgorithmDistillerBase

    result = []

    if include:
        include = f".*({include}).*"
    else:
        include = r".*"

    if exclude:
        exclude = f".*({exclude}).*"

    if modules is None:
        modules = []

        try:
            from autogoal.experimental.distillation.distillers import keras

            modules.append(keras)
        except ImportError as e:
            pass

        try:
            from autogoal.experimental.distillation.distillers import transformers

            modules.append(transformers)
        except ImportError as e:
            pass

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, AlgorithmDistillerBase):
                continue

            if not hasattr(cls, "can_distill") or not hasattr(cls, "distill"):
                continue

            if cls.__name__.startswith("_"):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue

            result.append(cls)

    return result
