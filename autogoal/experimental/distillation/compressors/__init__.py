def find_compressors(include=None, exclude=None, modules=None):
    import inspect
    import re

    from .base_compressor import ModelCompressorBase

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
            from autogoal.experimental.distillation.compressors import keras

            modules.append(keras)
        except ImportError as e:
            pass

    for module in modules:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if not issubclass(cls, ModelCompressorBase):
                continue

            if not hasattr(cls, "can_compress") or not hasattr(cls, "compress"):
                continue

            if cls.__name__.startswith("_"):
                continue

            if not re.match(include, repr(cls)):
                continue

            if exclude is not None and re.match(exclude, repr(cls)):
                continue

            result.append(cls)

    return result
