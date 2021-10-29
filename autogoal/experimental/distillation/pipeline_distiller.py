from typing import Any, Dict, List, Type
from autogoal.kb import Pipeline, AlgorithmBase
from autogoal.kb._algorithm import build_input_args
from autogoal.experimental.distillation.distillers import find_distillers
from autogoal.experimental.distillation.compressors import find_compressors
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)
from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)


class PipelineDistiller:
    def distill(
        self,
        pipeline: Pipeline,
        train_inputs,
        test_inputs,
        distillers_registry: List[Type[AlgorithmDistillerBase]] = None,
        compressors_registry: List[Type[ModelCompressorBase]] = None,
        distillers_kwargs: Dict[Type[AlgorithmDistillerBase], Dict[str, Any]] = {},
        compressors_kwargs: Dict[Type[ModelCompressorBase], Dict[str, Any]] = {},
    ) -> Pipeline:
        # Build train data
        train_data = {}

        for i, t in zip(train_inputs, pipeline.input_types):
            train_data[t] = i

        # Build test data
        test_data = {}

        for i, t in zip(test_inputs, pipeline.input_types):
            test_data[t] = i

        # Build registry
        if not distillers_registry:
            distillers_registry = find_distillers()

        if not compressors_registry:
            compressors_registry = find_compressors()

        destilled_pipeline = Pipeline([], input_types=pipeline.input_types)

        # Destill algorithms
        for algorithm in pipeline.algorithms:
            distilled_algorithm: AlgorithmBase = None

            # Build algorithm train and test args
            train_args = build_input_args(algorithm, train_data)
            test_args = build_input_args(algorithm, test_data)

            for distiller_cls in distillers_registry:
                # Try distilling algorithm
                distiller_kwargs = distillers_kwargs.get(distiller_cls, {})
                distiller: AlgorithmDistillerBase = distiller_cls(**distiller_kwargs,)
                if distiller.can_distill(algorithm):
                    distilled_algorithm = distiller.distill(
                        algorithm,
                        train_args,
                        test_args,
                        registry=compressors_registry,
                        compressors_kwargs=compressors_kwargs,
                    )
                    break

            # Add distilled algorithm to distilled pipeline
            if not distilled_algorithm:
                distilled_algorithm = algorithm
            destilled_pipeline.algorithms.append(distilled_algorithm)

            # Update train and test data
            train_output = distilled_algorithm.run(**train_args)
            test_output = distilled_algorithm.run(**test_args)
            output_type = distilled_algorithm.output_type()
            train_data[output_type] = train_output
            test_data[output_type] = test_output

        return destilled_pipeline
