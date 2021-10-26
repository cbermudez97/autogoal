from typing import List
from autogoal.kb import Pipeline, AlgorithmBase
from autogoal.kb._algorithm import build_input_args
from autogoal.experimental.distillation.distillers import find_distillers
from autogoal.experimental.distillation.compressors import find_compressors
from autogoal.experimental.distillation.distillers.base_distiller import (
    AlgorithmDistillerBase,
)


class PipelineDistiller:
    def distill(
        self,
        pipeline: Pipeline,
        train_inputs,
        test_inputs,
        distillers_registry: List = None,
        compressors_registry: List = None,
        compression_ratio: float = 0.5,
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
                distiller: AlgorithmDistillerBase = distiller_cls(
                    compression_ratio=compression_ratio
                )
                if distiller.can_distill(algorithm):
                    distilled_algorithm = distiller.distill(
                        algorithm, train_args, test_args, registry=compressors_registry,
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
