from typing import Literal, Union  # noqa: CPY001, D100, INP001

from pydantic import Field
from src.sampling.mcmc.StretchDto import SamplingMethod
from src.UQpyDTO import UQpyDTO
from typing_extensions import Annotated


class ReliabilityMethodBaseDTO(UQpyDTO):  # noqa: D101
    pass


class SubsetSimulationDTO(ReliabilityMethodBaseDTO):  # noqa: D101
    method: Literal['Subset Simulation'] = 'Subset Simulation'
    conditionalProbability: float  # noqa: N815
    failure_threshold: float = Field(..., alias='failureThreshold')
    maxLevels: int  # noqa: N815
    samples_per_subset: int
    samplingMethod: SamplingMethod  # noqa: N815

    # def __post_init__(self):
    # self.samplingMethod.n_chains=int(self.samples_per_subset*self.conditionalProbability)

    def init_to_text(self):  # noqa: ANN201, D102
        from UQpy.reliability.SubsetSimulation import (  # noqa: PLC0415
            SubsetSimulation,
        )
        from UQpy.sampling.MonteCarloSampling import (  # noqa: PLC0415
            MonteCarloSampling,
        )

        c = SubsetSimulation

        self.__create_postprocess_script()
        # output_script = Path('postprocess_script.py')

        initial_sampler = (
            'from '
            + MonteCarloSampling.__module__
            + ' import '
            + MonteCarloSampling.__module__.split('.')[-1]
            + '\n'
        )
        initial_sampler += f"monte_carlo = {MonteCarloSampling.__module__.split('.')[-1]}(distributions=marginals, nsamples={self.samples_per_subset}, random_state=sampling.random_state)\n"

        class_name = c.__module__.split('.')[-1]
        import_statement = 'from ' + c.__module__ + ' import ' + class_name

        input_str = 'subset'
        initializer = (
            f'{input_str} = {class_name}(sampling={self.samplingMethod}, '
            f'conditional_probability={self.conditionalProbability}, '
            f'max_level={self.maxLevels}, runmodel_object=run_model, '
            f'nsamples_per_subset={self.samples_per_subset}, '
            f'samples_init=monte_carlo.samples)\n'
        )

        results_script = '#\n# Creating the results\n#\n'
        results_script += (
            'samples_list = []\n'
            'for s in subset.samples:\n'
            '\tsamples_list.append(s.tolist())\n\n'
            'performance_function_list = []\n'
            'for p in subset.performance_function_per_level:\n'
            '\tperformance_function_list.append(p.tolist())\n\n'
        )
        results_script += (
            "output_data = {\n\t'failure_probability': subset.failure_probability, "
            "\n\t'time_to_completion_in_minutes': f'{(time.time() - t1)/60}', "
            "\n\t'number_of_model_evaluations': len(run_model.qoi_list), "
            "\n\t'num_levels': f'{len(subset.samples)}', "
            "\n\t'performance_threshold_per_level': subset.performance_threshold_per_level, "
            "\n\t'sample_values_per_level': samples_list, "
            "\n\t'performance_function_per_level': performance_function_list, "
            "\n\t'independent_chains_CoV': f'{subset.independent_chains_CoV}', "
            "\n\t'dependent_chains_CoV': f'{subset.dependent_chains_CoV}'"
            '\n}\n'
        )
        save_script = '#\n# Writing the UQ analysis results\n#\n'
        save_script += 'import json \n'
        save_script += (
            "with open('uqpy_results.json', 'w') as file:\n"
            '\tfile.write(json.dumps(output_data))\n'
        )

        prerequisite_str = '\n'.join(  # noqa: FLY002
            [
                initial_sampler,
                import_statement,
                initializer,
                results_script,
                save_script,
            ]
        )
        return prerequisite_str, input_str

    def __create_postprocess_script(self, results_filename: str = 'results.out'):  # noqa: ANN202
        postprocess_script_code = [
            'def compute_limit_state(index: int) -> float:',
            f"\twith open('{results_filename}', 'r') as f:",
            '\t\tres = f.read().strip()',
            '\tif res:',
            '\t\ttry:',
            '\t\t\tres = float(res)',
            '\t\texcept ValueError:',
            "\t\t\traise ValueError(f'Result should be a single float value, check results.out file for sample evaluation {index}')",
            '\t\texcept Exception:',
            '\t\t\traise',
            '\t\telse:',
            f'\t\t\treturn {self.failure_threshold} - res',
            '\telse:',
            "\t\traise ValueError(f'Result not found in results.out file for sample evaluation "  # noqa: ISC003
            + "{index}')",
        ]

        with open('postprocess_script.py', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
            f.write('\n'.join(postprocess_script_code))


class FormDTO(ReliabilityMethodBaseDTO):  # noqa: D101
    method: Literal['FORM'] = 'FORM'


ReliabilityMethod = Annotated[
    Union[SubsetSimulationDTO, FormDTO], Field(discriminator='method')
]
