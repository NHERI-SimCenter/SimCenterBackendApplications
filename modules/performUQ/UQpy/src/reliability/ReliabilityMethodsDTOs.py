from pathlib import Path

from pydantic import BaseModel, Field
from typing import Literal, Union
from typing_extensions import Annotated

from src.UQpyDTO import UQpyDTO
from src.sampling.mcmc.StretchDto import StretchDto, SamplingMethod


class ReliabilityMethodBaseDTO(UQpyDTO):
    pass


class SubsetSimulationDTO(ReliabilityMethodBaseDTO):
    method: Literal['Subset Simulation'] = 'Subset Simulation'
    conditionalProbability: float
    failure_threshold: float = Field(..., alias="failureThreshold")
    maxLevels: int
    samples_per_subset: int
    samplingMethod: SamplingMethod

    def __post_init__(self):
        self.samplingMethod.n_chains=int(self.samples_per_subset*self.conditionalProbability)

    def init_to_text(self):
        from UQpy.reliability.SubsetSimulation import SubsetSimulation
        from UQpy.sampling.MonteCarloSampling import MonteCarloSampling
        c = SubsetSimulation

        self.__create_postprocess_script()
        output_script = Path('postprocess_script.py')

        class_name = c.__module__.split(".")[-1]
        import_statement = "from " + c.__module__ + " import " + class_name + "\n"

        import_statement += "from " + MonteCarloSampling.__module__ + " import " + \
                            MonteCarloSampling.__module__.split(".")[-1] + "\n"

        import_statement += f"monte_carlo = {MonteCarloSampling.__module__.split('.')[-1]}(distributions=marginals, nsamples={self.samples_per_subset})\n"
        input_str = "subset"
        initializer = f'{input_str} = {class_name}(sampling={self.samplingMethod}, ' \
                      f'conditional_probability={self.conditionalProbability}, ' \
                      f'max_level={self.maxLevels}, runmodel_object=run_model,' \
                      f'nsamples_per_subset={self.samples_per_subset},'\
                      f'samples_init=monte_carlo.samples)\n'
        
        import_statement+="import json \n"
        save_script = "output_data = {'failure_probability' : subset.failure_probability,"\
                                      "'performance_threshold_per_level':subset.performance_threshold_per_level,"\
                                      "'independent_chains_CoV':subset.independent_chains_CoV,"\
                                      "'dependent_chains_CoV': subset.dependent_chains_CoV}\n"
        save_script+="with open('uqpy_results.json', 'w') as file:\n"\
                     "\tfile.write(json.dumps(output_data))\n"

        prerequisite_str = import_statement + initializer + save_script
        return prerequisite_str, input_str

    def __create_postprocess_script(self, results_filename: str = 'results.out'):
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
            f"\t\t\treturn {self.failure_threshold} - res",
            '\telse:',
            "\t\traise ValueError(f'Result not found in results.out file for sample evaluation "
            + "{index}')",
        ]

        with open("postprocess_script.py", "w") as f:
            f.write("\n".join(postprocess_script_code))


class FormDTO(ReliabilityMethodBaseDTO):
    method: Literal['FORM'] = 'FORM'


ReliabilityMethod = Annotated[Union[SubsetSimulationDTO, FormDTO], Field(discriminator='method')]
