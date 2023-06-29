from typing import List
from pathlib import Path

from src.quofemDTOs import RandomVariable


class RunModelDTO:
    @staticmethod
    def create_runmodel_with_variables_driver(variables: List[RandomVariable],
                                              driver_filename: str = 'driver'):
        RunModelDTO.__create_runmodel_input_teplate(variables)
        RunModelDTO.__create_model_script(driver_filename)
        RunModelDTO.__create_postprocess_script()

        #Validate file paths
        input_template = Path('params_template.in')
        model_script = Path('model_script.py')
        output_script = Path('postprocess_script.py')

        var_names = [f'{rv.name}' for rv in variables]
        run_model_code = [
            'from UQpy.run_model.RunModel import RunModel',
            'from UQpy.run_model.model_execution.ThirdPartyModel import ThirdPartyModel',
            f"third_party_model = ThirdPartyModel(var_names={var_names}, input_template='params_template.in', model_script='model_script.py', model_object_name='model', output_script='postprocess_script.py', output_object_name='compute_limit_state')",
            'run_model = RunModel(model=third_party_model)',
        ]

        return "\n".join(run_model_code)

    @staticmethod
    def __create_runmodel_input_teplate(variables: List[RandomVariable]):
        template_code = [f"{len(variables)}"]
        for rv in variables:
            template_code.append(f"{rv.name} <{rv.name}>")

        with open("params_template.in", "w") as f:
            f.write("\n".join(template_code))

    @staticmethod
    def __create_model_script(driver_filename):
        template_filepath = Path("params_template.in")
        template_file_base = template_filepath.stem
        template_file_suffix = template_filepath.suffix
        model_script_code = [
            'import subprocess',
            'import fire\n',
            'def model(sample_index: int) -> None:',
            f"\tcommand1 = f'mv ./InputFiles/{template_file_base}_"
            + "{sample_index}"
            + f"{template_file_suffix} ./params.in'",
            f"\tcommand2 = './{driver_filename}'\n",
            '\tsubprocess.run(command1, stderr=subprocess.STDOUT, shell=True)',
            '\tsubprocess.run(command2, stderr=subprocess.STDOUT, shell=True)\n',
            "if __name__ == '__main__':",
            '\tfire.Fire(model)',
        ]

        with open('model_script.py', "w") as f:
            f.write("\n".join(model_script_code))

    @staticmethod
    def __create_postprocess_script(results_filename: str = 'results.out'):
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
            "\t\t\treturn res",
            '\telse:',
            "\t\traise ValueError(f'Result not found in results.out file for sample evaluation "
            + "{index}')",
        ]

        with open("postprocess_script.py", "w") as f:
            f.write("\n".join(postprocess_script_code))
