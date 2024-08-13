import click  # noqa: CPY001, D100, EXE002, INP001
from src.quofemDTOs import Model
from src.runmodel.RunModelDTOs import RunModelDTO


@click.command()
@click.option(
    '--workflowInput',
    type=click.Path(exists=True, readable=True),
    required=True,
    help='Path to JSON file containing the details of FEM and UQ tools.',
)
@click.option(
    '--driverFile',
    type=click.Path(exists=True, readable=True),
    help='ASCII file containing the details on how to run the FEM application.',
)
@click.option(
    '--runType',
    type=click.Choice(['runningLocal', 'runningRemote']),
    default='runningLocal',
    help='Choose between local or cluster execution of workflow.',
)
@click.option(
    '--osType',
    type=click.Choice(['Linux', 'Windows']),
    help='Type of operating system the workflow will run on.',
)
def preprocess(workflowinput, driverfile, runtype, ostype):  # noqa: ARG001, D103
    # 1. Parse the input JSON file
    model = Model.parse_file(workflowinput)

    # 2. Generate code
    code = []
    code.append('import time\n')  # noqa: FURB113
    code.append('t1 = time.time()\n')

    # Create commands for defining distributions
    code.append('#\n# Creating the random variable distributions\n#')
    marginals_code = 'marginals = JointIndependent(['
    for distribution in model.randomVariables:
        (distribution_code, input) = distribution.init_to_text()  # noqa: A001
        code.append(distribution_code)
        marginals_code += input + ', '
    marginals_code += '])'
    code.append(marginals_code)  # noqa: FURB113
    code.append(f'numRV = {len(model.randomVariables)}\n')

    # Create files and commands for runmodel
    runmodel_code = RunModelDTO.create_runmodel_with_variables_driver(
        variables=model.randomVariables, driver_filename=driverfile
    )
    code.append('#\n# Creating the model\n#')  # noqa: FURB113
    code.append(runmodel_code)

    # Create commands for the UQ method
    (uqmethod_code, _) = model.UQ.methodData.generate_code()
    code.append('#\n# Defining and running the UQ analysis\n#')  # noqa: FURB113
    code.append(uqmethod_code)

    # 3. Write code to analysis script
    with open('UQpyAnalysis.py', 'w') as outfile:  # noqa: FURB103, PLW1514, PTH123
        outfile.write('\n'.join(code))


if __name__ == '__main__':
    preprocess()
