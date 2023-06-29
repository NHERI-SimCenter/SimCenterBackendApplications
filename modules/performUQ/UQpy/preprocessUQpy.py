import click
import os

from src.quofemDTOs import Model
from src.runmodel.RunModelDTOs import RunModelDTO


@click.command()
@click.option('--workflowInput', type=click.Path(exists=True, readable=True), required=True,
              help="Path to JSON file containing the details of FEM and UQ tools.")
@click.option('--driverFile', type=click.Path(exists=True, readable=True),
              help="ASCII file containing the details on how to run the FEM application.")
@click.option('--runType', type=click.Choice(['runningLocal', 'runningRemote']),
              default='runningLocal', help="Choose between local or cluster execution of workflow.")
@click.option('--osType', type=click.Choice(['Linux', 'Windows']),
              help="Type of operating system the workflow will run on.")
def preprocess(workflowinput, driverfile, runtype, ostype):
    code = []

    with open(os.path.join(os.getcwd(), workflowinput)) as my_file:
        json = my_file.read()
        model = Model.parse_raw(json)

    # inputFile = args.inputFile   # JSON FILE information from GUI
    # workflow_driver = args.workflow_driver
    # runType = args.runType
    # osType = args.osType
    
    # # Step 1: Read the JSON file
    # with open(inputFile, "r") as f:
    #     data = json.load(f)
    
    # uq_data = data["UQ"]
    # rv_data = data["randomVariables"]
    # fem_data = data["Applications"]["FEM"]
    # edp_data = data["EDP"]
    
    # cwd = os.getcwd()
    # templateDir = cwd
    # with open(os.path.join(templateDir, "UQpyAnalysis.py"), "w") as f:
    #     pass
    
    # # f.write("from UQpy import PythonModel\n")
    # # f.write("from UQpy.distributions import *\n")
    # # f.write("from UQpy.reliability import SubsetSimulation\n")
    # # f.write("from UQpy.run_model.RunModel import RunModel\n")
    # # f.write("from UQpy.sampling import ModifiedMetropolisHastings, Stretch, MetropolisHastings\n")
    # # f.write("distributionRV = []\n")
    # # f.write("num_of_rves = len(RVdata)\n")
    # # f.write("for i in range(num_rves):\n")
    # #     f.write("nameRV.append(RVdata[i]["name"])\n")
    # #     f.write("if RVdata[i]["distribution"] == 'Normal':\n")
    # #         f.write("distributionRV.append(Normal(loc={0}, scale={1}))".format(rv_data[i]["scaleparam"], rv_data[i]["shapeparam"]))
    # #         #distributionRV.append(Normal(loc=RVdata[i]["scaleparam"], scale=RVdata[i]["shapeparam"])) # check variance 

    marginals_code = 'marginals = JointIndependent(['
    for distribution in model.randomVariables:
        (distribution_code, input) = distribution.init_to_text()
        marginals_code += input + ','
        code.append(distribution_code)
    marginals_code += '])\n'
    code.append(marginals_code)

    runmodel_code = RunModelDTO.create_runmodel_with_variables_driver(variables=model.randomVariables,
                                                                      driver_filename=driverfile)
    (uqmethod_code, _) = model.UQ.methodData.generate_code()

    code.append(runmodel_code)
    code.append(uqmethod_code)

    with open("UQpyAnalysis.py", 'w') as outfile:
        outfile.write("\n".join(code))


if __name__ == "__main__":
    preprocess()
