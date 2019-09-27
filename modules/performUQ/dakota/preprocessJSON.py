# import functions for Python 2.X support
from __future__ import division, print_function
import sys
if sys.version.startswith('2'): 
    range=xrange

#else:
#    from past.builtins import basestring

import json
import os
import sys
import platform
import posixpath

numRandomVariables = 0

numNormalUncertain = 0
normalUncertainName=[]
normalUncertainMean =[]
normalUncertainStdDev =[]

numLognormalUncertain = 0;
lognormalUncertainName=[];
lognormalUncertainMean =[];
lognormalUncertainStdDev =[];

numUniformUncertain = 0;
uniformUncertainName=[];
uniformUncertainLower =[];
uniformUncertainUpper =[];

numContinuousDesign = 0;
continuousDesignName=[];
continuousDesignLower =[];
continuousDesignUpper =[];
continuousDesignInitialPoint =[];

numConstantState = 0;
constantStateName=[];
constantStateValue =[];

numWeibullUncertain = 0;
weibullUncertainName=[];
weibullUncertainAlphas =[];
weibullUncertainBetas =[];

numGammaUncertain = 0;
gammaUncertainName=[];
gammaUncertainAlphas =[];
gammaUncertainBetas =[];

numGumbellUncertain = 0;
gumbellUncertainName=[];
gumbellUncertainAlphas =[];
gumbellUncertainBetas =[];

numBetaUncertain = 0;
betaUncertainName=[];
betaUncertainLower =[];
betaUncertainHigher =[];
betaUncertainAlphas =[];

numDiscreteDesignSetString = 0
discreteDesignSetStringName=[]
discreteDesignSetStringValues =[]

numResultFiles = 0
outputResultFiles = []

def preProcessDakota(bimName, evtName, samName, edpName, simName, driverFile, uqData):

    global numRandomVariables
    global numNormalUncertain
    global normalUncertainName
    global normalUncertainMean
    global normalUncertainStdDev

    global numDiscreteDesignSetString
    global discreteDesignSetStringName
    global discreteDesignSetStringValues

    global outputResultFiles
    global numResultFiles

    #
    # get UQ method data
    #

    with open(bimName) as data_file:
    #with open('dakota.json') as data_file:    
        data = json.load(data_file)
        
    #uqData = data["UQ_Method"];
    #samplingData = uqData["samplingMethodData"];
    samplingData = uqData
    method = samplingData["method"];

    #if (method == "Monte Carlo"):
    #    method = 'random'
    #else:
    #    method = 'lhs'
    #numSamples=samplingData["samples"];
    #seed = samplingData["seed"];

    #
    # get result files
    #

    if 'ResultFiles' in data:
        numResultFiles = 1;
        outputResultFiles.append("EVENT.json");
    else:
        numResultFiles = 0;

    # 
    # parse the data
    #

    bimExists = parseFileForRV(bimName)
    evtExists = parseFileForRV(evtName)
    samExists = parseFileForRV(samName)
    simExists = parseFileForRV(simName)
    edpExists = parseFileForRV(edpName)

    # Add a dummy random variable if no other RV was defined
    if numRandomVariables == 0:
        add_dummy()

    #Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'
    remoteWorkflowDriverName = 'workflow_driver'
    if platform.system() == 'Windows':
        workflowDriverName = 'workflow_driver.bat'

    #
    # Write the input file: dakota.in 
    #

    # write out the method data
    f = open('dakota.in', 'w')

    # write out the env data
    dakota_input = ""
    
    dakota_input += (
    """environment
    tabular_data
    tabular_data_file = 'dakotaTab.out'
    
    method,
    """)
    
    if method == "Importance Sampling":
        numSamples=samplingData["samples"]
        seed = samplingData["seed"]
        imp_sams_arg = samplingData["ismethod"]

        dakota_input += (
    """importance_sampling
    {ismethod}
    samples = {samples}
    seed = {seed}
    
    """.format(
        ismethod = imp_sams_arg,
        samples = numSamples,
        seed = seed))
    
    elif method == "Monte Carlo":
        numSamples=samplingData["samples"]
        seed = samplingData["seed"]

        dakota_input += (
    """sampling
    sample_type = {sample_type}
    samples = {samples}
    seed = {seed}
    
    """.format(
        sample_type = 'random',
        samples = numSamples,
        seed = seed))
    
    elif method == "LHS":
        numSamples=samplingData["samples"]
        seed = samplingData["seed"]        

        dakota_input += (
    """sampling
    sample_type = {sample_type}
    samples = {samples}
    seed = {seed}
    
    """.format(
        sample_type = 'lhs' ,
        samples = numSamples,
        seed = seed))
    
    elif method == "Gaussian Process Regression":
        train_samples = samplingData["samples"]
        gpr_seed = samplingData["seed"]
        train_method = samplingData["dataMethod"]
        
        train_samples2 = samplingData["samples2"]
        gpr_seed2 = samplingData["seed2"]
        train_method2 = samplingData["dataMethod2"]
        
        # write out the env data
        dakota_input = ""
        
        dakota_input += (
        """environment
method_pointer = 'EvalSurrogate'
tabular_data
tabular_data_file = 'dakotaTab.out'
custom_annotated header eval_id
        
method
id_method = 'EvalSurrogate'
model_pointer = 'SurrogateModel'
        
sampling
samples = {no_surr_sams}
seed = {surr_seed}
sample_type {surr_sams_type}
        
model
id_model = 'SurrogateModel'
surrogate global
dace_method_pointer = 'DesignMethod'
gaussian_process surfpack
export_model
filename_prefix = 'dak_gp_model'
formats
text_archive
        
""").format(
        no_surr_sams = train_samples2,
        surr_seed = gpr_seed2,
        surr_sams_type = train_method2)

    # write out the variable data
    dakota_input += ('variables,\n')
    dakota_input += ('active uncertain \n')

    if (numNormalUncertain > 0):
        dakota_input += ('normal_uncertain = ' '{}'.format(numNormalUncertain))
        dakota_input += ('\n')
        dakota_input += ('means = ')
        for i in range(numNormalUncertain):
            dakota_input += ('{}'.format(normalUncertainMean[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('std_deviations = ')
        for i in range(numNormalUncertain):
            dakota_input += ('{}'.format(normalUncertainStdDev[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numNormalUncertain):
            dakota_input += ('\'')
            dakota_input += (normalUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numLognormalUncertain > 0):
        dakota_input += ('lognormal_uncertain = ' '{}'.format(numLognormalUncertain))
        dakota_input += ('\n')
        dakota_input += ('means = ')
        for i in range(numLognormalUncertain):
            dakota_input += ('{}'.format(lognormalUncertainMean[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('std_deviations = ')
        for i in range(numLognormalUncertain):
            dakota_input += ('{}'.format(lognormalUncertainStdDev[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
            
        dakota_input += ('descriptors = ')    
        for i in range(numLognormalUncertain):
            dakota_input += ('\'')
            dakota_input += (lognormalUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numUniformUncertain > 0):
        dakota_input += ('uniform_uncertain = ' '{}'.format(numUniformUncertain))
        dakota_input += ('\n')
        dakota_input += ('lower_bounds = ')
        for i in range(numUniformUncertain):
            dakota_input += ('{}'.format(uniformUncertainLower[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('upper_bounds = ')
        for i in range(numUniformUncertain):
            dakota_input += ('{}'.format(uniformUncertainUpper[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numUniformUncertain):
            dakota_input += ('\'')
            dakota_input += (uniformUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')


    if (numContinuousDesign > 0):
        dakota_input += ('continuous_design = ' '{}'.format(numContinuousDesign))
        dakota_input += ('\n')

        dakota_input += ('initial_point = ')
        for i in range(numContinuousDesign):
            dakota_input += ('{}'.format(continuousDesignInitialPoint[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('lower_bounds = ')
        for i in range(numContinuousDesign):
            dakota_input += ('{}'.format(continuousDesignLower[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('upper_bounds = ')
        for i in range(numContinuousDesign):
            dakota_input += ('{}'.format(continuousDesignUpper[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
        
        dakota_input += ('descriptors = ')    
        for i in range(numContinuousDesign):
            dakota_input += ('\'')
            dakota_input += (continuousDesignName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')
            

    numCState = 0
    if (numCState > 0):
        dakota_input += ('discrete_state_range = ' '{}'.format(numConstantState))
        dakota_input += ('\n')
        
        dakota_input += ('initial_state = ')
        for i in range(numConstantState):
            dakota_input += ('{}'.format(constantStateValue[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('descriptors = ')    
        for i in range(numConstantState):
            dakota_input += ('\'')
            dakota_input += (constantStateName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numConstantState > 0):
        dakota_input += ('discrete_design_set\nreal = ' '{}'.format(numConstantState))
        dakota_input += ('\n')
        
        dakota_input += ('num_set_values = ')
        for i in range(numConstantState):
            dakota_input += ('{}'.format(1))
            dakota_input += (' ')
        dakota_input += ('\n')
        
        dakota_input += ('set_values = ')
        for i in range(numConstantState):
            dakota_input += ('{}'.format(constantStateValue[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('descriptors = ')    
        for i in range(numConstantState):
            dakota_input += ('\'')
            dakota_input += (constantStateName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numBetaUncertain > 0):
        dakota_input += ('beta_uncertain = ' '{}'.format(numBetaUncertain))
        dakota_input += ('\n')
        dakota_input += ('alphas = ')
        for i in range(numBetaUncertain):
            dakota_input += ('{}'.format(betaUncertainAlphas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('betas = ')
        for i in range(numBetaUncertain):
            dakota_input += ('{}'.format(betaUncertainBetas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
        
        dakota_input += ('lower_bounds = ')
        for i in range(numBetaUncertain):
            dakota_input += ('{}'.format(betaUncertainLower[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('upper_bounds = ')
        for i in range(numBetaUncertain):
            dakota_input += ('{}'.format(betaUncertainHigher[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numBetaUncertain):
            dakota_input += ('\'')
            dakota_input += (betaUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numGammaUncertain > 0):
        dakota_input += ('gamma_uncertain = ' '{}'.format(numGammaUncertain))
        dakota_input += ('\n')
        dakota_input += ('alphas = ')
        for i in range(numGammaUncertain):
            dakota_input += ('{}'.format(gammaUncertainAlphas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('betas = ')
        for i in range(numGammaUncertain):
            dakota_input += ('{}'.format(gammaUncertainBetas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numGammaUncertain):
            dakota_input += ('\'')
            dakota_input += (gammaUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numGumbellUncertain > 0):
        dakota_input += ('gamma_uncertain = ' '{}'.format(numGumbellUncertain))
        dakota_input += ('\n')
        dakota_input += ('alphas = ')
        for i in range(numGumbellUncertain):
            dakota_input += ('{}'.format(gumbellUncertainAlphas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')

        dakota_input += ('betas = ')
        for i in range(numGumbellUncertain):
            dakota_input += ('{}'.format(gumbellUncertainBetas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numGumbellUncertain):
            dakota_input += ('\'')
            dakota_input += (gumbellUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    if (numWeibullUncertain > 0):
        dakota_input += ('weibull_uncertain = ' '{}'.format(numWeibullUncertain))
        dakota_input += ('\n')
        dakota_input += ('alphas = ')
        for i in range(numWeibullUncertain):
            dakota_input += ('{}'.format(weibullUncertainAlphas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
            
        dakota_input += ('betas = ')
        for i in range(numWeibullUncertain):
            dakota_input += ('{}'.format(weibullUncertainBetas[i]))
            dakota_input += (' ')
        dakota_input += ('\n')
    
        dakota_input += ('descriptors = ')    
        for i in range(numWeibullUncertain):
            dakota_input += ('\'')
            dakota_input += (weibullUncertainName[i])
            dakota_input += ('\' ')
        dakota_input += ('\n')

    dakota_input += ('\n')

    if (numDiscreteDesignSetString > 0):
        dakota_input += 'discrete_uncertain_set\n'
        dakota_input += 'string ' '{}'.format(numDiscreteDesignSetString)
        dakota_input += '\n'

        dakota_input += 'num_set_values = '  
        for i in range(numDiscreteDesignSetString):
            numElements = len(discreteDesignSetStringValues[i])
            dakota_input += ' ' '{}'.format(numElements)
            print(discreteDesignSetStringValues[i])
            print(numElements)

        dakota_input += '\n'
        dakota_input += 'set_values  '   
        for i in range(numDiscreteDesignSetString):
            elements = discreteDesignSetStringValues[i]
            for j in elements:
                dakota_input += '\'' '{}'.format(j)
                dakota_input += '\' '
            dakota_input += '\n'

        dakota_input += 'descriptors = '   
        for i in range(numDiscreteDesignSetString):
            dakota_input += '\''
            dakota_input += discreteDesignSetStringName[i]
            dakota_input += '\' '

        dakota_input += '\n'

    dakota_input += ('\n\n')

    if method == "Gaussian Process Regression":
        
        train_samples = samplingData["samples"]
        gpr_seed = samplingData["seed"]
        train_method = samplingData["dataMethod"]

        dakota_input += (
        """method
id_method = 'DesignMethod'
model_pointer = 'SimulationModel'
sampling
seed = {setseed}
sample_type {settype}
samples = {setsamples}
        
model
id_model = 'SimulationModel'
single
interface_pointer = 'SimulationInterface'
        
""").format(
    setseed = gpr_seed,
    settype = train_method,
    setsamples = train_samples
    )


    # write out the interface data
    dakota_input += ('interface,\n')

    runType = data.get("runType", "local");
    remoteDir = data.get("remoteAppDir", None);
    localDir = data.get("localAppDir", None);

    if method == "Gaussian Process Regression":
        dakota_input += ('id_interface = \'SimulationInterface\',\n')

    if (runType == "local"):
        uqData['concurrency'] = 2
    
    if uqData['concurrency'] == None:
        dakota_input += "fork asynchronous\n"
    elif uqData['concurrency'] > 1:
        dakota_input += "fork asynchronous evaluation_concurrency = {}\n".format(uqData['concurrency'])
    
    if runType == "local":    
        dakota_input += "analysis_driver = '{}'\n".format(workflowDriverName)
    else:
        dakota_input += "analysis_driver = '{}'\n".format(remoteWorkflowDriverName)


    # dakota_input += ('\nanalysis_driver = \'python analysis_driver.py\' \n')
    dakota_input += ('parameters_file = \'params.in\' \n')
    dakota_input += ('results_file = \'results.out\' \n')
    dakota_input += ('work_directory directory_tag directory_save\n')
    dakota_input += ('copy_files = \'templatedir/*\' \n')
    # dakota_input += ('named \'workdir\' file_save  directory_save \n')
    dakota_input += ('named \'workdir\' \n')
    dakota_input += ('aprepro \n')
    dakota_input += ('\n')


    f.write(dakota_input)
    
    #
    # write out the responses
    #

    with open(edpName) as data_file:    
        data = json.load(data_file)

    numResponses=data["total_number_edp"]



    f.write('responses, \n')
    f.write('response_functions = ' '{}'.format(numResponses))
    f.write('\n')
    f.write('response_descriptors = ')    

    
    for event in data["EngineeringDemandParameters"]:
        eventIndex = data["EngineeringDemandParameters"].index(event)
        for edp in event["responses"]:
            known = False
            if(edp["type"] == "max_abs_acceleration"):
                edpAcronym = "PFA"
                floor = edp["floor"]
                known = True

            elif(edp["type"] == "max_drift"):
                edpAcronym = "PID"
                floor = edp["floor2"]
                known = True

            elif(edp["type"] == "max_pressure"):
                edpAcronym = "PSP"
                floor = edp["floor2"]
                known = True

            elif(edp["type"] == "max_rel_disp"):
                edpAcronym = "PFD"
                floor = edp["floor"]
                known = True
            elif(edp["type"] == "peak_wind_gust_speed"):
                edpAcronym = "PWS"
                floor = edp["floor"]
                known = True

            else:
                f.write("'{}' ".format(edp["type"]))

            if (known == True):
                for dof in edp["dofs"]:
                    f.write("'{}-{}-{}-{}' ".format(eventIndex + 1, edpAcronym, floor, dof))

    f.write('\n')
    f.write('no_gradients\n')
    f.write('no_hessians\n\n')
    f.close()  # you can omit in most cases as the destructor will call it

    #
    # Write the workflow driver
    #

    if platform.system() == 'Darwin':
        f = open(workflowDriverName, 'w')
    else:
        f = open(workflowDriverName, 'w', newline='\n')

    # want to dprepro the files with the random variables
    if bimExists == True: f.write('perl dpreproSimCenter params.in bim.j ' + bimName + '\n')
    if samExists == True: f.write('perl dpreproSimCenter params.in sam.j ' + samName + '\n')
    if evtExists == True: f.write('perl dpreproSimCenter params.in evt.j ' + evtName + '\n')
    if edpExists == True: f.write('perl dpreproSimCenter params.in edp.j ' + edpName + '\n')

    scriptDir = os.path.dirname(os.path.realpath(__file__))

    with open(driverFile) as fp:
        for line in fp:
            #print(line)
            #print(localDir)
            if remoteDir is not None:
                line = line.replace(localDir,remoteDir)
            f.write(line)
            print(line)

    f.write('#comment to fix a bug\n')
    files = " "
    files =  files + "".join([str(i) for i in outputResultFiles])
    numR = str(numResultFiles)

    if (runType == "local"):
        f.write('"{}'.format(scriptDir) + '/extractEDP" ' + edpName + ' results.out ' + bimName + ' ' + numR + ' ' + files + '\n')

    elif remoteDir is not None:
        extractEDPCommand = posixpath.join(remoteDir, 'applications/performUQ/dakota/extractEDP')
        f.write(extractEDPCommand + ' ' + edpName + ' results.out ' + bimName + ' ' + numR + ' ' + files + '\n')
    else:
        f.write('\n')
        f.write('"'+os.path.join(scriptDir,'extractEDP')+'" ' + edpName + ' results.out \n')

    # Run 
    #f.write('rm -f *.com *.done *.dat *.log *.sta *.msg')
    #f.write('echo 1 >> results.out\n')
    f.close()

    return numRandomVariables

def parseFileForRV(fileName):
    global numRandomVariables

    global numNormalUncertain
    global normalUncertainName
    global normalUncertainMean
    global normalUncertainStdDev

    global numLognormalUncertain
    global lognormalUncertainName
    global lognormalUncertainMean
    global lognormalUncertainStdDev

    global numUniformUncertain
    global uniformUncertainName
    global uniformUncertainLower
    global uniformUncertainUpper

    global numContinuousDesign
    global continuousDesignName
    global continuousDesignLower
    global continuousDesignUpper
    global continuousDesignInitialPoint

    global numConstantState
    global constantStateName
    global constantStateValue

    global numWeibullUncertain
    global weibullUncertainName
    global weibullUncertainAlphas
    global weibullUncertainBetas

    global numGammaUncertain
    global gammaUncertainName
    global gammaUncertainAlphas
    global gammaUncertainBetas

    global numGumbellUncertain
    global gumbellUncertainName
    global gumbellUncertainAlphas
    global gumbellUncertainBetas

    global numBetaUncertain
    global betaUncertainName
    global betaUncertainLower
    global betaUncertainHigher
    global betaUncertainAlphas

    global numDiscreteDesignSetString
    global normalDiscreteDesignSetName
    global normalDiscreteSetValues


    global numResultFiles
    global outputResultFiles

    exists = os.path.isfile(fileName)

    print(exists)

    if not exists:
        return False

    with open(fileName,'r') as data_file:
        data = json.load(data_file)
        if data.get("randomVariables"):
            for k in data["randomVariables"]:

                if (k["distribution"] == "Normal"):
                    normalUncertainName.append(k["name"])
                    normalUncertainMean.append(k["mean"])
                    normalUncertainStdDev.append(k["stdDev"])
                    numNormalUncertain += 1
                    numRandomVariables += 1

                elif (k["distribution"] == "Lognormal"):
                    lognormalUncertainName.append(k["name"])
                    lognormalUncertainMean.append(k["mean"])
                    lognormalUncertainStdDev.append(k["stdDev"])
                    numLognormalUncertain += 1
                elif (k["distribution"] == "Constant"):
                    constantStateName.append(k["name"])
                    constantStateValue.append(k["value"])
                    numConstantState += 1
                elif (k["distribution"] == "Uniform"):
                    print("Hellooo,, Setting lower upper bounds...")
                    uniformUncertainName.append(k["name"])
                    uniformUncertainLower.append(k["lowerbound"])
                    uniformUncertainUpper.append(k["upperbound"])
                    numUniformUncertain += 1
                elif (k["distribution"] == "ContinuousDesign"):
                    continuousDesignName.append(k["name"])
                    continuousDesignLower.append(k["lowerbound"])
                    continuousDesignUpper.append(k["upperbound"])
                    continuousDesignInitialPoint.append(k["initialpoint"])
                    numContinuousDesign += 1
                elif (k["distribution"] == "Weibull"):
                    weibullUncertainName.append(k["name"])
                    weibullUncertainAlphas.append(k["scaleparam"])
                    weibullUncertainBetas.append(k["shapeparam"])
                    numWeibullUncertain += 1
                elif (k["distribution"] == "Gamma"):
                    gammaUncertainName.append(k["name"])
                    gammaUncertainAlphas.append(k["alphas"])
                    gammaUncertainBetas.append(k["betas"])
                    numGammaUncertain += 1
                elif (k["distribution"] == "Gumbell"):
                    gumbellUncertainName.append(k["name"])
                    gumbellUncertainAlphas.append(k["alphas"])
                    gumbellUncertainBetas.append(k["betas"])
                    numGumbellUncertain += 1
                elif (k["distribution"] == "Beta"):
                    betaUncertainName.append(k["name"])
                    betaUncertainLower.append(k["upperBounds"])
                    betaUncertainUpper.append(k["lowerBounds"])
                    betaUncertainAlphas.append(k["alphas"])
                    betaUncertainBetas.append(k["betas"])
                    numBetaUncertain += 1
                elif (k["distribution"] == "discrete_design_set_string"):
                    discreteDesignSetStringName.append(k["name"])
                    elements =[];
                    for l in k["elements"]:
                        elements.append(l)
                    elements.sort()
                    discreteDesignSetStringValues.append(elements)
                    print(elements)
                    numDiscreteDesignSetString += 1
                    numRandomVariables += 1

        if data.get("resultFiles"):
            for k in data["resultFiles"]:
                outputResultFiles.append(k)
                numResultFiles += 1
                print(k)

    return True

def add_dummy():

    global numRandomVariables

    global numDiscreteDesignSetString
    global discreteDesignSetStringName
    global discreteDesignSetStringValues

    discreteDesignSetStringName.append("dummy")
    elements =["1", "2"];
    discreteDesignSetStringValues.append(elements)
    numDiscreteDesignSetString += 1
    numRandomVariables += 1