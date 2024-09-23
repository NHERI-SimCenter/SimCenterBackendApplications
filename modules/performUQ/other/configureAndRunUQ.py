# written: Michael Gardner @ UNR, Aakash Bangalore Satish @ UCB  # noqa: INP001, D100

# Use the UQpy driver as a starting point if you want to add other UQ capabilities


def configureAndRunUQ(  # noqa: N802
    uqData,  # noqa: N803
    simulationData,  # noqa: N803
    randomVarsData,  # noqa: N803
    demandParams,  # noqa: N803
    workingDir,  # noqa: N803
    runType,  # noqa: N803
    localAppDir,  # noqa: N803
    remoteAppDir,  # noqa: N803
):
    """This function configures and runs a UQ simulation based on the input
    UQ driver and its associated inputs, simulation configuration, random
    variables, and requested demand parameters

    Input:
    uqData:         JsonObject that specifies the UQ driver and other options as input into the quoFEM GUI
    simulationData: JsonObject that contains information on the analysis package to run and its
                    configuration as input in the quoFEM GUI
    randomVarsData: JsonObject that specifies the input random variables, their distributions,
                    and associated parameters as input in the quoFEM GUI
    demandParams:   JsonObject that specifies the demand parameters as input in the quoFEM GUI
    workingDir:     Directory in which to run simulations and store temporary results
    runType:        Specifies whether computations are being run locally or on an HPC cluster
    localAppDir:    Directory containing apps for local run
    remoteAppDir:   Directory containing apps for remote run
    """  # noqa: D205, D400, D401, D404
    uqDriverOptions = ['UQpy', 'HeirBayes']  # noqa: N806

    for val in uqData['Parameters']:
        if val['name'] == 'UQ Driver':
            uqDriver = val['value']  # noqa: N806

    if uqDriver not in uqDriverOptions:
        raise ValueError(
            'ERROR: configureAndRunUQ.py: UQ driver not recognized.'  # noqa: ISC003
            + ' Either input incorrectly or class to run UQ driver not'
            + ' implemented: ',
            uqDriver,
        )
    else:  # noqa: RET506
        if uqDriver == 'UQpy' or uqDriver == 'HeirBayes':  # noqa: PLR1714
            pass

        uqDriverClass = locals()[uqDriver + 'Runner']  # noqa: N806
        uqDriverClass().runUQ(
            uqData,
            simulationData,
            randomVarsData,
            demandParams,
            workingDir,
            runType,
            localAppDir,
            remoteAppDir,
        )
