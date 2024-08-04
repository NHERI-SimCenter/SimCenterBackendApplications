# written: fmk  # noqa: CPY001, D100, EXE002, INP001

import json
import os
import sys
from time import gmtime, strftime

divider = '#' * 80
log_output = []

from WorkflowUtils import *  # noqa: E402, F403


def main(run_type, inputFile, applicationsRegistry):  # noqa: ANN001, ANN201, C901, D103, N803, PLR0912, PLR0914, PLR0915
    # the whole workflow is wrapped within a 'try' block.
    # a number of exceptions (files missing, explicit application failures, etc.) are
    # handled explicitly to aid the user.
    # But unhandled exceptions case the workflow to stop with an error, handled in the
    # exception block way at the bottom of this main() function
    try:  # noqa: PLR1702
        workflow_log(divider)  # noqa: F405
        workflow_log('Start of run')  # noqa: F405
        workflow_log(divider)  # noqa: F405
        workflow_log('workflow input file:       %s' % inputFile)  # noqa: F405, UP031
        workflow_log('application registry file: %s' % applicationsRegistry)  # noqa: F405, UP031
        workflow_log('runtype:                   %s' % run_type)  # noqa: F405, UP031
        workflow_log(divider)  # noqa: F405

        #
        # first we parse the applications registry to load all possible applications
        #  - for each application type we place in a dictionary key being name, value containing path to executable
        #
        with open(applicationsRegistry) as data_file:  # noqa: PLW1514, PTH123
            registryData = json.load(data_file)  # noqa: N806
            # convert all relative paths to full paths
            relative2fullpath(registryData)  # noqa: F405

        A = 'Applications'  # noqa: N806
        Applications = dict()  # noqa: C408, N806
        appList = 'Building Event Modeling EDP Simulation UQ DamageAndLoss'.split(  # noqa: N806
            ' '
        )
        appList = [a + A for a in appList]  # noqa: N806

        for app_type in appList:
            if app_type in registryData:
                xApplicationData = registryData[app_type]  # noqa: N806
                applicationsData = xApplicationData['Applications']  # noqa: N806

                for app in applicationsData:
                    appName = app['Name']  # noqa: N806
                    appExe = app['ExecutablePath']  # noqa: N806
                    if app_type not in Applications:
                        Applications[app_type] = dict()  # noqa: C408
                    Applications[app_type][appName] = appExe

        #
        # open input file, and parse json into data
        #

        with open(inputFile) as data_file:  # noqa: PLW1514, PTH123
            data = json.load(data_file)
            # convert all relative paths to full paths
            relative2fullpath(data)  # noqa: F405

        #
        # get all application data, quit if error
        #

        if 'WorkflowType' in data:
            typeWorkflow = data['WorkflowType']  # noqa: N806, F841
        else:
            raise WorkFlowInputError('Need a Workflow Type')  # noqa: EM101, F405, TRY003, TRY301

        # check correct workflow type

        #
        # now we parse for the applications & app specific data in workflow
        #

        if 'Applications' in data:
            available_apps = data['Applications']
        else:
            raise WorkFlowInputError('Need an Applications Entry')  # noqa: EM101, F405, TRY003, TRY301

        #
        # get building application and its data
        #

        if 'Buildings' in available_apps:
            buildingApp = available_apps['Buildings']  # noqa: N806

            if 'BuildingApplication' in buildingApp:
                buildingApplication = buildingApp['BuildingApplication']  # noqa: N806

                # check building app in registry, if so get full executable path
                buildingAppData = buildingApp['ApplicationData']  # noqa: N806
                if (
                    buildingApplication  # noqa: SIM118
                    in Applications['BuildingApplications'].keys()
                ):
                    buildingAppExe = Applications['BuildingApplications'].get(  # noqa: N806
                        buildingApplication
                    )
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY301
                        'Building application %s not in registry'  # noqa: UP031
                        % buildingApplication
                    )

            else:
                raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                    'Need a Building Generator Application in Buildings'  # noqa: EM101
                )

        else:
            raise WorkFlowInputError('Need a Buildings Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        #
        # get events, for each the  application and its data .. FOR NOW 1 EVENT
        #

        if 'Events' in available_apps:
            events = available_apps['Events']

            for event in events:
                if 'EventClassification' in event:
                    eventClassification = event['EventClassification']  # noqa: N806
                    if eventClassification == 'Earthquake':
                        if 'EventApplication' in event:
                            eventApplication = event['EventApplication']  # noqa: N806
                            eventAppData = event['ApplicationData']  # noqa: N806
                            eventData = event['ApplicationData']  # noqa: N806, F841
                            if (
                                eventApplication  # noqa: SIM118
                                in Applications['EventApplications'].keys()
                            ):
                                eventAppExe = Applications['EventApplications'].get(  # noqa: N806
                                    eventApplication
                                )
                            else:
                                raise WorkFlowInputError(  # noqa: F405, TRY301
                                    'Event application %s not in registry'  # noqa: UP031
                                    % eventApplication
                                )

                        else:
                            raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                                'Need an EventApplication section'  # noqa: EM101
                            )

                    else:
                        raise WorkFlowInputError(  # noqa: F405, TRY301
                            'Event classification must be Earthquake, not %s'  # noqa: UP031
                            % eventClassification
                        )

                else:
                    raise WorkFlowInputError('Need Event Classification')  # noqa: EM101, F405, TRY003, TRY301

        else:
            raise WorkFlowInputError('Need an Events Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        #
        # get modeling application and its data
        #

        if 'Modeling' in available_apps:
            modelingApp = available_apps['Modeling']  # noqa: N806

            if 'ModelingApplication' in modelingApp:
                modelingApplication = modelingApp['ModelingApplication']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                modelingAppData = modelingApp['ApplicationData']  # noqa: N806
                if (
                    modelingApplication  # noqa: SIM118
                    in Applications['ModelingApplications'].keys()
                ):
                    modelingAppExe = Applications['ModelingApplications'].get(  # noqa: N806
                        modelingApplication
                    )
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY301
                        'Modeling application %s not in registry'  # noqa: UP031
                        % modelingApplication
                    )

            else:
                raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                    'Need a ModelingApplication in Modeling data'  # noqa: EM101
                )

        else:
            raise WorkFlowInputError('Need a Modeling Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        #
        # get edp application and its data
        #

        if 'EDP' in available_apps:
            edpApp = available_apps['EDP']  # noqa: N806

            if 'EDPApplication' in edpApp:
                edpApplication = edpApp['EDPApplication']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                edpAppData = edpApp['ApplicationData']  # noqa: N806
                if edpApplication in Applications['EDPApplications'].keys():  # noqa: SIM118
                    edpAppExe = Applications['EDPApplications'].get(edpApplication)  # noqa: N806
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        'EDP application %s not in registry',  # noqa: EM101
                        edpApplication,
                    )

            else:
                raise WorkFlowInputError('Need an EDPApplication in EDP data')  # noqa: EM101, F405, TRY003, TRY301

        else:
            raise WorkFlowInputError('Need an EDP Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        if 'Simulation' in available_apps:
            simulationApp = available_apps['Simulation']  # noqa: N806

            if 'SimulationApplication' in simulationApp:
                simulationApplication = simulationApp['SimulationApplication']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                simAppData = simulationApp['ApplicationData']  # noqa: N806
                if (
                    simulationApplication  # noqa: SIM118
                    in Applications['SimulationApplications'].keys()
                ):
                    simAppExe = Applications['SimulationApplications'].get(  # noqa: N806
                        simulationApplication
                    )
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        'Simulation application %s not in registry',  # noqa: EM101
                        simulationApplication,
                    )

            else:
                raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                    'Need an SimulationApplication in Simulation data'  # noqa: EM101
                )

        else:
            raise WorkFlowInputError('Need a Simulation Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        if 'UQ-Simulation' in available_apps:
            uqApp = available_apps['UQ-Simulation']  # noqa: N806

            if 'UQApplication' in uqApp:
                uqApplication = uqApp['UQApplication']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                uqAppData = uqApp['ApplicationData']  # noqa: N806
                if uqApplication in Applications['UQApplications'].keys():  # noqa: SIM118
                    uqAppExe = Applications['UQApplications'].get(uqApplication)  # noqa: N806
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        'UQ application %s not in registry',  # noqa: EM101
                        uqApplication,
                    )

            else:
                raise WorkFlowInputError('Need a UQApplication in UQ data')  # noqa: EM101, F405, TRY003, TRY301

        else:
            raise WorkFlowInputError('Need a Simulation Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        if 'Damage&Loss' in available_apps:
            DLApp = available_apps['Damage&Loss']  # noqa: N806

            if 'Damage&LossApplication' in DLApp:
                dlApplication = DLApp['Damage&LossApplication']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                dlAppData = DLApp['ApplicationData']  # noqa: N806
                if dlApplication in Applications['DamageAndLossApplications'].keys():  # noqa: SIM118
                    dlAppExe = Applications['DamageAndLossApplications'].get(  # noqa: N806
                        dlApplication
                    )
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY301
                        'Dmage & Loss application %s not in registry' % dlApplication  # noqa: UP031
                    )

            else:
                raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                    'Need a Damage&LossApplicationApplication in Damage & Loss data'  # noqa: EM101
                )

        else:
            raise WorkFlowInputError('Need a Simulation Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        workflow_log('SUCCESS: Parsed workflow input')  # noqa: F405
        workflow_log(divider)  # noqa: F405

        #
        # now invoke the applications
        #

        #
        # put building generator application data into list and exe
        #

        buildingsFile = 'buildings.json'  # noqa: N806
        buildingAppDataList = [buildingAppExe, buildingsFile]  # noqa: N806

        for key in buildingAppData.keys():  # noqa: SIM118
            buildingAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
            buildingAppDataList.append(
                buildingAppData.get(key).encode('ascii', 'ignore')
            )

        buildingAppDataList.append('--getRV')
        command, result, returncode = runApplication(buildingAppDataList)  # noqa: F405
        log_output.append([command, result, returncode])

        del buildingAppDataList[-1]

        #
        # now we need to open buildingsfile and for each building
        #  - get RV for EVENT file for building
        #  - get RV for SAM file for building
        #  - get EDP for buildings and event
        #  - get SAM for buildings, event and EDP
        #  - perform Simulation
        #  - getDL

        with open(buildingsFile) as data_file:  # noqa: PLW1514, PTH123
            data = json.load(data_file)

        for building in data:
            id = building['id']  # noqa: A001
            bimFILE = building['file']  # noqa: N806
            eventFILE = id + '-EVENT.json'  # noqa: N806
            samFILE = id + '-SAM.json'  # noqa: N806
            edpFILE = id + '-EDP.json'  # noqa: N806
            dlFILE = id + '-DL.json'  # noqa: N806
            simFILE = id + '-SIM.json'  # noqa: N806
            driverFile = id + '-driver'  # noqa: N806

            # open driver file & write building app (minus the --getRV) to it
            driverFILE = open(driverFile, 'w')  # noqa: N806, PLW1514, PTH123, SIM115
            for item in buildingAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031
            driverFILE.write('\n')

            # get RV for event
            eventAppDataList = [  # noqa: N806
                eventAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameEVENT',
                eventFILE,
            ]
            if eventAppExe.endswith('.py'):
                eventAppDataList.insert(0, 'python')

            for key in eventAppData.keys():  # noqa: SIM118
                eventAppDataList.append('-' + key.encode('ascii', 'ignore'))
                value = eventAppData.get(key)
                if os.path.exists(value) and not os.path.isabs(value):  # noqa: PTH110, PTH117
                    value = os.path.abspath(value)  # noqa: PTH100
                eventAppDataList.append(value.encode('ascii', 'ignore'))

            for item in eventAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031
            driverFILE.write('\n')

            eventAppDataList.append('--getRV')
            command, result, returncode = runApplication(eventAppDataList)  # noqa: F405
            log_output.append([command, result, returncode])

            # get RV for building model
            modelAppDataList = [  # noqa: N806
                modelingAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameEVENT',
                eventFILE,
                '--filenameSAM',
                samFILE,
            ]

            for key in modelingAppData.keys():  # noqa: SIM118
                modelAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
                modelAppDataList.append(
                    modelingAppData.get(key).encode('ascii', 'ignore')
                )

            for item in modelAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031
            driverFILE.write('\n')

            modelAppDataList.append('--getRV')
            command, result, returncode = runApplication(modelAppDataList)  # noqa: F405
            log_output.append([command, result, returncode])

            # get RV for EDP!
            edpAppDataList = [  # noqa: N806
                edpAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameEVENT',
                eventFILE,
                '--filenameSAM',
                samFILE,
                '--filenameEDP',
                edpFILE,
            ]

            for key in edpAppData.keys():  # noqa: SIM118
                edpAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
                edpAppDataList.append(edpAppData.get(key).encode('ascii', 'ignore'))

            for item in edpAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031
            driverFILE.write('\n')

            edpAppDataList.append('--getRV')
            command, result, returncode = runApplication(edpAppDataList)  # noqa: F405
            log_output.append([command, result, returncode])

            # get RV for Simulation
            simAppDataList = [  # noqa: N806
                simAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameSAM',
                samFILE,
                '--filenameEVENT',
                eventFILE,
                '--filenameEDP',
                edpFILE,
                '--filenameSIM',
                simFILE,
            ]

            for key in simAppData.keys():  # noqa: SIM118
                simAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
                simAppDataList.append(simAppData.get(key).encode('ascii', 'ignore'))

            for item in simAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031
            driverFILE.write('\n')

            simAppDataList.append('--getRV')
            command, result, returncode = runApplication(simAppDataList)  # noqa: F405
            log_output.append([command, result, returncode])

            # Adding CreateLoss to Dakota Driver
            dlAppDataList = [  # noqa: N806
                dlAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameEDP',
                edpFILE,
                '--filenameLOSS',
                dlFILE,
            ]

            for key in dlAppData.keys():  # noqa: SIM118
                dlAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
                dlAppDataList.append(dlAppData.get(key).encode('ascii', 'ignore'))

            for item in dlAppDataList:
                driverFILE.write('%s ' % item)  # noqa: UP031

            # perform the simulation
            driverFILE.close()

            uqAppDataList = [  # noqa: N806
                uqAppExe,
                '--filenameAIM',
                bimFILE,
                '--filenameSAM',
                samFILE,
                '--filenameEVENT',
                eventFILE,
                '--filenameEDP',
                edpFILE,
                '--filenameLOSS',
                dlFILE,
                '--filenameSIM',
                simFILE,
                'driverFile',
                driverFile,
            ]

            for key in uqAppData.keys():  # noqa: SIM118
                uqAppDataList.append('-' + key.encode('ascii', 'ignore'))  # noqa: FURB113
                uqAppDataList.append(simAppData.get(key).encode('ascii', 'ignore'))

            if run_type == 'run':
                workflow_log('Running Simulation...')  # noqa: F405
                workflow_log(' '.join(uqAppDataList))  # noqa: F405
                command, result, returncode = runApplication(uqAppDataList)  # noqa: F405
                log_output.append([command, result, returncode])
                workflow_log('Simulation ended...')  # noqa: F405
            else:
                workflow_log('Check run only. No simulation performed.')  # noqa: F405

    except WorkFlowInputError as e:  # noqa: F405
        workflow_log('workflow error: %s' % e.value)  # noqa: F405, UP031
        workflow_log(divider)  # noqa: F405
        exit(1)  # noqa: PLR1722

    # unhandled exceptions are handled here
    except:
        raise
        workflow_log('unhandled exception... exiting')  # noqa: F405
        exit(1)  # noqa: PLR1722


if __name__ == '__main__':
    if len(sys.argv) != 4:  # noqa: PLR2004
        print('\nNeed three arguments, e.g.:\n')  # noqa: T201
        print(  # noqa: T201
            '    python %s action workflowinputfile.json workflowapplications.json'  # noqa: UP031
            % sys.argv[0]
        )
        print('\nwhere: action is either check or run\n')  # noqa: T201
        exit(1)  # noqa: PLR1722

    run_type = sys.argv[1]
    inputFile = sys.argv[2]  # noqa: N816
    applicationsRegistry = sys.argv[3]  # noqa: N816

    main(run_type, inputFile, applicationsRegistry)

    workflow_log_file = 'workflow-log-%s.txt' % (  # noqa: UP031
        strftime('%Y-%m-%d-%H-%M-%S-utc', gmtime())
    )
    log_filehandle = open(workflow_log_file, 'wb')  # noqa: SIM115, PTH123

    print >> log_filehandle, divider  # noqa: F633
    print >> log_filehandle, 'Start of Log'  # noqa: F633
    print >> log_filehandle, divider  # noqa: F633
    print >> log_filehandle, workflow_log_file  # noqa: F633
    # nb: log_output is a global variable, defined at the top of this script.
    for result in log_output:
        print >> log_filehandle, divider  # noqa: F633
        print >> log_filehandle, 'command line:\n%s\n' % result[0]  # noqa: F633, UP031
        print >> log_filehandle, divider  # noqa: F633
        print >> log_filehandle, 'output from process:\n%s\n' % result[1]  # noqa: F633, UP031

    print >> log_filehandle, divider  # noqa: F633
    print >> log_filehandle, 'End of Log'  # noqa: F633
    print >> log_filehandle, divider  # noqa: F633

    workflow_log('Log file: %s' % workflow_log_file)  # noqa: F405, UP031
    workflow_log('End of run.')  # noqa: F405
