# written: fmk, adamzs  # noqa: EXE002, INP001, D100

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import json
import os
import posixpath
from time import gmtime, strftime

divider = '#' * 80
log_output = []

from WorkflowUtils import *  # noqa: E402, F403


def main(run_type, inputFile, applicationsRegistry):  # noqa: ANN001, ANN201, C901, N803, D103, PLR0912, PLR0915
    # the whole workflow is wrapped within a 'try' block.
    # a number of exceptions (files missing, explicit application failures, etc.) are
    # handled explicitly to aid the user.
    # But unhandled exceptions case the workflow to stop with an error, handled in the  # noqa: E501
    # exception block way at the bottom of this main() function
    try:
        workflow_log(divider)  # noqa: F405
        workflow_log('Start of run')  # noqa: F405
        workflow_log(divider)  # noqa: F405
        workflow_log('workflow input file:       %s' % inputFile)  # noqa: F405, UP031
        workflow_log('application registry file: %s' % applicationsRegistry)  # noqa: F405, UP031
        workflow_log('runtype:                   %s' % run_type)  # noqa: F405, UP031
        workflow_log(divider)  # noqa: F405

        #
        # first we parse the applications registry to load all possible applications
        #  - for each application type we place in a dictionary key being name, value containing path to executable  # noqa: E501
        #
        with open(applicationsRegistry) as data_file:  # noqa: PTH123
            registryData = json.load(data_file)  # noqa: N806
            # convert all relative paths to full paths

        A = 'Applications'  # noqa: N806
        Applications = dict()  # noqa: C408, N806
        appList = 'Event Modeling EDP Simulation UQ'.split(' ')  # noqa: N806
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

        with open(inputFile) as data_file:  # noqa: PTH123
            data = json.load(data_file)
            # convert all relative paths to full paths
            # relative2fullpath(data)  # noqa: ERA001

        if 'runDir' in data:
            runDIR = data['runDir']  # noqa: N806
        else:
            raise WorkFlowInputError('Need a runDir Entry')  # noqa: EM101, F405, TRY003, TRY301

        if 'remoteAppDir' in data:
            remoteAppDir = data['remoteAppDir']  # noqa: N806
        else:
            raise WorkFlowInputError('Need a remoteAppDir Entry')  # noqa: EM101, F405, TRY003, TRY301

        if 'localAppDir' in data:
            localAppDir = data['localAppDir']  # noqa: N806
        else:
            raise WorkFlowInputError('Need a localAppDir Entry')  # noqa: EM101, F405, TRY003, TRY301

        #
        # before running chdir to templatedir
        #

        workflow_log('run Directory:               %s' % runDIR)  # noqa: F405, UP031

        os.chdir(runDIR)
        os.chdir('templatedir')

        #
        # now we parse for the applications & app specific data in workflow
        #

        if 'Applications' in data:
            available_apps = data['Applications']
        else:
            raise WorkFlowInputError('Need an Applications Entry')  # noqa: EM101, F405, TRY003, TRY301

        #
        # get events, for each the  application and its data .. FOR NOW 1 EVENT
        #

        if 'Events' in available_apps:
            events = available_apps['Events']

            for event in events:
                if 'EventClassification' in event:
                    eventClassification = event['EventClassification']  # noqa: N806
                    if eventClassification == 'Earthquake':
                        if 'Application' in event:
                            eventApplication = event['Application']  # noqa: N806
                            eventAppData = event['ApplicationData']  # noqa: N806
                            eventData = event['ApplicationData']  # noqa: N806, F841

                            if (
                                eventApplication  # noqa: SIM118
                                in Applications['EventApplications'].keys()
                            ):
                                eventAppExe = Applications['EventApplications'].get(  # noqa: N806
                                    eventApplication
                                )
                                workflow_log(remoteAppDir)  # noqa: F405
                                workflow_log(eventAppExe)  # noqa: F405
                                eventAppExeLocal = posixpath.join(  # noqa: N806
                                    localAppDir, eventAppExe
                                )
                                eventAppExeRemote = posixpath.join(  # noqa: N806
                                    remoteAppDir, eventAppExe
                                )
                                workflow_log(eventAppExeRemote)  # noqa: F405
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

        if 'EDP' in available_apps:
            edpApp = available_apps['EDP']  # noqa: N806

            if 'Application' in edpApp:
                edpApplication = edpApp['Application']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                edpAppData = edpApp['ApplicationData']  # noqa: N806
                if edpApplication in Applications['EDPApplications'].keys():  # noqa: SIM118
                    edpAppExe = Applications['EDPApplications'].get(edpApplication)  # noqa: N806
                    edpAppExeLocal = posixpath.join(localAppDir, edpAppExe)  # noqa: N806
                    edpAppExeRemote = posixpath.join(remoteAppDir, edpAppExe)  # noqa: N806
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        f'EDP application {edpApplication} not in registry'  # noqa: EM102
                    )

            else:
                raise WorkFlowInputError('Need an EDPApplication in EDP data')  # noqa: EM101, F405, TRY003, TRY301

        else:
            raise WorkFlowInputError('Need an EDP Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        if 'Simulation' in available_apps:
            simulationApp = available_apps['Simulation']  # noqa: N806

            if 'Application' in simulationApp:
                simulationApplication = simulationApp['Application']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                simAppData = simulationApp['ApplicationData']  # noqa: N806
                if (
                    simulationApplication  # noqa: SIM118
                    in Applications['SimulationApplications'].keys()
                ):
                    simAppExe = Applications['SimulationApplications'].get(  # noqa: N806
                        simulationApplication
                    )
                    simAppExeLocal = posixpath.join(localAppDir, simAppExe)  # noqa: N806
                    simAppExeRemote = posixpath.join(remoteAppDir, simAppExe)  # noqa: N806
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        f'Simulation application {simulationApplication} not in registry'  # noqa: EM102, E501
                    )

            else:
                raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                    'Need an SimulationApplication in Simulation data'  # noqa: EM101
                )

        else:
            raise WorkFlowInputError('Need a Simulation Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        if 'UQ' in available_apps:
            uqApp = available_apps['UQ']  # noqa: N806

            if 'Application' in uqApp:
                uqApplication = uqApp['Application']  # noqa: N806

                # check modeling app in registry, if so get full executable path
                uqAppData = uqApp['ApplicationData']  # noqa: N806
                if uqApplication in Applications['UQApplications'].keys():  # noqa: SIM118
                    uqAppExe = Applications['UQApplications'].get(uqApplication)  # noqa: N806
                    uqAppExeLocal = posixpath.join(localAppDir, uqAppExe)  # noqa: N806
                    uqAppExeRemote = posixpath.join(localAppDir, uqAppExe)  # noqa: N806, F841
                else:
                    raise WorkFlowInputError(  # noqa: F405, TRY003, TRY301
                        f'UQ application {uqApplication} not in registry'  # noqa: EM102
                    )

            else:
                raise WorkFlowInputError('Need a UQApplication in UQ data')  # noqa: EM101, F405, TRY003, TRY301

        else:
            raise WorkFlowInputError('Need a Simulation Entry in Applications')  # noqa: EM101, F405, TRY003, TRY301

        workflow_log('SUCCESS: Parsed workflow input')  # noqa: F405
        workflow_log(divider)  # noqa: F405

        #
        # now invoke the applications
        #

        # now we need to open buildingsfile and for each building
        #  - get RV for SAM file for building
        #  - get EDP for buildings and event
        #  - get SAM for buildings, event and EDP
        #  - perform Simulation
        #  - getDL

        inputFILE = 'dakota.json'  # noqa: N806
        eventFILE = 'EVENT.json'  # noqa: N806
        edpFILE = 'EDP.json'  # noqa: N806
        driverFile = 'driver'  # noqa: N806

        # open driver file & write building app (minus the --getRV) to it
        driverFILE = open(driverFile, 'w')  # noqa: SIM115, PTH123, N806

        # get RV for event
        eventAppDataList = [  # noqa: N806
            f'"{eventAppExeRemote}"',
            '--filenameBIM',
            inputFILE,
            '--filenameEVENT',
            eventFILE,
        ]
        if eventAppExe.endswith('.py'):
            eventAppDataList.insert(0, 'python')

        for key in eventAppData.keys():  # noqa: SIM118
            eventAppDataList.append('--' + key)
            value = eventAppData.get(key)
            eventAppDataList.append('' + value)

        for item in eventAppDataList:
            driverFILE.write('%s ' % item)  # noqa: UP031
        driverFILE.write('\n')

        eventAppDataList.append('--getRV')
        if eventAppExe.endswith('.py'):
            eventAppDataList[1] = '' + eventAppExeLocal
        else:
            eventAppDataList[0] = '' + eventAppExeLocal

        command, result, returncode = runApplication(eventAppDataList)  # noqa: F405
        log_output.append([command, result, returncode])

        edpAppDataList = [  # noqa: N806
            f'"{edpAppExeRemote}"',
            '--filenameBIM',
            inputFILE,
            '--filenameEVENT',
            eventFILE,
            '--filenameSAM',
            'drivel',
            '--filenameEDP',
            edpFILE,
        ]

        if edpAppExe.endswith('.py'):
            edpAppDataList.insert(0, 'python')

        for key in edpAppData.keys():  # noqa: SIM118
            edpAppDataList.append('--' + key)
            edpAppDataList.append('' + edpAppData.get(key))

        for item in edpAppDataList:
            driverFILE.write('%s ' % item)  # noqa: UP031
        driverFILE.write('\n')

        if edpAppExe.endswith('.py'):
            edpAppDataList[1] = edpAppExeLocal
        else:
            edpAppDataList[0] = edpAppExeLocal

        edpAppDataList.append('--getRV')
        command, result, returncode = runApplication(edpAppDataList)  # noqa: F405
        log_output.append([command, result, returncode])

        # get RV for Simulation
        simAppDataList = [  # noqa: N806
            f'"{simAppExeRemote}"',
            '--filenameBIM',
            inputFILE,
            '--filenameSAM',
            'drivel',
            '--filenameEVENT',
            eventFILE,
            '--filenameEDP',
            edpFILE,
            '--filenameSIM',
            'drivel',
        ]

        if simAppExe.endswith('.py'):
            simAppDataList.insert(0, 'python')

        for key in simAppData.keys():  # noqa: SIM118
            simAppDataList.append('--' + key)
            simAppDataList.append('' + simAppData.get(key))

        for item in simAppDataList:
            driverFILE.write('%s ' % item)  # noqa: UP031
        driverFILE.write('\n')

        simAppDataList.append('--getRV')
        if simAppExe.endswith('.py'):
            simAppDataList[1] = simAppExeLocal
        else:
            simAppDataList[0] = simAppExeLocal

        command, result, returncode = runApplication(simAppDataList)  # noqa: F405
        log_output.append([command, result, returncode])

        # perform the simulation
        driverFILE.close()
        print('HELLO')  # noqa: T201

        uqAppDataList = [  # noqa: N806
            f'"{uqAppExeLocal}"',
            '--filenameBIM',
            inputFILE,
            '--filenameSAM',
            'drivel',
            '--filenameEVENT',
            eventFILE,
            '--filenameEDP',
            edpFILE,
            '--filenameSIM',
            'drivel',
            '--driverFile',
            driverFile,
        ]

        #        uqAppDataList = ['"{}"'.format(uqAppExeLocal), '--filenameBIM', inputFILE, '--filenameEVENT', eventFILE, '--driverFile', driverFile]  # noqa: ERA001, E501

        print(uqAppDataList)  # noqa: T201

        if uqAppExe.endswith('.py'):
            uqAppDataList.insert(0, 'python')
            uqAppDataList[1] = uqAppExeLocal

        uqAppDataList.append('--runType')
        uqAppDataList.append(run_type)

        print(uqAppDataList)  # noqa: T201

        for key in uqAppData.keys():  # noqa: SIM118
            uqAppDataList.append('--' + key)
            value = uqAppData.get(key)
            if isinstance(value, string_types):
                uqAppDataList.append('' + value)
            else:
                uqAppDataList.append('' + str(value))

        if run_type == 'run' or run_type == 'set_up':  # noqa: PLR1714
            workflow_log('Running Simulation...')  # noqa: F405
            workflow_log(' '.join(uqAppDataList))  # noqa: F405
            command, result, returncode = runApplication(uqAppDataList)  # noqa: F405
            log_output.append([command, result, returncode])
            workflow_log('Simulation ended...')  # noqa: F405
        else:
            workflow_log('Setup run only. No simulation performed.')  # noqa: F405

    except WorkFlowInputError as e:  # noqa: F405
        print('workflow error: %s' % e.value)  # noqa: T201, UP031
        workflow_log('workflow error: %s' % e.value)  # noqa: F405, UP031
        workflow_log(divider)  # noqa: F405
        exit(1)  # noqa: PLR1722

    # unhandled exceptions are handled here
    except Exception:
        # print('workflow error: %s' % e.value)  # noqa: ERA001
        workflow_log('unhandled exception... exiting')  # noqa: F405
        raise


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
    log_filehandle = open(workflow_log_file, 'w')  # noqa: SIM115, PTH123

    print(type(log_filehandle))  # noqa: T201
    print(divider, file=log_filehandle)
    print('Start of Log', file=log_filehandle)
    print(divider, file=log_filehandle)
    print(workflow_log_file, file=log_filehandle)
    # nb: log_output is a global variable, defined at the top of this script.
    for result in log_output:
        print(divider, file=log_filehandle)
        print('command line:\n%s\n' % result[0], file=log_filehandle)  # noqa: UP031
        print(divider, file=log_filehandle)
        print('output from process:\n%s\n' % result[1], file=log_filehandle)  # noqa: UP031

    print(divider, file=log_filehandle)
    print('End of Log', file=log_filehandle)
    print(divider, file=log_filehandle)

    workflow_log('Log file: %s' % workflow_log_file)  # noqa: F405, UP031
    workflow_log('End of run.')  # noqa: F405
