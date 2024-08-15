# import functions for Python 2.X support  # noqa: INP001, D100
import os
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120

import argparse
import platform
import shutil
import stat
import subprocess

import numpy as np
from preprocessJSON import preProcessDakota


def main(args):  # noqa: D103
    # First we need to set the path and environment
    home = os.path.expanduser('~')  # noqa: PTH111
    env = os.environ
    if os.getenv('PEGASUS_WF_UUID') is not None:
        print('Pegasus job detected - Pegasus will set up the env')  # noqa: T201
    elif platform.system() == 'Darwin':
        env['PATH'] = env['PATH'] + f':{home}/bin'
        env['PATH'] = env['PATH'] + f':{home}/dakota/bin'
    elif platform.system() == 'Linux':
        env['PATH'] = env['PATH'] + f':{home}/bin'
        env['PATH'] = env['PATH'] + f':{home}/dakota/dakota-6.5/bin'
    elif platform.system() == 'Windows':
        pass
    else:
        print(f'PLATFORM {platform.system} NOT RECOGNIZED')  # noqa: T201

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM')

    parser.add_argument('--driverFile')

    parser.add_argument('--method', default='LHS')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=np.random.randint(1, 1000))
    parser.add_argument('--ismethod', default=None)
    parser.add_argument('--dataMethod', default=None)

    parser.add_argument('--trainingSamples', type=int, default=None)
    parser.add_argument('--trainingSeed', type=int, default=None)
    parser.add_argument('--trainingMethod', default=None)
    parser.add_argument('--samplingSamples', type=int, default=None)
    parser.add_argument('--samplingSeed', type=int, default=None)
    parser.add_argument('--samplingMethod', default=None)

    parser.add_argument('--type')
    parser.add_argument('--concurrency', type=int, default=None)
    parser.add_argument('--keepSamples', default='True')
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    # Reading input arguments
    aimName = args.filenameBIM  # noqa: N806
    samName = args.filenameSAM  # noqa: N806
    evtName = args.filenameEVENT  # noqa: N806
    edpName = args.filenameEDP  # noqa: N806
    simName = args.filenameSIM  # noqa: N806
    driverFile = args.driverFile  # noqa: N806

    uqData = dict(  # noqa: C408, N806
        method=args.method,
        samples=args.samples,
        seed=args.seed,
        ismethod=args.ismethod,
        dataMethod=args.dataMethod,
        samplingSamples=args.samplingSamples,
        samplingSeed=args.samplingSeed,
        samplingMethod=args.samplingMethod,
        trainingSamples=args.trainingSamples,
        trainingSeed=args.trainingSeed,
        trainingMethod=args.trainingMethod,
        concurrency=args.concurrency,
        keepSamples=args.keepSamples
        not in ['False', 'False', 'false', 'false', False],
    )

    runDakota = args.runType  # noqa: N806

    myScriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806

    # desktop applications
    if (
        uqData['samples'] is None
    ):  # this happens with new applications, workflow to change
        print('RUNNING PREPROCESSOR\n')  # noqa: T201
        osType = platform.system()  # noqa: N806
        preprocessorCommand = f'"{myScriptDir}/preprocessDakota" {aimName} {samName} {evtName} {edpName} {simName} {driverFile} {runDakota} {osType}'  # noqa: N806
        subprocess.Popen(preprocessorCommand, shell=True).wait()  # noqa: S602
        print('DONE RUNNING PREPROCESSOR\n')  # noqa: T201

    else:
        scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806, F841
        numRVs = preProcessDakota(  # noqa: N806, F841
            aimName,
            evtName,
            samName,
            edpName,
            simName,
            driverFile,
            runDakota,
            uqData,
        )

        shutil.move(aimName, 'aim.j')
        shutil.move(evtName, 'evt.j')
        if os.path.isfile(samName):  # noqa: PTH113
            shutil.move(samName, 'sam.j')
        shutil.move(edpName, 'edp.j')

    # Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'  # noqa: N806
    if (platform.system() == 'Windows') and (runDakota == 'runningLocal'):
        workflowDriverName = 'workflow_driver.bat'  # noqa: N806

    # Change permission of workflow driver
    st = os.stat(workflowDriverName)  # noqa: PTH116
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)  # noqa: PTH101

    # copy the dakota input file to the main working dir for the structure
    shutil.move('dakota.in', '../')

    # change dir to the main working dir for the structure
    os.chdir('../')

    if runDakota == 'runningLocal':
        dakotaCommand = (  # noqa: N806
            'dakota -input dakota.in -output dakota.out -error dakota.err'
        )
        print('running Dakota: ', dakotaCommand)  # noqa: T201
        try:
            result = subprocess.check_output(  # noqa: S602
                dakotaCommand, stderr=subprocess.STDOUT, shell=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output  # noqa: F841
            returncode = e.returncode  # noqa: F841


if __name__ == '__main__':
    main(sys.argv[1:])
