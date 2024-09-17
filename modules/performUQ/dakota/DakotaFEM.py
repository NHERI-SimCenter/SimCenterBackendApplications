import json  # noqa: INP001, D100
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120

import argparse
import platform
import shutil
import stat
import subprocess
from random import randrange

from preprocessJSON import preProcessDakota


def str2bool(v):  # noqa: D103
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):  # noqa: RET505
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # noqa: EM101, TRY003


def main(args):  # noqa: C901, D103
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

    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM')

    parser.add_argument('--driverFile')

    parser.add_argument('--method', default='LHS')
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=randrange(1, 1000))
    parser.add_argument('--samples2', type=int, default=None)
    parser.add_argument('--seed2', type=int, default=None)
    parser.add_argument('--ismethod', default=None)
    parser.add_argument('--dataMethod', default=None)
    parser.add_argument('--dataMethod2', default=None)

    parser.add_argument('--type')
    parser.add_argument('--concurrency', type=int, default=None)
    parser.add_argument('--keepSamples', default=True, type=str2bool)
    parser.add_argument('--detailedLog', default=False, type=str2bool)
    parser.add_argument('--runType')

    args, unknowns = parser.parse_known_args()

    # Reading input arguments
    aimName = args.filenameAIM  # noqa: N806
    samName = args.filenameSAM  # noqa: N806
    evtName = args.filenameEVENT  # noqa: N806
    edpName = args.filenameEDP  # noqa: N806
    simName = args.filenameSIM  # noqa: N806
    driverFile = args.driverFile  # noqa: N806

    uqData = dict(  # noqa: C408, N806
        method=args.method,
        samples=args.samples,
        samples2=args.samples2,
        seed=args.seed,
        seed2=args.seed2,
        ismethod=args.ismethod,
        dataMethod=args.dataMethod,
        dataMethod2=args.dataMethod2,
        concurrency=args.concurrency,
        keepSamples=args.keepSamples,
    )

    if (
        uqData['samples'] is None
    ):  # this happens when the uq details are stored at the wrong place in the AIM file
        with open(aimName, encoding='utf-8') as data_file:  # noqa: PTH123
            uq_info = json.load(data_file)['UQ']

        if 'samplingMethodData' in uq_info.keys():  # noqa: SIM118
            uq_info = uq_info['samplingMethodData']
            for attribute in uqData:
                if attribute not in ['concurrency', 'keepSamples']:
                    uqData[attribute] = uq_info.get(attribute, None)

    runDakota = args.runType  # noqa: N806

    # Run Preprocess for Dakota
    scriptDir = os.path.dirname(os.path.realpath(__file__))  # noqa: PTH120, N806, F841
    numRVs = preProcessDakota(  # noqa: N806, F841
        aimName, evtName, samName, edpName, simName, driverFile, runDakota, uqData
    )

    # Setting Workflow Driver Name
    workflowDriverName = 'workflow_driver'  # noqa: N806
    if (platform.system() == 'Windows') and (runDakota == 'run'):
        workflowDriverName = 'workflow_driver.bat'  # noqa: N806

    # Create Template Directory and copy files
    st = os.stat(workflowDriverName)  # noqa: PTH116
    os.chmod(workflowDriverName, st.st_mode | stat.S_IEXEC)  # noqa: PTH101
    # shutil.copy(workflowDriverName, "templatedir")
    # shutil.copy("{}/dpreproSimCenter".format(scriptDir), os.getcwd())
    shutil.move(aimName, 'aim.j')
    shutil.move(evtName, 'evt.j')
    if os.path.isfile(samName):  # noqa: PTH113
        shutil.move(samName, 'sam.j')
    shutil.move(edpName, 'edp.j')
    # if os.path.isfile(simName): shutil.move(simName, "sim.j")

    # copy the dakota input file to the main working dir for the structure
    shutil.move('dakota.in', '../')

    # change dir to the main working dir for the structure
    os.chdir('../')

    if runDakota == 'run':
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
            result = e.output
            returncode = e.returncode

        if args.detailedLog:  # print detailed output if detailed log is requested
            if platform.system() == 'Windows':
                result = result.decode(sys.stdout.encoding)

            print(result, returncode)  # noqa: T201


if __name__ == '__main__':
    main(sys.argv[1:])
