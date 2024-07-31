import os  # noqa: INP001, D100
import shutil
from sys import platform

import fire


def runWorkflow(index):  # noqa: ANN001, ANN201, N802, D103
    index = int(index)

    shutil.copy(
        os.path.join(  # noqa: PTH118
            os.getcwd(),  # noqa: PTH109
            'InputFiles',
            'params_' + str(index) + '.template',
        ),
        os.path.join(os.getcwd(), 'params.in'),  # noqa: PTH109, PTH118
    )

    command2 = 'blank'
    if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
        command2 = os.path.join(os.getcwd(), 'driver')  # noqa: PTH109, PTH118
    elif platform == 'win32':
        command2 = os.path.join(os.getcwd(), 'driver.bat')  # noqa: PTH109, PTH118

    # os.system(command1)
    os.system(command2)  # noqa: S605


if __name__ == '__main__':
    fire.Fire(runWorkflow)
