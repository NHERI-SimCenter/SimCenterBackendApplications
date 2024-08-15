#  # noqa: INP001, D100
# Copyright (c) 2024 The Regents of the University of California
#
# This file is a part of SimCenter backend applications.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 03-27-2024

# Import packages needed for setting up required packages:
import subprocess
import sys
from importlib import metadata as importlib_metadata

print('Initializing BRAILS...')  # noqa: T201

# If not installed, install BRAILS, argparse, and requests:
required = {'BRAILS', 'argparse', 'requests'}
installed = set()

# Detect installed packages using Python-provided importlib.metadata:
for x in importlib_metadata.distributions():
    try:  # noqa: SIM105
        installed.add(x.name)
    except:  # noqa: S110, PERF203, E722
        pass

# If installed packages could not be detected, use importlib_metadata backport:
if not installed:
    import importlib_metadata

    for x in importlib_metadata.distributions():
        try:  # noqa: SIM105
            installed.add(x.name)
        except:  # noqa: S110, PERF203, E722
            pass
missing = required - installed

# Install missing packages:
python = sys.executable
if missing:
    print('\nInstalling packages required for running this widget...')  # noqa: T201
    subprocess.check_call(  # noqa: S603
        [python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL
    )
    print('Successfully installed the required packages')  # noqa: T201

# If requests and BRAILS were previously installed ensure they are at their latest versions:
subprocess.check_call(  # noqa: S603
    [python, '-m', 'pip', 'install', 'requests', '-U'], stdout=subprocess.DEVNULL
)

import requests  # noqa: E402

latestBrailsVersion = requests.get('https://pypi.org/pypi/BRAILS/json').json()[  # noqa: S113, N816
    'info'
]['version']
if importlib_metadata.version('BRAILS') != latestBrailsVersion:
    print(  # noqa: T201
        '\nAn older version of BRAILS was detected. Updating to the latest BRAILS version..'
    )
    subprocess.check_call(  # noqa: S603
        [python, '-m', 'pip', 'install', 'BRAILS', '-U'], stdout=subprocess.DEVNULL
    )
    print('Successfully installed the latest version of BRAILS')  # noqa: T201

# Import packages required for running the latest version of BRAILS:
import argparse  # noqa: E402
import os  # noqa: E402
from time import gmtime, strftime  # noqa: E402

from brails.EnabledAttributes import BldgAttributes  # noqa: E402


# Define a standard way of printing program outputs:
def log_msg(msg):  # noqa: D103
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)  # noqa: T201


# Define a way to call BRAILS BldgAttributes and write them in a file:
def runBrails(outputfile):  # noqa: N802, D103
    attributes = BldgAttributes()
    with open(outputfile, 'w') as f:  # noqa: PTH123
        f.write('\n'.join(attributes))


# Define a way to collect GUI input:
def main(args):  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputFile', default=None)

    args = parser.parse_args(args)

    # Create the folder for the output file, if it does not exist:
    outdir = os.path.abspath(args.outputFile).replace(  # noqa: PTH100
        os.path.split(args.outputFile)[-1], ''
    )
    os.makedirs(outdir, exist_ok=True)  # noqa: PTH103

    # Run BRAILS  with the user-defined arguments:
    runBrails(args.outputFile)

    log_msg('BRAILS was successfully initialized')


# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])
