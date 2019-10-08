from __future__ import division, print_function
import os, sys
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    string_types = str

import argparse, posixpath, ntpath, json

def create_SAM(BIM_file, EVENT_file, SAM_file, model_file, getRV):

    root_SAM = {}

    root_SAM['mainScript'] = model_file
    root_SAM['type'] = 'OpenSeesPyInput'

    with open(BIM_file, 'r') as f:
        root_BIM = json.load(f)

    try:
        root_SIM = root_BIM['StructuralInformation']
        nodes = root_SIM['nodes']
        root_SAM['ndm'] = root_SIM['ndm']
    except:
        raise ValueError("OpenSeesPyInput - structural information missing")

    if 'ndf' in root_SIM.keys():
        root_SAM['ndf'] = root_SIM['ndf']

    node_map = []
    for floor, node in enumerate(nodes):
        node_entry = {}
        node_entry['node'] = node
        node_entry['cline'] = '1'
        node_entry['floor'] = '{}'.format(floor)
        node_map.append(node_entry)

    root_SAM['NodeMapping'] = node_map

    root_SAM['numStory'] = floor

    try:
        root_RV = root_SIM['randomVar']
    except:
        raise ValueError("OpenSeesPyInput - randomVar section missing")

    rv_array = []
    for rv in root_RV:
        rv_array.append(rv)

    root_SAM['randomVar'] = rv_array

    with open(SAM_file, 'w') as f:
        json.dump(root_SAM, f, indent=2)    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--fileName')
    parser.add_argument('--filePath')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(create_SAM(
        args.filenameBIM, args.filenameEVENT, args.filenameSAM,
        args.fileName, args.getRV))