from __future__ import division, print_function
import os, sys
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    # prepare the header
    header_out = []
    for h_label in header:
        h_label = h_label.strip()
        if h_label.endswith('_h'):
            header_out.append(f'1-{h_label[:-2]}-1-1')
        elif h_label.endswith('_v'):
            header_out.append(f'1-{h_label[:-2]}-1-3')
        elif h_label.endswith('_x'):
            header_out.append(f'1-{h_label[:-2]}-1-1')
        elif h_label.endswith('_y'):
            header_out.append(f'1-{h_label[:-2]}-1-2')
        else:
            header_out.append(f'1-{h_label.strip()}-1-1')
    string_types = str

import argparse, posixpath, ntpath, json

def write_RV():
    pass 

def create_EDP(EVENT_input_path, EDP_input_path):
    
    # load the EDP file
    with open(EDP_input_path, 'r') as f:
        EDP_in = json.load(f)

    # load the EVENT file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)

    # store the IM(s) in the EDP file
    for edp in EDP_in["EngineeringDemandParameters"][0]["responses"]:
        for im in EVENT_in["Events"]:
            if edp["type"] in im.keys():
                edp["scalar_data"] = [im[edp["type"]]]

    with open(EDP_input_path, 'w') as f:
        json.dump(EDP_in, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM', default=None)
    parser.add_argument('--filenameSAM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV())
    else:
        sys.exit(create_EDP(args.filenameEVENT, args.filenameEDP))