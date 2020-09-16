# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# 

import argparse, posixpath, json, sys
import numpy as np

def write_RV(BIM_file, EVENT_file, data_dir):
	
	with open(BIM_file, 'r') as f:
		bim_data = json.load(f)

	event_file = {
		'randomVariables': [],
		'Events': []
	}

	events = bim_data['Events']['Events']

	if len(events) > 1:
		event_file['randomVariables'].append({
			'distribution': 'discrete_design_set_string',
			'name': 'eventID',
			'value': 'RV.eventID',
			'elements': []
		})
		event_file['Events'].append({
			'type': 'Seismic',
			'subtype': bim_data['Events']['Events'][0]['type'],
			'event_id': 'RV.eventID',
			'data_dir': data_dir
			})

		RV_elements = []
		for event in events:
			if event['EventClassification'] == 'Earthquake':
				RV_elements.append(event['fileName'])

		event_file['randomVariables'][0]['elements'] = RV_elements
	else:
		event_file['Events'].append({
			'type': 'Seismic',
			'subtype': bim_data['Events']['Events'][0]['type'],
			'event_id': events[0]['fileName'],
			'data_dir': data_dir
			})

	# if time histories are used, then load the first event
	if events[0]['type'] == 'timeHistory':
		event_file['Events'][0].update(load_record(events[0]['fileName'], 
									               data_dir, 
									               empty=len(events) > 1))

	with open(EVENT_file, 'w') as f:
		json.dump(event_file, f, indent=2)

def load_record(fileName, data_dir, scale_factor=1.0, empty=False):

	fileName = fileName.split('x')[0]

	with open(posixpath.join(data_dir,'{}.json'.format(fileName)), 'r') as f:
		event_data = json.load(f)

	event_dic = {
		'name': fileName,
		'dT' : event_data['dT'],
		'numSteps': len(event_data['data_x']),
		'timeSeries': [],
		'pattern': []
	}

	if not empty:
		for i, (src_label, tar_label) in enumerate(zip(['data_x', 'data_y'],
													   ['accel_X', 'accel_Y'])):
			if src_label in event_data.keys():

				event_dic['timeSeries'].append({
					'name': tar_label,
					'type': 'Value',
					'dT': event_data['dT'],
					'data': list(np.array(event_data[src_label])*scale_factor)
				})
				event_dic['pattern'].append({
					'type': 'UniformAcceleration',
					'timeSeries': tar_label,
					'dof': i+1
					})

	return event_dic

def get_records(BIM_file, EVENT_file, data_dir):
	
	with open(BIM_file, 'r') as f:
		bim_file = json.load(f)

	with open(EVENT_file, 'r') as f:
		event_file = json.load(f)

	event_id = event_file['Events'][0]['event_id']

	scale_factor = dict([(evt['fileName'], evt.get('factor',1.0)) for evt in bim_file["Events"]["Events"]])[event_id]

	event_file['Events'][0].update(
		load_record(event_id, data_dir, scale_factor))

	with open(EVENT_file, 'w') as f:
		json.dump(event_file, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--pathEventData')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
    	sys.exit(write_RV(args.filenameBIM, args.filenameEVENT, args.pathEventData))
    else:
    	sys.exit(get_records(args.filenameBIM, args.filenameEVENT, args.pathEventData))