# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of the SimCenter Backend Applications.
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
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
#
# Modified 'cb-cities' code provided by the Soga Research Group UC Berkeley
# Dr. Stevan Gavrilovic


from scipy.spatial import cKDTree
import itertools
import os, sys, json, posixpath
import pandas as pd
import numpy as np

from operator import itemgetter

def ckdnearest(gdfA, gdfB, gdfB_cols=['pgv']):
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True)], axis=1)
    return gdf

# def pgv_node2pipe(pipe_info,node_info):
#     pgvs = []
#     for index, row in pipe_info.iterrows():
#         nodes = [str(row['node1']),str(row['node2'])]
#         end_nodes_info = node_info.loc[node_info['node_id'].isin(nodes)]
#         pgv = np.mean(end_nodes_info['pgv'])
#         pgvs.append(pgv)
#     return pgvs

def pgv_node2pipe(pipe_info,node_info):
    res = []
    
    node_ids = np.array(node_info['node_id'])
    pgvs = np.array(node_info['pgv'])
    
    n1s = np.array(pipe_info['node1'])
    n2s = np.array(pipe_info['node2'])
    
    for n1,n2 in zip(n1s,n2s):
        pgv = (pgvs[node_ids == n1]+pgvs[node_ids == n2])[0]/2
        res.append(pgv)
    return res

def get_prefix(file_path):
    file_name = file_path.split('/')[-1]
    prefix = file_name.split('.')[0]
    return prefix


# Get the PGV value for the pipe
def add_pgv2pipe(pipe):

    reg_event = pipe['RegionalEvent']
    events = pipe['Events'][0]
    
    event_folder_path = events['EventFolderPath']
    
    event_array = events['Events']

    event_units = reg_event['units']

    pgvs = np.array([])
    
    for eventFile, scaleFactor  in event_array :
        
        # Discard the numbering at the end of the csv file name
        eventFile = eventFile[:len(eventFile)-8]

        # Get the path to the event file
        path_Event_File = posixpath.join(event_folder_path,eventFile)
        
        # Read in the event file IM List
        eventIMList = pd.read_csv(path_Event_File, header=0)
        
        PGVCol = eventIMList.loc[:,'PGV']
        
        pgv_unit = event_units['PGV']
        
        # Scale the PGVs and account for units - fragility functions are in inch per second
        if  pgv_unit == 'cmps' :
            PGVCol = PGVCol.apply(lambda x: cm2inch(x) * scaleFactor)
        elif pgv_unit == 'inps' :
            continue
        else :
            print("Error, only 'cmps' and 'inps' units are supported for PGV")
                
        pgvs = np.append(pgvs,PGVCol.values)

        
    pipe['pgv'] = pgvs

    return pipe

#    pgv_info = pd.read_csv(pgv_path)
#    gd_pgv = gpd.GeoDataFrame(
#        pgv_info, geometry=gpd.points_from_xy(pgv_info.lon, pgv_info.lat))
#    df = ckdnearest(node_info,gd_pgv)
#    pgvs = pgv_node2pipe(pipe_info,df)
#    pipe_info['pgv'] = pgvs
#
#    return pipe_info
    

k_dict = {'A':1,
          'C': 1,
          'D':0.5,
          'F':1,
          'H':1,
          'K':1,
          'N':1,
          None:1,
          'T':1,
          'R':1,
          'L':1,
          'S':0.6,
          'W':1}
          
def cm2inch(cm):
    return 39.3701*cm/100

def calculate_fail_repairrate(k,pgv,l):

    rr = k*0.00187*pgv/1000
    failure_rate = 1- np.power(np.e,-rr*l)
    
    return failure_rate

def get_pipe_failrate(pipe):

    pipe_GI = pipe['GeneralInformation']
    
    m,l,pgv = pipe_GI['material'],pipe_GI['length'],pipe['pgv']
    
    pipeRR = calculate_fail_repairrate(k_dict[m],l,pgv)
            
    return pipeRR

def add_failrate2pipe(pipe):

    pipe = add_pgv2pipe(pipe)
        
    pipe['fail_prob'] = get_pipe_failrate(pipe)
    
    return pipe
    
#
#
#    pgv_prefix = get_prefix(pgv_path)
#    save_path = save_folder + 'pipes_'+ pgv_prefix +'.geojson'
#    pipe_info.to_file(save_path, driver="GeoJSON")
#    print (f'saved to {save_path}')



#    pipe_info['fail_prob'] = get_pipe_failrate(pipe_info)
#
#
#    pgv_prefix = get_prefix(pgv_path)
#    save_path = save_folder + 'pipes_'+ pgv_prefix +'.geojson'
#    pipe_info.to_file(save_path, driver="GeoJSON")
#    print (f'saved to {save_path}')

def get_bar_ranges(space):
    ranges = []
    for i in range(1,len(space)):
         ranges.append((space[i-1],space[i]))
    return ranges

def get_failure_groups(fail_probs,min_thre = 1e-3,num_groups = 10):
    valid_fails = [fail_prob for fail_prob in fail_probs if fail_prob > min_thre]
    count, space = np.histogram(valid_fails, num_groups)
    ranges = get_bar_ranges(space)
    return ranges

def get_failed_pipes_mask(pipe_info,groups):
    broken_pipes = np.zeros(len(pipe_info))

    for r in groups:
        pipes_mask = list((pipe_info['fail_prob'] > r[0]) & (pipe_info['fail_prob'] < r[1]))

        valid_indices = np.nonzero(pipes_mask)[0]
        num_fails = int(np.mean(r) * sum(pipes_mask))
        
        fail_indices = np.random.choice(valid_indices,num_fails,replace = False)
        
        broken_pipes[fail_indices] = 1
        
    return broken_pipes
    
def generate_leak_diameter(pipe_diam,min_ratio = 0.05, max_ratio = 0.25):
    r = np.random.uniform(min_ratio,max_ratio)
    return pipe_diam*r
    
def get_leak_sizes(pipe_info):
    leak_size = np.zeros(len(pipe_info))
    for index, row in pipe_info.iterrows():
        d,repair = row['diameter'],row['repair']
        if repair:
            leak_d = generate_leak_diameter(d)
            leak_size[index] = leak_d
    return leak_size

def fail_pipes_number(pipe_info):
    
    fail_probs = np.array(pipe_info['fail_prob'])
    groups = get_failure_groups(fail_probs)
    
    failed_pipes_mask = get_failed_pipes_mask(pipe_info,groups)
    num_failed_pipes = sum(failed_pipes_mask)
    print (f'number of failed pipes are : {num_failed_pipes}')
    return num_failed_pipes
