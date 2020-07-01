# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

"""
This module has classes and methods to manage databases used by pelicun.

.. rubric:: Contents

.. autosummary::

    convert_P58_data_to_json
    create_HAZUS_EQ_json_files
    create_HAZUS_EQ_PGA_json_files
    create_HAZUS_HU_json_files

"""

from .base import *
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import shutil

def dict_generator(indict, pre=None):
    """
    Lists all branches of a tree defined by a dictionary.

    The dictionary can have nested dictionaries and lists. When encountering a
    list, its elements are returned as separate branches with each element's id
    created as a combination of the parent key and #i where i stands for the
    element number in the list.

    This method can process a json file and break it up into independent
    branches.

    """
    pre = pre[:] if pre else []
    if isinstance(indict, dict):
        for key, value in indict.items():
            if isinstance(value, dict):
                for d in dict_generator(value, pre + [key]):
                    yield d
            elif isinstance(value, list) or isinstance(value, tuple):
                for v_id, v in enumerate(value):
                    for d in dict_generator(v, pre + [key + f'#{v_id}']):
                        yield d
            else:
                yield pre + [key, value]
    else:
        yield pre + [indict]

def get_val_from_dict(indict, col):
    """
    Gets the value from a branch of a dictionary.

    The dictionary can have nested dictionaries and lists. When walking through
    lists, #i in the branch data identifies the ith element of the list.

    This method can be used to travel branches of a dictionary previously
    defined by the dict_generator method above.

    """

    val = indict

    for col_i in col:
        if col_i != ' ':
            if '#' in col_i:
                col_name, col_id = col_i.split('#')
                col_id = int(col_id)
                if (col_name in val.keys()) and (col_id < len(val[col_name])):
                    val = val[col_name][int(col_id)]
                else:
                    return None

            elif col_i in val.keys():
                val = val[col_i]
            else:
                return None

    return val

def convert_jsons_to_table(json_id_list, json_list, json_template):
    # Define the header for the data table based on the template structure
    header = np.array(
        [[col[:-1], len(col[:-1])] for col in dict_generator(json_template)])
    lvls = max(np.transpose(header)[1])
    header = [col + (lvls - size) * [' ', ] for col, size in header]

    # Use the header to initialize the DataFrame that will hold the data
    MI = pd.MultiIndex.from_tuples(header)

    json_DF = pd.DataFrame(columns=MI, index=json_id_list)
    json_DF.index.name = 'ID'

    # Load the data into the DF
    for json_id, json_data in zip(json_id_list, json_list):

        for col in json_DF.columns:

            val = get_val_from_dict(json_data, col)

            if val is not None:
                json_DF.at[json_id, col] = val

    # Remove empty rows and columns
    json_DF = json_DF.dropna(axis=0, how='all')
    json_DF = json_DF.dropna(axis=1, how='all')

    # Set the dtypes for the columns based on the template
    for col in json_DF.columns:
        dtype = get_val_from_dict(json_template, col)

        if dtype != 'string':
            try:
                json_DF[col] = json_DF[col].astype(dtype)
            except:
                print(col, dtype)
        else:
            json_DF[col] = json_DF[col].apply(str)

    return json_DF

def convert_Series_to_dict(comp_Series):
    """
    Converts data from a table to a json file

    """

    comp_Series = comp_Series.dropna(how='all')

    comp_dict = {}

    for branch in comp_Series.index:

        nested_dict = comp_dict
        parent_dict = None
        parent_val = None
        parent_id = None

        for val in branch:
            if val != ' ':
                if '#' in val:
                    val, list_id = val.split('#')
                    list_id = int(list_id)
                else:
                    list_id = None

                if val not in nested_dict.keys():
                    if list_id is not None:
                        nested_dict.update({val: []})
                    else:
                        nested_dict.update({val: {}})

                if list_id is not None:
                    if list_id > len(nested_dict[val]) - 1:
                        nested_dict[val].append({})
                    parent_dict = nested_dict
                    nested_dict = nested_dict[val][list_id]

                    parent_id = list_id

                else:
                    parent_dict = nested_dict
                    nested_dict = nested_dict[val]

                parent_val = val

        if isinstance(parent_dict[parent_val], dict):
            parent_dict[parent_val] = comp_Series[branch]
        else:
            parent_dict[parent_val][parent_id] = comp_Series[branch]

    return comp_dict