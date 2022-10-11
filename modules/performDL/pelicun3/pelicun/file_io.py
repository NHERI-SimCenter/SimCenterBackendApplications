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
# Pouria Kourehpaz
# Kuanshi Zhong

"""
This module has classes and methods that handle file input and output.

.. rubric:: Contents

.. autosummary::

    float_or_None
    int_or_None
    process_loc
    get_required_resources
    load_default_options
    merge_default_config
    save_to_csv
    load_data

"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import base


convert_dv_name = {
    'DV_rec_cost': 'Reconstruction Cost',
    'DV_rec_time': 'Reconstruction Time',
    'DV_injuries_0': 'Injuries lvl. 1',
    'DV_injuries_1': 'Injuries lvl. 2',
    'DV_injuries_2': 'Injuries lvl. 3',
    'DV_injuries_3': 'Injuries lvl. 4',
    'DV_red_tag': 'Red Tag ',
}

dependency_to_acronym = {
        'btw. Fragility Groups':   'FG',
        'btw. Performance Groups': 'PG',
        'btw. Floors':             'LOC',
        'btw. Directions':         'DIR',
        'btw. Component Groups':   'CSG',
        'btw. Damage States':      'DS',
        'Independent':             'IND',
        'per ATC recommendation':  'ATC',
    }

HAZUS_occ_converter = {
        'RES':  'Residential',
        'COM':  'Commercial',
        'REL':  'Commercial',
        'EDU':  'Educational',
        'IND':  'Industrial',
        'AGR':  'Industrial'
    }


# this is a convenience function for converting strings to float or None
def float_or_None(string):
    try:
        res = float(string)
        return res
    except ValueError:
        return None


def int_or_None(string):
    try:
        res = int(string)
        return res
    except ValueError:
        return None


def process_loc(string, stories):
    try:
        res = int(string)
        return [res, ]
    except ValueError:
        if "-" in string:
            s_low, s_high = string.split('-')
            s_low = process_loc(s_low, stories)
            s_high = process_loc(s_high, stories)
            return list(range(s_low[0], s_high[0]+1))
        if string == "all":
            return list(range(1, stories+1))
        if string == "top":
            return [stories, ]
        if string == "roof":
            return [stories, ]
        return None


def get_required_resources(input_path, assessment_type):
    """
    List the data files required to perform an assessment.

    It extracts the information from the config file about the methods and
    functional data required for the analysis and provides a list of paths to
    the files that would be used.
    This method is helpful in an HPC context to copy the required resources to
    the local node from the shared file storage.

    Parameters
    ----------
    input_path: string
        Location of the DL input json file.
    assessment_type: {'P58', 'HAZUS_EQ', 'HAZUS_HU'}
        Identifies the default databases based on the type of assessment.

    Returns
    -------
    resources: list of strings
        A list of paths to the required resource files.
    """

    resources = {}

    AT = assessment_type

    with open(input_path, 'r', encoding='utf-8') as f:
        jd = json.load(f)

    DL_input = jd['DamageAndLoss']

    loss = DL_input.get('LossModel', None)
    if loss is not None:
        inhabitants = loss.get('Inhabitants', None)
        dec_vars = loss.get('DecisionVariables', None)

        if dec_vars is not None:
            injuries = bool(dec_vars.get('Injuries', False))
    else:
        inhabitants = None
        dec_vars = None
        injuries = False

    # check if the user specified custom data sources
    path_CMP_data = DL_input.get("ComponentDataFolder", "")

    if path_CMP_data == "":
        # Use the P58 path as default
        path_CMP_data = base.pelicun_path / base.CMP_data_path[AT]

    resources.update({'component': path_CMP_data})

    # HAZUS combination of flood and wind losses
    if ((AT == 'HAZUS_HU') and (DL_input.get('Combinations', None) is not None)):
        path_combination_data = base.pelicun_path / base.CMP_data_path['HAZUS_MISC']
        resources.update({'combination': path_combination_data})

    # The population data is only needed if we are interested in injuries
    if inhabitants is not None:
        path_POP_data = inhabitants.get("PopulationDataFile", "")
    else:
        path_POP_data = ""

    if ((injuries) and (path_POP_data == "")):
        path_POP_data = base.pelicun_path / base.POP_data_path[AT]
        resources.update({'population': path_POP_data})

    return resources


def load_default_options(options_object):
    """
    Load the default_config.json file to set options to default values

    Parameters
    ----------
    options_object: Options
        Options object to be modified.
    """

    with open(base.pelicun_path / "settings/default_config.json",
              'r', encoding='utf-8') as f:
        options_object.defaults = json.load(f)

    options_object.set_options(options_object.defaults.get('Options', None))


def merge_default_config(config, options_object):
    """
    Merge the default_config.json file with the user specified config
    file.
    Parameters
    ----------
    options_object: Options
        Options object to be modified.
    """

    defaults = options_object.defaults

    if config is not None:

        if config.get('DemandAssessment', False):

            demand_def = defaults['DemandAssessment']
            demand_config = config['DemandAssessment']

            if 'Calibration' in demand_config.keys():

                calib_config = demand_config['Calibration']
                calib_def = demand_def['Calibration']

                for key, value in calib_def.items():

                    if key in {'Marginals', }:
                        continue

                    if key not in calib_def:
                        calib_def.update({key: value})

                marginal_config = calib_config['Marginals']
                marginal_def = calib_def['Marginals']

                for key, value in marginal_def.items():

                    if key not in marginal_config:
                        marginal_config.update({key: value})

            if 'Sampling' in demand_config.keys():

                sample_config = demand_config['Sampling']

                for key, value in demand_def['Sampling'].items():

                    if key not in sample_config:
                        sample_config.update({key: value})

            if 'OutputUnits' in demand_def.keys():

                if 'OutputUnits' not in demand_config.keys():
                    demand_config.update({'OutputUnits': {}})

                for key, value in demand_def['OutputUnits'].items():

                    if key not in demand_config['OutputUnits']:
                        demand_config['OutputUnits'].update({key: value})

    else:
        config = defaults

    return config


def save_to_csv(data, filepath, units=None, unit_conversion_factors=None,
                orientation=0, use_simpleindex=True, log_msg_method=None):
    """
    Saves data to a CSV file following standard SimCenter schema.

    The produced CSV files have a single header line and an index column. The
    second line may start with 'Units' in the index or the first column may be
    'Units' to provide the units for the data in the file.

    The following data types in pelicun can be saved with this function:

    Demand Data: Each column in a table corresponds to a demand type; each
    row corresponds to a simulation/sample. The header identifies each demand
    type. The user guide section of the documentation provides more
    information about the header format. Target need to be specified in the
    second row of the DataFrame.

    Parameters
    ----------
    data: DataFrame
        The data to save
    filepath: string
        The location of the destination file. If None, the data is not saved,
        but returned in the end.
    units: Series, optional
        Provides a Series with variables and corresponding units.
    unit_conversion_factors: dict
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the number of
        times a base unit fits in the alternative unit.
    level: string, optional
        Identifies the level referenced in the units dictionary when the data
        has a MultiIndex header.
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.
    use_simpleindex: bool, default: True
        If True, MultiIndex columns and indexes are converted to SimpleIndex
        before saving
    log_msg_method:
        Logging method to be used. Arguments: msg (str),
        prepend_timestamp (bool), prepend_blank_space (bool). If no method
        is specified, no logging is performed.

    Raises
    ------
    ValueError:
        If units is not None but unit_conversion_factors is None
    ValueError:
        If writing to a file fails.
    
    """

    def log_msg(msg='', prepend_timestamp=True, prepend_blank_space=True):
        """
        Use the logging method if it has been specified
        """
        if log_msg_method:
            log_msg_method(msg, prepend_timestamp, prepend_blank_space)

    if filepath is None:
        log_msg('Preparing data ...', prepend_timestamp=False)

    else:
        log_msg(f'Saving data to {filepath}...', prepend_timestamp=False)

    if data is not None:

        # make sure we do not modify the original data
        data = data.copy()

        # convert units and add unit information, if needed
        if units is not None:

            if unit_conversion_factors is None:
                raise ValueError(
                    'When units is not None, unit_conversion_factors must be provided')

            log_msg('Converting units...', prepend_timestamp=False)

            # if the orientation is 1, we might not need to scale all columns
            if orientation == 1:
                cols_to_scale = [dt in [float, int] for dt in data.dtypes]
                cols_to_scale = data.columns[cols_to_scale]

            labels_to_keep = []

            for unit_name in units.unique():

                labels = units.loc[units == unit_name].index.values

                unit_factor = 1./unit_conversion_factors[unit_name]

                active_labels = []

                if orientation == 0:
                    for label in labels:
                        if label in data.columns:
                            active_labels.append(label)

                    if len(active_labels) > 0:
                        data.loc[:, active_labels] *= unit_factor

                else:  # elif orientation == 1:
                    for label in labels:
                        if label in data.index:
                            active_labels.append(label)

                    if len(active_labels) > 0:
                        data.loc[active_labels, cols_to_scale] *= unit_factor

                labels_to_keep += active_labels

            units = units.loc[labels_to_keep].to_frame()

            if orientation == 0:
                data = pd.concat([units.T, data], axis=0)
                data.sort_index(axis=1, inplace=True)
            else:
                data = pd.concat([units, data], axis=1)
                data.sort_index(inplace=True)

            log_msg('Unit conversion successful.', prepend_timestamp=False)

        if use_simpleindex:
            # convert MultiIndex to regular index with '-' separators
            if isinstance(data.index, pd.MultiIndex):
                data = base.convert_to_SimpleIndex(data)

            # same thing for the columns
            if isinstance(data.columns, pd.MultiIndex):
                data = base.convert_to_SimpleIndex(data, axis=1)

        if filepath is not None:

            filepath = Path(filepath).resolve()
            if filepath.suffix == '.csv':

                # save the contents of the DataFrame into a csv
                data.to_csv(filepath)

                log_msg('Data successfully saved to file.',
                             prepend_timestamp=False)

            else:
                raise ValueError(
                    f'ERROR: Unexpected file type received when trying '
                    f'to save to csv: {filepath}')

            return None

        # at this line, filepath is None
        return data

    # at this line, data is None
    log_msg('WARNING: Data was empty, no file saved.',
            prepend_timestamp=False)
    return None


def load_data(data_source, unit_conversion_factors,
              orientation=0, reindex=True, return_units=False,
              convert=None, log_msg_method=None):
    """
    Loads data assuming it follows standard SimCenter tabular schema.

    The data is assumed to have a single header line and an index column. The
    second line may start with 'Units' in the index and provide the units for
    each column in the file.

    Parameters
    ----------
    data_source: string or DataFrame
        If it is a string, the data_source is assumed to point to the location
        of the source file. If it is a DataFrame, the data_source is assumed to
        hold the raw data.
    unit_conversion_factors: dict
        Dictionary containing key-value pairs of unit names and their
        corresponding factors. Conversion factors are defined as the number of
        times a base unit fits in the alternative unit.
    orientation: int, {0, 1}, default: 0
        If 0, variables are organized along columns; otherwise they are along
        the rows. This is important when converting values to follow the
        prescribed units.
    reindex: bool
        If True, reindexes the table to ensure a 0-based, continuous index
    return_units: bool
        If True, returns the units as well as the data to allow for adjustments
        in unit conversion.
    convert: list of string
        Specifies the columns (or rows if orientation==1) where unit conversion
        needs to be applied.
    log_msg_method:
        Logging method to be used. Arguments: msg (str),
        prepend_timestamp (bool), prepend_blank_space (bool). If no method
        is specified, no logging is performed.

    Returns
    -------
    data: DataFrame
        Parsed data.
    units: Series
        Labels from the data and corresponding units specified. If no units
        are specified, this return value is "None". units are only returned if
        return_units is set to True.
    """

    def log_msg(msg='', prepend_timestamp=True, prepend_blank_space=True):
        """
        Use the logging method if it has been specified
        """
        if log_msg_method:
            log_msg_method(msg, prepend_timestamp, prepend_blank_space)

    # if the provided data_source is already a DataFrame...
    if isinstance(data_source, pd.DataFrame):

        # we can just store it at proceed
        # (copying is needed to avoid changing the original)
        data = data_source.copy()

    else:
        # otherwise, load the data from a file
        data = load_from_file(data_source)

    # if there is information about units, perform the conversion to SI
    if ('Units' in data.index) or ('Units' in data.columns):

        log_msg('Converting units...', prepend_timestamp=False)

        if orientation == 0:
            units = data.loc['Units', :].copy().dropna()
            data.drop('Units', inplace=True)
            data = data.astype(float)

        else:  # elif orientation==1:
            units = data.loc[:, 'Units'].copy().dropna()
            data.drop('Units', axis=1, inplace=True)

            if convert is None:
                cols_to_scale = []
                for col in data.columns:
                    try:
                        data.loc[:, col] = data.loc[:, col].astype(float)
                        cols_to_scale.append(col)
                    except:
                        pass
            else:
                cols_to_scale = convert

        unique_unit_names = units.unique()

        for unit_name in unique_unit_names:

            unit_factor = unit_conversion_factors[unit_name]
            unit_labels = units.loc[units == unit_name].index

            if orientation == 0:
                data.loc[:, unit_labels] *= unit_factor

            else:  # elif orientation==1:
                data.loc[unit_labels, cols_to_scale] *= unit_factor

        log_msg('Unit conversion successful.', prepend_timestamp=False)

    else:

        # data = data.convert_dtypes()
        # enforcing float datatype is important even if there is no unit
        # conversion
        units = None
        if orientation == 0:
            data = data.astype(float)

        else:
            for col in data.columns:
                try:
                    data.loc[:, col] = data.loc[:, col].astype(float)
                except:
                    pass

    # convert column to MultiIndex if needed
    data = base.convert_to_MultiIndex(data, axis=1)

    data.sort_index(axis=1, inplace=True)

    # reindex the data, if needed
    if reindex:

        data.index = np.arange(data.shape[0])

    else:
        # convert index to MultiIndex if needed
        data = base.convert_to_MultiIndex(data, axis=0)

        data.sort_index(inplace=True)

    log_msg('Data successfully loaded from file.', prepend_timestamp=False)

    if return_units:

        # convert index in units Series to MultiIndex if needed
        units = base.convert_to_MultiIndex(units, axis=0)

        units.sort_index(inplace=True)

        return data, units

    # return_units=False
    return data


def load_from_file(filepath, log_msg_method=None):
    """
    Loads data from a file and stores it in a DataFrame.

    Currently, only CSV files are supported, but the function is easily
    extensible to support other file formats.

    Parameters
    ----------
    filepath: string
        The location of the source file

    Returns
    -------
    data: DataFrame
        Data loaded from the file.
    log_msg_method:
        Logging method to be used. Arguments: msg (str),
        prepend_timestamp (bool), prepend_blank_space (bool). If no method
        is specified, no logging is performed.
    """

    def log_msg(msg='', prepend_timestamp=True, prepend_blank_space=True):
        """
        Use the logging method if it has been specified
        """
        if log_msg_method:
            log_msg_method(msg, prepend_timestamp, prepend_blank_space)

    log_msg(f'Loading data from {filepath}...')

    # check if the filepath is valid
    filepath = Path(filepath).resolve()

    if not filepath.is_file():
        raise ValueError(f"The filepath provided does not point to an existing "
                         f"file: {filepath}")

    if filepath.suffix == '.csv':

        # load the contents of the csv into a DataFrame

        data = pd.read_csv(filepath, header=0, index_col=0, low_memory=False,
                           encoding_errors='replace')

        log_msg('File successfully opened.', prepend_timestamp=False)

    else:
        raise ValueError(f'ERROR: Unexpected file type received when trying '
                         f'to load from csv: {filepath}')

    return data

def parse_units(additional_file=None):
    """
    Parse the unit conversion factor json file and return a dictionary.
    
    Parameters
    ----------
    additional_file: str, optional
        If an additional file is provided, the function loads the
        default unit conversion factors and then overrides definitions
        using the additional user-specified json file.

    Raises
    ------
    KeyError:
        If a key is defined twice in any parsed json file.
    ValueError:
        If a unit conversion factor is not a float.
    FileNotFoundError:
        If a file does not exist.
    Exception:
        If a file does not have the json format.
    """

    def add_unique_keys_only(key, value, dictionary, file_path):
        if key not in dictionary:
            dictionary[key] = value
        else:
            raise KeyError(f'Unit {key} is defined twice in {file_path}\n'
               'Please check the file and make sure '
               'that unit conversion factors are only defined once.')

    def get_contents(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dictionary = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                'settings/default_units.json was not found.') from exc
        except json.decoder.JSONDecodeError as exc:
            raise Exception(
                'settings/default_units.json is not a valid json file.') from exc
        for key, val in dictionary.items():
            try:
                dictionary[key] = float(val)
            except ValueError as exc:
                raise ValueError(
                    f'Unit {key} has a value of {val} '
                    'which cannot be interpreted as a float') from exc
            
            
        return dictionary

    ucf_init = get_contents(base.pelicun_path / "settings/default_units.json")

    ucf = {}
    for key, value in ucf_init.items():
        add_unique_keys_only(key, value, ucf, 'settings/default_units.json')

    if additional_file:
        ucf_init = get_contents(additional_file)
        ucf_additional = {}
        for key, value in ucf_init.items():
            add_unique_keys_only(
                key, value, ucf_additional, additional_file)
        # combine the two: additional file takes precedent.
        # i.e., redefining default units with the additional ones is allowed
        for key, value in ucf_additional.items():
            ucf[key] = value

    return ucf
