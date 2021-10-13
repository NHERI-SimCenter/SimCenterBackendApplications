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
# Kuanshi Zhong
#

import numpy as np
import pandas as pd
import os
import getpass
import logging
from os.path import getsize
import sys
import shutil
import stat
import subprocess
import time
import importlib

install_requires = []
default_oq_version = '3.12.0'


def openquake_config(site_info, scen_info, event_info, dir_info):

    dir_input = dir_info['Input']
    dir_output = dir_info['Output']
    import configparser
    cfg = configparser.ConfigParser()
    # general section
    if scen_info['EqRupture']['Type'] == 'OpenQuakeScenario':
        cfg['general'] = {'description': 'Scenario Hazard Config File',
                          'calculation_mode': 'scenario'}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
        cfg['general'] = {'description': 'Scenario Hazard Config File',
                          'calculation_mode': 'event_based',
                          'ses_seed': scen_info['EqRupture'].get('Seed', 24)}
        cfg['logic_tree'] = {'number_of_logic_tree_samples': 0}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeClassicalPSHA':
        cfg['general'] = {'description': 'Scenario Hazard Config File',
                          'calculation_mode': 'classical',
                          'random_seed': scen_info['EqRupture'].get('Seed', 24)}
        cfg['logic_tree'] = {'number_of_logic_tree_samples': 0} # 0 here indicates full logic tree realization
    else:
        print('FetchOpenQuake: please specify Scenario[\'Generator\'], options: OpenQuakeScenario, OpenQuakeEventBased or OpenQuakeClassicalPSHA.')
        return 0
    # sites
    tmpSites = pd.read_csv(os.path.join(dir_input, site_info['input_file']), header=0, index_col=0)
    tmpSitesLoc = tmpSites.loc[:, ['Longitude','Latitude']]
    tmpSitesLoc.loc[site_info['min_ID']:site_info['max_ID']].to_csv(os.path.join(dir_input, 'sites_oq.csv'), header=False, index=False)
    cfg['geometry'] = {'sites_csv': 'sites_oq.csv'}
    # rupture
    cfg['erf'] = {'rupture_mesh_spacing': scen_info['EqRupture'].get('RupMesh', 2.0), 
                  'width_of_mfd_bin': scen_info['EqRupture'].get('MagFreqDistBin', 0.1),
                  'area_source_discretization': scen_info['EqRupture'].get('AreaMesh', 10.0)}
    # site_params (saved in the output_file)
    cfg['site_params'] = {'site_model_file': site_info['output_file']}
    # hazard_calculation
    mapGMPE = {'Abrahamson, Silva & Kamai (2014)': 'AbrahamsonEtAl2014',
               'AbrahamsonEtAl2014': 'AbrahamsonEtAl2014',
               'Boore, Stewart, Seyhan & Atkinson (2014)': 'BooreEtAl2014',
               'BooreEtAl2014': 'BooreEtAl2014',
               'Campbell & Bozorgnia (2014)': 'CampbellBozorgnia2014',
               'CampbellBozorgnia2014': 'CampbellBozorgnia2014',
               'Chiou & Youngs (2014)': 'ChiouYoungs2014',
               'ChiouYoungs2014': 'ChiouYoungs2014'
               }
    
    if scen_info['EqRupture']['Type'] == 'OpenQuakeScenario':
        imt = ''
        if event_info['IntensityMeasure']['Type'] == 'SA':
            for curT in event_info['IntensityMeasure']['Periods']:
                imt = imt + 'SA(' + str(curT) + '), '
            imt = imt[:-2]
        else:
            imt = event_info['IntensityMeasure']['Type']
        cfg['calculation'] = {'rupture_model_file': scen_info['EqRupture']['Filename'], 
                              'gsim': mapGMPE[event_info['GMPE']['Type']],
                              'intensity_measure_types': imt, 
                              'random_seed': 42, 
                              'truncation_level': event_info['IntensityMeasure'].get('Truncation', 3.0), 
                              'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
                              'number_of_ground_motion_fields': event_info['NumberPerSite']}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
        imt = ''
        imt_levels = event_info['IntensityMeasure'].get('Levels', [0.01,10,100])
        imt_scale = event_info['IntensityMeasure'].get('Scale', 'Log')
        if event_info['IntensityMeasure']['Type'] == 'SA':
            for curT in event_info['IntensityMeasure']['Periods']:
                #imt = imt + '"SA(' + str(curT) + ')": {}, '.format(imt_levels)
                if imt_scale == 'Log':
                    imt = imt + '"SA(' + str(curT) + ')": logscale({}, {}, {}), '.format(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                else:
                    imt_values = np.linspace(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                    imt_strings = ''
                    for imt_v in imt_values:
                        imt_strings = imt_strings+str(imt_v)+', '
                    imt_strings = imt_strings[:-2]
                    imt = imt + '"SA(' + str(curT) + ')": [{}], '.format(imt_strings)
            imt = imt[:-2]
        elif event_info['IntensityMeasure']['Type'] == 'PGA':
            if imt_scale == 'Log':
                imt = '"PGA": logscale({}, {}, {}), '.format(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
            else:
                imt_values = np.linspace(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                imt_strings = ''
                for imt_v in imt_values:
                    imt_strings = imt_strings+str(imt_v)+', '
                imt_strings = imt_strings[:-2]
                imt = 'PGA": [{}], '.format(imt_strings)
        else:
            imt = event_info['IntensityMeasure']['Type'] + ': logscale(1, 200, 45)'
        cfg['calculation'] = {'source_model_logic_tree_file': scen_info['EqRupture']['Filename'],
                              'gsim_logic_tree_file': event_info['GMPE']['Parameters'],
                              'investigation_time': scen_info['EqRupture']['TimeSpan'],
                              'intensity_measure_types_and_levels': '{' + imt + '}', 
                              'random_seed': 42, 
                              'truncation_level': event_info['IntensityMeasure'].get('Truncation', 3.0), 
                              'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
                              'number_of_ground_motion_fields': event_info['NumberPerSite']}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeClassicalPSHA':
        imt = ''
        imt_levels = event_info['IntensityMeasure'].get('Levels', [0.01,10,100])
        imt_scale = event_info['IntensityMeasure'].get('Scale', 'Log')
        if event_info['IntensityMeasure']['Type'] == 'SA':
            for curT in event_info['IntensityMeasure']['Periods']:
                #imt = imt + '"SA(' + str(curT) + ')": {}, '.format(imt_levels)
                if imt_scale == 'Log':
                    imt = imt + '"SA(' + str(curT) + ')": logscale({}, {}, {}), '.format(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                else:
                    imt_values = np.linspace(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                    imt_strings = ''
                    for imt_v in imt_values:
                        imt_strings = imt_strings+str(imt_v)+', '
                    imt_strings = imt_strings[:-2]
                    imt = imt + '"SA(' + str(curT) + ')": [{}], '.format(imt_strings)
            imt = imt[:-2]
        elif event_info['IntensityMeasure']['Type'] == 'PGA':
            if imt_scale == 'Log':
                imt = '"PGA": logscale({}, {}, {}), '.format(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
            else:
                imt_values = np.linspace(float(imt_levels[0]),float(imt_levels[1]),int(imt_levels[2]))
                imt_strings = ''
                for imt_v in imt_values:
                    imt_strings = imt_strings+str(imt_v)+', '
                imt_strings = imt_strings[:-2]
                imt = '"PGA": [{}], '.format(imt_strings)
        else:
            imt = event_info['IntensityMeasure']['Type'] + ': logscale(1, 200, 45)'
        cfg['calculation'] = {'source_model_logic_tree_file': scen_info['EqRupture']['Filename'],
                              'gsim_logic_tree_file': event_info['GMPE']['Parameters'],
                              'investigation_time': scen_info['EqRupture']['TimeSpan'],
                              'intensity_measure_types_and_levels': '{' + imt + '}', 
                              'truncation_level': event_info['IntensityMeasure'].get('Truncation', 3.0), 
                              'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0)}
        cfg_quan = ''
        cfg['output'] = {'export_dir': dir_output,
                         'individual_curves': scen_info['EqRupture'].get('IndivHazCurv', False), 
                         'mean': scen_info['EqRupture'].get('MeanHazCurv', True),
                         'quantiles': ' '.join([str(x) for x in scen_info['EqRupture'].get('Quantiles', [0.05, 0.5, 0.95])]),
                         'hazard_maps': scen_info['EqRupture'].get('HazMap', False),
                         'uniform_hazard_spectra': scen_info['EqRupture'].get('UHS', False),
                         'poes': np.round(1-np.exp(-float(scen_info['EqRupture']['TimeSpan'])*1.0/float(scen_info['EqRupture'].get('ReturnPeriod', 100))),decimals=3)}
    else:
        print('FetchOpenQuake: please specify Scenario[\'Generator\'], options: OpenQuakeScenario or OpenQuakeEventBased.')
        return 0
    # Write the ini
    filename_ini = os.path.join(dir_input, 'oq_job.ini')
    with open(filename_ini, 'w') as configfile:
        cfg.write(configfile)

    # openquake module
    oq_ver_loaded = None
    from importlib.metadata import version
    if scen_info['EqRupture'].get('OQLocal',None):
        # using user-specific local OQ
        # first to validate the path
        if not os.path.isdir(scen_info['EqRupture'].get('OQLocal')):
            print('FetchOpenQuake: Local OpenQuake instance {} not found.'.format(scen_info['EqRupture'].get('OQLocal')))
            return 0
        else:
            # getting version
            try:
                oq_ver = version('openquake.engine')
                if oq_ver:
                    print('FetchOpenQuake: Removing previous installation of OpenQuake {}.'.format(oq_ver))
                    sys.modules.pop('openquake')
                    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "openquake.engine"])
            except:
                # no installed OQ python package
                # do nothing
                print('FetchOpenQuake: No previous installation of OpenQuake python package found.')
            # load the local OQ
            try:
                print('FetchOpenQuake: Setting up the user-specified local OQ.')
                sys.path.insert(0,os.path.dirname(scen_info['EqRupture'].get('OQLocal')))
                #owd = os.getcwd()
                #os.chdir(os.path.dirname(scen_info['EqRupture'].get('OQLocal')))
                if 'openquake' in list(sys.modules.keys()):
                    sys.modules.pop('openquake')
                from openquake import baselib
                oq_ver_loaded = baselib.__version__
                #sys.modules.pop('openquake')
                #os.chdir(owd)
            except:
                print('FetchOpenQuake: {} cannot be loaded.'.format(scen_info['EqRupture'].get('OQLocal')))

    else:
        # using the offical released OQ
        try:
            oq_ver = version('openquake.engine')
            if oq_ver != scen_info['EqRupture'].get('OQVersion',default_oq_version):
                print('FetchOpenQuake: Required OpenQuake version is not found and being installed now.')
                if oq_ver:
                    # pop the old version first
                    sys.modules.pop('openquake')
                    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "openquake.engine"])
                
                # install the required version
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openquake.engine=="+scen_info['EqRupture'].get('OQVersion',default_oq_version), "--user"])
                oq_ver_loaded = version('openquake.engine')
                
            else:
                oq_ver_loaded = oq_ver

        except:
            print('FetchOpenQuake: No OpenQuake is not found and being installed now.')
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "openquake.engine=="+scen_info['EqRupture'].get('OQVersion',default_oq_version), "--user"])
                oq_ver_loaded = version('openquake.engine')
            except:
                print('FetchOpenQuake: Install of OpenQuake {} failed - please check the version.'.format(scen_info['EqRupture'].get('OQVersion',default_oq_version)))

    print('FetchOpenQuake: OpenQuake configured.')

    # return
    return filename_ini, oq_ver_loaded


def oq_run_classical_psha(job_ini, exports='csv', oq_version=default_oq_version):
    """
    Run a classical PSHA by OpenQuake

    :param job_ini:
        Path to configuration file/archive or
        dictionary of parameters with at least a key "calculation_mode"
    """    
    # the run() method has been turned into private since v3.11
    # the get_last_calc_id() and get_datadir() have been moved to commonlib.logs since v3.12
    # the datastore has been moved to commonlib since v3.12
    # Note: the extracting realizations method was kindly shared by Dr. Anne Husley
    vtag = int(oq_version.split('.')[1])
    if vtag <= 10:
        try:
            print('FetchOpenQuake: running Version {}.'.format(oq_version))
            # reloading 
            from openquake.commands.run import run
            from openquake.baselib import datastore
            from openquake.calculators.export.hazard import export_realizations

            # initialize database db.sqlite3 (version-sensitive data log)
            path_sqlite3 = os.path.join(datastore.get_datadir(),'db.sqlite3')
            if os.path.isfile(path_sqlite3):
                # removing the previous data log
                try:
                    shutil.rmtree(datastore.get_datadir())
                except:
                    print('FetchOpenQuake: cannot remove {}'.format(datastore.get_datadir()))
                    return 1

            run([job_ini], exports=exports)
            calc_id = datastore.get_last_calc_id()
            path = os.path.join(datastore.get_datadir(), 'calc_%d.hdf5' % calc_id)
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
            return 0
        except:
            print('FetchOpenQuake: Classical PSHA failed.')
            return 1
    elif vtag == 11:
        try:
            print('FetchOpenQuake: running Version {}.'.format(oq_version))
            # reloading 
            from openquake.commands import run
            from openquake.baselib import datastore
            from openquake.calculators.export.hazard import export_realizations
            
            path_sqlite3 = os.path.join(datastore.get_datadir(),'db.sqlite3')
            if os.path.isfile(path_sqlite3):
                # removing the previous data log
                try:
                    shutil.rmtree(datastore.get_datadir())
                except:
                    print('FetchOpenQuake: cannot remove {}'.format(datastore.get_datadir()))
                    return 1

            run.main([job_ini], exports=exports)
            calc_id = datastore.get_last_calc_id()
            path = os.path.join(datastore.get_datadir(), 'calc_%d.hdf5' % calc_id)
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
            return 0
        except:
            print('FetchOpenQuake: Classical PSHA failed.')
            return 1
    else:
        try:
            print('FetchOpenQuake: running Version {}.'.format(oq_version))
            # reloading 
            from openquake.commands import run
            from openquake.commonlib import logs, datastore
            from openquake.calculators.export.hazard import export_realizations

            path_sqlite3 = os.path.join(datastore.get_datadir(),'db.sqlite3')
            if os.path.isfile(path_sqlite3):
                # removing the previous data log
                try:
                    shutil.rmtree(datastore.get_datadir())
                except:
                    print('FetchOpenQuake: cannot remove {}'.format(datastore.get_datadir()))
                    return 1

            run.main([job_ini], exports=exports)
            calc_id = logs.get_last_calc_id()
            path = os.path.join(logs.get_datadir(), 'calc_%d.hdf5' % calc_id)
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
            return 0
        except:
            print('FetchOpenQuake: Classical PSHA failed.')
        


def oq_read_uhs_classical_psha(scen_info, event_info, dir_info):
    """
    Collect the UHS from a classical PSHA by OpenQuake
    """
    import glob
    import random
    # number of scenario
    num_scen = scen_info['Number']
    if num_scen > 1:
        print('FetchOpenQuake: currently only supporting a single scenario for PHSA')
        num_scen = 1
    # number of realizations per site
    num_rlz = event_info['NumberPerSite']
    # directory of the UHS
    res_dir = dir_info['Output']
    # mean UHS
    cur_uhs_file = glob.glob(os.path.join(res_dir,'hazard_uhs-mean_*.csv'))[0]
    print(cur_uhs_file)
    # read csv
    tmp = pd.read_csv(cur_uhs_file,skiprows=1)
    # number of stations
    num_stn = len(tmp.index)
    # number of IMs
    num_IMs = len(tmp.columns) - 2
    # IM list
    list_IMs = tmp.columns.tolist()[2:]
    ln_psa_mr = []
    mag_maf = []
    for i in range(num_scen):
        # initialization
        ln_psa = np.zeros((num_stn, num_IMs, num_rlz))
        # collecting UHS
        if num_rlz == 1:
            ln_psa[:, :, 0] = np.log(tmp.iloc[:, 2:])
        else:
            num_r1 = np.min([len(glob.glob(os.path.join(res_dir,'hazard_uhs-rlz-*.csv'))), num_rlz])
            for i in range(num_r1):
                cur_uhs_file = glob.glob(os.path.join(res_dir,'hazard_uhs-rlz-*.csv'))[i]
                tmp = pd.read_csv(cur_uhs_file,skiprows=1)
                ln_psa[:, :, i] = np.log(tmp.iloc[:, 2:])
            if num_rlz > num_r1:
                # randomly resampling available spectra
                for i in range(num_rlz-num_r1):
                    rnd_tag = random.randrange(num_r1)
                    print(int(rnd_tag))
                    cur_uhs_file = glob.glob(os.path.join(res_dir,'hazard_uhs-rlz-*.csv'))[int(rnd_tag)]
                    tmp = pd.read_csv(cur_uhs_file,skiprows=1)
                    ln_psa[:, :, i] = np.log(tmp.iloc[:, 2:])
        ln_psa_mr.append(ln_psa)
        mag_maf.append([0.0,float(list_IMs[0].split('~')[0]),0.0])
    
    # return
    return ln_psa_mr, mag_maf
   

class OpenQuakeHazardCalc:

    def __init__(self, job_ini, event_info, oq_version, no_distribute=False):
        """
        Initialize a calculation (reinvented from openquake.engine.engine)

        :param job_ini:
            Path to configuration file/archive or
            dictionary of parameters with at least a key "calculation_mode"
        """

        self.vtag = int(oq_version.split('.')[1])

        from openquake.baselib import config, performance, general, zeromq, hdf5, parallel
        from openquake.hazardlib import const, calc, gsim
        from openquake import commonlib
        from openquake.commonlib import readinput, logictree, logs
        if self.vtag >= 12:
            from openquake.commonlib import datastore
        else:
            from openquake.baselib import datastore
        from openquake.calculators import base
        from openquake.server import dbserver
        from openquake.commands import dbserver as cdbs

        user_name = getpass.getuser()

        if no_distribute:
            os.environ['OQ_DISTRIBUTE'] = 'no'

        # check if the datadir exists
        datadir = datastore.get_datadir()
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        #dbserver.ensure_on()
        if dbserver.get_status() == 'not-running':
            if config.dbserver.multi_user:
                sys.exit('Please start the DbServer: '
                        'see the documentation for details')
            # otherwise start the DbServer automatically; NB: I tried to use
            # multiprocessing.Process(target=run_server).start() and apparently
            # it works, but then run-demos.sh hangs after the end of the first
            # calculation, but only if the DbServer is started by oq engine (!?)
            # Here is a trick to activate OpenQuake's dbserver
            # We first cd to the openquake directory and invoke subprocess to open/hold on dbserver
            # Then, we cd back to the original working directory 
            owd = os.getcwd()
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            self.prc = subprocess.Popen([sys.executable, '-m', 'openquake.commands', 'dbserver', 'start'])
            os.chdir(owd)

            # wait for the dbserver to start
            waiting_seconds = 30
            while dbserver.get_status() == 'not-running':
                if waiting_seconds == 0:
                    sys.exit('The DbServer cannot be started after 30 seconds. '
                            'Please check the configuration')
                time.sleep(1)
                waiting_seconds -= 1

        # check if we are talking to the right server
        err = dbserver.check_foreign()
        if err:
            sys.exit(err)

        # Copy the event_info
        self.event_info = event_info

        # Create a job
        #self.job = logs.init("job", job_ini, logging.INFO, None, None, None)
        if self.vtag >= 11:
            dic = readinput.get_params(job_ini)
        else:
            dic = readinput.get_params([job_ini])
        #dic['hazard_calculation_id'] = self.job.calc_id

        if self.vtag >= 12:
            # Create the job log
            self.log = logs.init('job', dic, logging.INFO, None, None, None)
            # Get openquake parameters
            self.oqparam = self.log.get_oqparam()
            self.calculator = base.calculators(self.oqparam, self.log.calc_id)
        else:
            # Create the job log
            self.calc_id = logs.init('job', logging.INFO)
            # Get openquake parameters
            self.oqparam = readinput.get_oqparam(dic)
            self.calculator = base.calculators(self.oqparam, self.calc_id)

        # Create the calculator
        self.calculator.from_engine = True

        print('FetchOpenQuake: OpenQuake Hazard Calculator initiated.')

    def run_calc(self):
        """
        Run a calculation and return results (reinvented from openquake.calculators.base)
        """

        from openquake.calculators import base, getters
        from openquake.baselib import config, performance, zeromq
        if self.vtag >= 11:
            from openquake.baselib import version
        else:
            from openquake.baselib import __version__ as version

        with self.calculator._monitor:
            self.calculator._monitor.username = ''
            try:
                # Pre-execute setups
                self.calculator.pre_execute()

                #self.calculator.datastore.swmr_on()
                oq = self.calculator.oqparam
                dstore = self.calculator.datastore
                self.calculator.set_param()
                self.calculator.offset = 0

                # Source model
                #print('self.__dict__ = ')
                #print(self.calculator.__dict__)
                if oq.hazard_calculation_id:  # from ruptures
                    dstore.parent = self.calculator.datastore.read(
                        oq.hazard_calculation_id)
                elif hasattr(self.calculator, 'csm'):  # from sources
                    self.calculator_build_events_from_sources()
                    #self.calculator.build_events_from_sources()
                    if (oq.ground_motion_fields is False and oq.hazard_curves_from_gmfs is False):
                        return {}
                elif 'rupture_model' not in oq.inputs:
                    logging.warning(
                        'There is no rupture_model, the calculator will just '
                        'import data without performing any calculation')
                    fake = logictree.FullLogicTree.fake()
                    dstore['full_lt'] = fake  # needed to expose the outputs
                    dstore['weights'] = [1.]
                    return {}
                else:  # scenario
                    self.calculator._read_scenario_ruptures()
                    if (oq.ground_motion_fields is False and oq.hazard_curves_from_gmfs is False):
                        return {}

                # Intensity measure models
                if oq.ground_motion_fields:
                    if self.vtag >= 12:
                        imts = oq.get_primary_imtls()
                        nrups = len(dstore['ruptures'])
                        base.create_gmf_data(dstore, imts, oq.get_sec_imts())
                        dstore.create_dset('gmf_data/sigma_epsilon',
                                        getters.sig_eps_dt(oq.imtls))
                        dstore.create_dset('gmf_data/time_by_rup',
                                        getters.time_dt, (nrups,), fillvalue=None)
                    elif self.vtag == 11:
                        imts = oq.get_primary_imtls()
                        nrups = len(dstore['ruptures'])
                        base.create_gmf_data(dstore, len(imts), oq.get_sec_imts())
                        dstore.create_dset('gmf_data/sigma_epsilon',
                                        getters.sig_eps_dt(oq.imtls))
                        dstore.create_dset('gmf_data/time_by_rup',
                                        getters.time_dt, (nrups,), fillvalue=None)
                    else:
                        pass

                # Prepare inputs for GmfGetter
                nr = len(dstore['ruptures'])
                logging.info('Reading {:_d} ruptures'.format(nr))
                if self.vtag >= 12:
                    rgetters = getters.get_rupture_getters(dstore, oq.concurrent_tasks * 1.25,
                                                        srcfilter=self.calculator.srcfilter)
                elif self.vtag == 11:
                    rgetters = getters.gen_rupture_getters(dstore, oq.concurrent_tasks)
                else:
                    rgetters = getters.gen_rupture_getters(dstore, self.calculator.srcfilter, oq.concurrent_tasks)

                
                args = [(rgetter, self.calculator.param) for rgetter in rgetters]
                mon = performance.Monitor()
                mon.version = version
                mon.config = config
                rcvr = 'tcp://%s:%s' % (config.dbserver.listen,
                                        config.dbserver.receiver_ports)
                skt = zeromq.Socket(rcvr, zeromq.zmq.PULL, 'bind').__enter__()
                mon.backurl = 'tcp://%s:%s' % (config.dbserver.host, skt.port)
                mon = mon.new(
                    operation='total ' + self.calculator.core_task.__func__.__name__, measuremem=True)
                mon.weight = getattr(args[0], 'weight', 1.)  # used in task_info
                mon.task_no = 1  # initialize the task number
                args += (mon,)

                self.args = args
                self.mon = mon
                self.dstore = dstore

            finally:
                print('FetchOpenQuake: OpenQuake Hazard Calculator defined.')
                # parallel.Starmap.shutdown()

    def eval_calc(self):
        """
        Evaluate each calculators for different IMs
        """

        # Define the GmfGetter

        #for args_tag in range(len(self.args)-1):
            # Looping over all source models (Note: the last attribute in self.args is a monitor - so skipping it)
        
        from openquake.calculators import getters
        from openquake.baselib import general
        from openquake.hazardlib import const, calc, gsim
        from openquake.commands import dbserver as cdbs
        if self.vtag >= 12:
            from openquake.hazardlib.const import StdDev

        cur_getter = getters.GmfGetter(self.args[0][0], calc.filters.SourceFilter(
            self.dstore['sitecol'], self.dstore['oqparam'].maximum_distance), 
            self.calculator.param['oqparam'], self.calculator.param['amplifier'], 
            self.calculator.param['sec_perils'])

        # Evaluate each computer
        print('FetchOpenQuake: Evaluting ground motion models.')
        for computer in cur_getter.gen_computers(self.mon):
            # Looping over rupture(s) in the current realization
            sids = computer.sids
            #print('eval_calc: site ID sids = ')
            #print(sids)
            eids_by_rlz = computer.ebrupture.get_eids_by_rlz(
                cur_getter.rlzs_by_gsim)
            mag = computer.ebrupture.rupture.mag
            data = general.AccumDict(accum=[])
            cur_T = self.event_info['IntensityMeasure'].get('Periods', None)
            for cur_gs, rlzs in cur_getter.rlzs_by_gsim.items():
                # Looping over GMPE(s)
                #print('eval_calc: cur_gs = ')
                #print(cur_gs)
                num_events = sum(len(eids_by_rlz[rlz]) for rlz in rlzs)
                if num_events == 0:  # it may happen
                    continue
                    # NB: the trick for performance is to keep the call to
                    # .compute outside of the loop over the realizations;
                    # it is better to have few calls producing big arrays
                tmpMean = []
                tmpstdtot = []
                tmpstdinter = []
                tmpstdintra = []
                if self.vtag >= 12:
                    mean_stds_all = computer.cmaker.get_mean_stds([computer.ctx], StdDev.EVENT)[0]
                for imti, imt in enumerate(computer.imts): 
                    # Looping over IM(s)
                    #print('eval_calc: imt = ', imt)
                    if str(imt) in ['PGA', 'PGV', 'PGD']:
                        cur_T = [0.0]
                        imTag = 'ln' + str(imt)
                    else:
                        imTag = 'lnSA'
                    if isinstance(cur_gs, gsim.multi.MultiGMPE):
                        gs = cur_gs[str(imt)]  # MultiGMPE
                    else:
                        gs = cur_gs  # regular GMPE
                    try:
                        if self.vtag >= 12:
                            mean_stds = mean_stds_all[:, imti]
                            num_sids = len(computer.sids)
                            num_stds = len(mean_stds)
                            if num_stds == 1:
                                # no standard deviation is available
                                # for truncation_level = 0 there is only mean, no stds
                                if computer.correlation_model:
                                    raise ValueError('truncation_level=0 requires '
                                                    'no correlation model')
                                mean = mean_stds[0]
                                stddev_intra = 0
                                stddev_inter = 0
                                stddev_total = 0
                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = np.concatenate((tmpstdtot, stddev_total), axis=0)
                            elif num_stds == 2:
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                # By default, we evaluate stddev_inter as the stddev_total

                                if self.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(
                                        self.correlation_model, gsim)

                                mean, stddev_total = mean_stds
                                stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))
                                stddev_inter = stddev_total
                                stddev_intra = 0
                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = np.concatenate((tmpstdtot, stddev_total), axis=0)
                            else:
                                mean, stddev_inter, stddev_intra = mean_stds
                                stddev_intra = stddev_intra.reshape(stddev_intra.shape + (1, ))
                                stddev_inter = stddev_inter.reshape(stddev_inter.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))
                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = np.concatenate((tmpstdtot,np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)), axis=1)

                        elif self.vtag == 11:
                            # v11
                            dctx = computer.dctx.roundup(
                                cur_gs.minimum_distance)
                            if computer.distribution is None:
                                if computer.correlation_model:
                                    raise ValueError('truncation_level=0 requires '
                                                    'no correlation model')
                                mean, _stddevs = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt, stddev_types=[])
                            num_sids = len(computer.sids)
                            if cur_gs.DEFINED_FOR_STANDARD_DEVIATION_TYPES == {const.StdDev.TOTAL}:
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                if computer.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(
                                        computer.correlation_model, cur_gs)

                                mean, [stddev_total] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt, [const.StdDev.TOTAL])
                                stddev_total = stddev_total.reshape(
                                    stddev_total.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))
                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)
                                    tmpstdtot = np.concatenate((tmpstdtot, stddev_total), axis=0)
                            else:
                                mean, [stddev_inter, stddev_intra] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt,
                                    [const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT])
                                stddev_intra = stddev_intra.reshape(
                                    stddev_intra.shape + (1, ))
                                stddev_inter = stddev_inter.reshape(
                                    stddev_inter.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))

                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = np.concatenate((tmpstdtot,np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)), axis=1)

                        else:
                            # v10
                            dctx = computer.dctx.roundup(
                                cur_gs.minimum_distance)
                            if computer.truncation_level == 0:
                                if computer.correlation_model:
                                    raise ValueError('truncation_level=0 requires '
                                                    'no correlation model')
                                mean, _stddevs = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt, stddev_types=[])
                            num_sids = len(computer.sids)
                            if cur_gs.DEFINED_FOR_STANDARD_DEVIATION_TYPES == {const.StdDev.TOTAL}:
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                if computer.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(
                                        computer.correlation_model, cur_gs)

                                mean, [stddev_total] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt, [const.StdDev.TOTAL])
                                stddev_total = stddev_total.reshape(
                                    stddev_total.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))
                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)
                                    tmpstdtot = np.concatenate((tmpstdtot, stddev_total), axis=0)
                            else:
                                mean, [stddev_inter, stddev_intra] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx, computer.rctx, dctx, imt,
                                    [const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT])
                                stddev_intra = stddev_intra.reshape(
                                    stddev_intra.shape + (1, ))
                                stddev_inter = stddev_inter.reshape(
                                    stddev_inter.shape + (1, ))
                                mean = mean.reshape(mean.shape + (1, ))

                                if imti == 0:
                                    tmpMean = mean
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)
                                    tmpstdinter = np.concatenate((tmpstdinter, stddev_inter), axis=1)
                                    tmpstdintra = np.concatenate((tmpstdintra, stddev_intra), axis=1)
                                    tmpstdtot = np.concatenate((tmpstdtot,np.sqrt(stddev_inter * stddev_inter + stddev_intra * stddev_intra)), axis=1)

                    except Exception as exc:
                        raise RuntimeError(
                            '(%s, %s, source_id=%r) %s: %s' %
                            (gs, imt, computer.source_id.decode('utf8'),
                            exc.__class__.__name__, exc)
                        ).with_traceback(exc.__traceback__)

                # initialize
                # NOTE: needs to be extended for gmpe logic tree
                gm_collector = []
                # collect data
                for k in range(tmpMean.shape[0]):
                    imResult = {}
                    if len(tmpMean):
                        imResult.update({'Mean': [float(x) for x in tmpMean[k].tolist()]})
                    if len(tmpstdtot):
                        imResult.update({'TotalStdDev':  [float(x) for x in tmpstdtot[k].tolist()]})
                    if len(tmpstdinter):
                        imResult.update({'InterEvStdDev':  [float(x) for x in tmpstdinter[k].tolist()]})
                    if len(tmpstdintra):
                        imResult.update({'IntraEvStdDev':  [float(x) for x in tmpstdintra[k].tolist()]})
                    gm_collector.append({imTag: imResult})
                #print(gm_collector)
        
        # close datastore instance
        self.calculator.datastore.close()
        
        # stop dbserver
        if self.vtag >= 11:
            cdbs.main('stop')
        else:
            cdbs.dbserver('stop')
        
        # terminate the subprocess
        self.prc.kill()

        # Final results
        res = {'Magnitude': mag,
               'Periods': cur_T,
               'GroundMotions': gm_collector}
        
        # return
        return res

    def calculator_build_events_from_sources(self):
        """
        Prefilter the composite source model and store the source_info
        """
        gsims_by_trt = self.calculator.csm.full_lt.get_gsims_by_trt()
        print('FetchOpenQuake: self.calculator.csm.src_groups = ')
        print(self.calculator.csm.src_groups)
        sources = self.calculator.csm.get_sources()
        print('FetchOpenQuake: sources = ')
        print(sources)
        for src in sources:
            src.nsites = 1  # avoid 0 weight
            src.num_ruptures = src.count_ruptures()
        maxweight = sum(sg.weight for sg in self.calculator.csm.src_groups) / (
            self.calculator.oqparam.concurrent_tasks or 1)
        print('FetchOpenQuake: weights = ')
        print([sg.weight for sg in self.calculator.csm.src_groups])
        print('FetchOpenQuake: maxweight = ')
        print(maxweight)
        eff_ruptures = general.AccumDict(accum=0)  # trt => potential ruptures
        calc_times = general.AccumDict(accum=np.zeros(3, np.float32))  # nr, ns, dt
        allargs = []
        if self.calculator.oqparam.is_ucerf():
            # manage the filtering in a special way
            for sg in self.calculator.csm.src_groups:
                for src in sg:
                    src.src_filter = self.calculator.srcfilter
            srcfilter = calc.filters.nofilter  # otherwise it would be ultra-slow
        else:
            srcfilter = self.calculator.srcfilter
        logging.info('Building ruptures')
        for sg in self.calculator.csm.src_groups:
            if not sg.sources:
                continue
            logging.info('Sending %s', sg)
            par = self.calculator.param.copy()
            par['gsims'] = gsims_by_trt[sg.trt]
            for src_group in sg.split(maxweight):
                allargs.append((src_group, srcfilter, par))

        smap = []        
        for curargs in allargs:
            smap.append(calc.stochastic.sample_ruptures(curargs[0], curargs[1], curargs[2]))

        print('smap = ')
        print(smap)
        self.calculator.nruptures = 0
        mon = self.calculator.monitor('saving ruptures')
        for tmp in smap:
            dic = next(tmp)
            print(dic)
            # NB: dic should be a dictionary, but when the calculation dies
            # for an OOM it can become None, thus giving a very confusing error
            if dic is None:
                raise MemoryError('You ran out of memory!')
            rup_array = dic['rup_array']
            if len(rup_array) == 0:
                continue
            if dic['calc_times']:
                calc_times += dic['calc_times']
            if dic['eff_ruptures']:
                eff_ruptures += dic['eff_ruptures']
            with mon:
                n = len(rup_array)
                rup_array['id'] = np.arange(
                    self.calculator.nruptures, self.calculator.nruptures + n)
                self.calculator.nruptures += n
                hdf5.extend(self.calculator.datastore['ruptures'], rup_array)
                hdf5.extend(self.calculator.datastore['rupgeoms'], rup_array.geom)

        if len(self.calculator.datastore['ruptures']) == 0:
            raise RuntimeError('No ruptures were generated, perhaps the '
                               'investigation time is too short')

        # must be called before storing the events
        self.calculator.store_rlz_info(eff_ruptures)  # store full_lt
        self.calculator.store_source_info(calc_times)
        imp = commonlib.calc.RuptureImporter(self.calculator.datastore)
        print('self.calculator.datastore.getitem(ruptures)')
        print(self.calculator.datastore.getitem('ruptures'))
        with self.calculator.monitor('saving ruptures and events'):
            imp.import_rups_events(self.calculator.datastore.getitem('ruptures')[()], getters.get_rupture_getters)



class CorrelationButNoInterIntraStdDevs(Exception):
    def __init__(self, corr, gsim):
        self.corr = corr
        self.gsim = gsim

    def __str__(self):
        return '''\
        You cannot use the correlation model %s with the GSIM %s, \
        that defines only the total standard deviation. If you want to use a \
        correlation model you have to select a GMPE that provides the inter and \
        intra event standard deviations.''' % (
            self.corr.__class__.__name__, self.gsim.__class__.__name__)


def to_imt_unit_values(vals, imt):
    """
    Exponentiate the values unless the IMT is MMI
    """
    if str(imt) == 'MMI':
        return vals
    return np.exp(vals)
