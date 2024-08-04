#  # noqa: INP001, D100
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

import getpass
import logging
import os
import shutil
import socket
import subprocess  # noqa: S404
import sys
import time

import numpy as np
import pandas as pd

install_requires = []
default_oq_version = '3.17.1'


def openquake_config(site_info, scen_info, event_info, workDir):  # noqa: ANN001, ANN201, C901, N803, D103, PLR0912, PLR0915
    dir_input = os.path.join(workDir, 'Input')  # noqa: PTH118
    dir_output = os.path.join(workDir, 'Output')  # noqa: PTH118
    import configparser  # noqa: PLC0415

    cfg = configparser.ConfigParser()
    # general section
    if scen_info['EqRupture']['Type'] == 'oqSourceXML':  # OpenQuakeScenario
        cfg['general'] = {
            'description': 'Scenario Hazard Config File',
            'calculation_mode': 'scenario',
        }
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
        cfg['general'] = {
            'description': 'Scenario Hazard Config File',
            'calculation_mode': 'event_based',
            'ses_seed': scen_info['EqRupture'].get('Seed', 24),
        }
        cfg['logic_tree'] = {'number_of_logic_tree_samples': 0}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeClassicalPSHA':
        cfg['general'] = {
            'description': 'Scenario Hazard Config File',
            'calculation_mode': 'classical',
            'random_seed': scen_info['EqRupture'].get('Seed', 24),
        }
        cfg['logic_tree'] = {
            'number_of_logic_tree_samples': 0
        }  # 0 here indicates full logic tree realization
    elif scen_info['EqRupture']['Type'] in [  # noqa: PLR6201
        'OpenQuakeUserConfig',
        'OpenQuakeClassicalPSHA-User',
    ]:
        filename_ini = scen_info['EqRupture'].get('ConfigFile', None)
        if filename_ini is None:
            print(  # noqa: T201
                "FetchOpenQuake: please specify Scenario['EqRupture']['ConfigFile']."
            )
            return 0
        else:  # noqa: RET505
            filename_ini = os.path.join(dir_input, filename_ini)  # noqa: PTH118
            # updating the export_dir
            cfg.read(filename_ini)
            cfg['output']['export_dir'] = dir_output
    else:
        print(  # noqa: T201
            "FetchOpenQuake: please specify Scenario['Generator'], options: OpenQuakeScenario, OpenQuakeEventBased, OpenQuakeClassicalPSHA, or OpenQuakeUserConfig."
        )
        return 0

    if scen_info['EqRupture']['Type'] in [  # noqa: PLR1702, PLR6201
        'OpenQuakeUserConfig',
        'OpenQuakeClassicalPSHA-User',
    ]:
        # sites
        tmpSites = pd.read_csv(  # noqa: N806
            os.path.join(dir_input, site_info['input_file']),  # noqa: PTH118
            header=0,
            index_col=0,
        )
        tmpSitesLoc = tmpSites.loc[:, ['Longitude', 'Latitude']]  # noqa: N806
        tmpSitesLoc.loc[site_info['min_ID'] : site_info['max_ID']].to_csv(
            os.path.join(dir_input, 'sites_oq.csv'),  # noqa: PTH118
            header=False,
            index=False,
        )
        if cfg.has_section('geometry'):
            cfg['geometry']['sites_csv'] = 'sites_oq.csv'
        else:
            cfg['geometry'] = {'sites_csv': 'sites_oq.csv'}
        if cfg.has_section('site_params'):
            cfg['site_params']['site_model_file'] = site_info['output_file']
        else:
            cfg['site_params'] = {'site_model_file': site_info['output_file']}
        # copy that file to the rundir
        shutil.copy(
            os.path.join(dir_input, site_info['output_file']),  # noqa: PTH118
            os.path.join(dir_output, site_info['output_file']),  # noqa: PTH118
        )

        # im type and period
        tmp0 = (
            cfg['calculation'].get('intensity_measure_types_and_levels').split('"')
        )
        tmp = []
        for jj, cur_tmp in enumerate(tmp0):
            if jj % 2:
                tmp.append(cur_tmp)
        im_type = []
        tmp_T = []  # noqa: N806
        for cur_tmp in tmp:
            if 'PGA' in cur_tmp:
                im_type = 'PGA'
            elif 'SA' in cur_tmp:
                im_type = 'SA'
                tmp_T.append(float(cur_tmp.split('(')[-1].split(')')[0]))
            else:
                pass
        event_info['IntensityMeasure']['Type'] = im_type
        event_info['IntensityMeasure']['Periods'] = tmp_T
        cfg['calculation']['source_model_logic_tree_file'] = os.path.join(  # noqa: PTH118
            cfg['calculation'].get('source_model_logic_tree_file')
        )
        cfg['calculation']['gsim_logic_tree_file'] = os.path.join(  # noqa: PTH118
            cfg['calculation'].get('gsim_logic_tree_file')
        )
    else:
        # sites
        # tmpSites = pd.read_csv(site_info['siteFile'], header=0, index_col=0)
        # tmpSitesLoc = tmpSites.loc[:, ['Longitude','Latitude']]
        # tmpSitesLoc.to_csv(os.path.join(dir_input, 'sites_oq.csv'), header=False, index=False)
        # cfg['geometry'] = {'sites_csv': 'sites_oq.csv'}
        cfg['geometry'] = {'sites_csv': os.path.basename(site_info['siteFile'])}  # noqa: PTH119
        # rupture
        cfg['erf'] = {
            'rupture_mesh_spacing': scen_info['EqRupture'].get('RupMesh', 2.0),
            'width_of_mfd_bin': scen_info['EqRupture'].get('MagFreqDistBin', 0.1),
            'area_source_discretization': scen_info['EqRupture'].get(
                'AreaMesh', 10.0
            ),
        }
        # site_params (saved in the output_file)
        cfg['site_params'] = {'site_model_file': 'tmp_oq_site_model.csv'}
        # hazard_calculation
        mapGMPE = {  # noqa: N806
            'Abrahamson, Silva & Kamai (2014)': 'AbrahamsonEtAl2014',
            'AbrahamsonEtAl2014': 'AbrahamsonEtAl2014',
            'Boore, Stewart, Seyhan & Atkinson (2014)': 'BooreEtAl2014',
            'BooreEtAl2014': 'BooreEtAl2014',
            'Campbell & Bozorgnia (2014)': 'CampbellBozorgnia2014',
            'CampbellBozorgnia2014': 'CampbellBozorgnia2014',
            'Chiou & Youngs (2014)': 'ChiouYoungs2014',
            'ChiouYoungs2014': 'ChiouYoungs2014',
        }

        if scen_info['EqRupture']['Type'] == 'oqSourceXML':  # OpenQuakeScenario
            imt = ''
            if event_info['IntensityMeasure']['Type'] == 'SA':
                for curT in event_info['IntensityMeasure']['Periods']:  # noqa: N806
                    imt = imt + 'SA(' + str(curT) + '), '
                imt = imt[:-2]
            else:
                imt = event_info['IntensityMeasure']['Type']
            cfg['calculation'] = {
                'rupture_model_file': scen_info['EqRupture']['sourceFile'],
                'gsim': mapGMPE[event_info['GMPE']['Type']],
                'intensity_measure_types': imt,
                'random_seed': 42,
                'truncation_level': event_info['IntensityMeasure'].get(
                    'Truncation', 3.0
                ),
                'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
                'number_of_ground_motion_fields': event_info['NumberPerSite'],
            }
        elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
            imt = ''
            imt_levels = event_info['IntensityMeasure'].get(
                'Levels', [0.01, 10, 100]
            )
            imt_scale = event_info['IntensityMeasure'].get('Scale', 'Log')
            if event_info['IntensityMeasure']['Type'] == 'SA':
                for curT in event_info['IntensityMeasure']['Periods']:  # noqa: N806
                    # imt = imt + '"SA(' + str(curT) + ')": {}, '.format(imt_levels)
                    if imt_scale == 'Log':
                        imt = (
                            imt
                            + '"SA('
                            + str(curT)
                            + f')": logscale({float(imt_levels[0])}, {float(imt_levels[1])}, {int(imt_levels[2])}), '
                        )
                    else:
                        imt_values = np.linspace(
                            float(imt_levels[0]),
                            float(imt_levels[1]),
                            int(imt_levels[2]),
                        )
                        imt_strings = ''
                        for imt_v in imt_values:
                            imt_strings = imt_strings + str(imt_v) + ', '
                        imt_strings = imt_strings[:-2]
                        imt = imt + '"SA(' + str(curT) + f')": [{imt_strings}], '
                imt = imt[:-2]
            elif event_info['IntensityMeasure']['Type'] == 'PGA':
                if imt_scale == 'Log':
                    imt = f'"PGA": logscale({float(imt_levels[0])}, {float(imt_levels[1])}, {int(imt_levels[2])}), '
                else:
                    imt_values = np.linspace(
                        float(imt_levels[0]),
                        float(imt_levels[1]),
                        int(imt_levels[2]),
                    )
                    imt_strings = ''
                    for imt_v in imt_values:
                        imt_strings = imt_strings + str(imt_v) + ', '
                    imt_strings = imt_strings[:-2]
                    imt = f'PGA": [{imt_strings}], '
            else:
                imt = (
                    event_info['IntensityMeasure']['Type'] + ': logscale(1, 200, 45)'
                )
            cfg['calculation'] = {
                'source_model_logic_tree_file': scen_info['EqRupture']['Filename'],
                'gsim_logic_tree_file': event_info['GMPE']['Parameters'],
                'investigation_time': scen_info['EqRupture']['TimeSpan'],
                'intensity_measure_types_and_levels': '{' + imt + '}',
                'random_seed': 42,
                'truncation_level': event_info['IntensityMeasure'].get(
                    'Truncation', 3.0
                ),
                'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
                'number_of_ground_motion_fields': event_info['NumberPerSite'],
            }
        elif scen_info['EqRupture']['Type'] == 'OpenQuakeClassicalPSHA':
            imt = ''
            imt_levels = event_info['IntensityMeasure'].get(
                'Levels', [0.01, 10, 100]
            )
            imt_scale = event_info['IntensityMeasure'].get('Scale', 'Log')
            if event_info['IntensityMeasure']['Type'] == 'SA':
                for curT in event_info['IntensityMeasure']['Periods']:  # noqa: N806
                    # imt = imt + '"SA(' + str(curT) + ')": {}, '.format(imt_levels)
                    if imt_scale == 'Log':
                        imt = (
                            imt
                            + '"SA('
                            + str(curT)
                            + f')": logscale({float(imt_levels[0])}, {float(imt_levels[1])}, {int(imt_levels[2])}), '
                        )
                    else:
                        imt_values = np.linspace(
                            float(imt_levels[0]),
                            float(imt_levels[1]),
                            int(imt_levels[2]),
                        )
                        imt_strings = ''
                        for imt_v in imt_values:
                            imt_strings = imt_strings + str(imt_v) + ', '
                        imt_strings = imt_strings[:-2]
                        imt = imt + '"SA(' + str(curT) + f')": [{imt_strings}], '
                imt = imt[:-2]
            elif event_info['IntensityMeasure']['Type'] == 'PGA':
                if imt_scale == 'Log':
                    imt = f'"PGA": logscale({float(imt_levels[0])}, {float(imt_levels[1])}, {int(imt_levels[2])}), '
                else:
                    imt_values = np.linspace(
                        float(imt_levels[0]),
                        float(imt_levels[1]),
                        int(imt_levels[2]),
                    )
                    imt_strings = ''
                    for imt_v in imt_values:
                        imt_strings = imt_strings + str(imt_v) + ', '
                    imt_strings = imt_strings[:-2]
                    imt = f'"PGA": [{imt_strings}], '
            else:
                imt = (
                    event_info['IntensityMeasure']['Type'] + ': logscale(1, 200, 45)'
                )
            cfg['calculation'] = {
                'source_model_logic_tree_file': scen_info['EqRupture']['Filename'],
                'gsim_logic_tree_file': event_info['GMPE']['Parameters'],
                'investigation_time': scen_info['EqRupture']['TimeSpan'],
                'intensity_measure_types_and_levels': '{' + imt + '}',
                'truncation_level': event_info['IntensityMeasure'].get(
                    'Truncation', 3.0
                ),
                'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
            }
            cfg_quan = ''  # noqa: F841
            cfg['output'] = {
                'export_dir': dir_output,
                'individual_curves': scen_info['EqRupture'].get(
                    'IndivHazCurv', False
                ),
                'mean': scen_info['EqRupture'].get('MeanHazCurv', True),
                'quantiles': ' '.join(
                    [
                        str(x)
                        for x in scen_info['EqRupture'].get(
                            'Quantiles', [0.05, 0.5, 0.95]
                        )
                    ]
                ),
                'hazard_maps': scen_info['EqRupture'].get('HazMap', False),
                'uniform_hazard_spectra': scen_info['EqRupture'].get('UHS', False),
                'poes': np.round(
                    1
                    - np.exp(
                        -float(scen_info['EqRupture']['TimeSpan'])
                        * 1.0
                        / float(scen_info['EqRupture'].get('ReturnPeriod', 100))
                    ),
                    decimals=3,
                ),
            }
        else:
            print(  # noqa: T201
                "FetchOpenQuake: please specify Scenario['Generator'], options: OpenQuakeScenario, OpenQuakeEventBased, OpenQuakeClassicalPSHA, or OpenQuakeUserConfig."
            )
            return 0

    # Write the ini
    filename_ini = os.path.join(dir_input, 'oq_job.ini')  # noqa: PTH118
    with open(filename_ini, 'w') as configfile:  # noqa: PLW1514, PTH123
        cfg.write(configfile)

    # openquake module
    oq_ver_loaded = None
    try:
        from importlib_metadata import version  # noqa: PLC0415
    except:  # noqa: E722
        from importlib.metadata import version  # noqa: PLC0415
    if scen_info['EqRupture'].get('OQLocal', None):
        # using user-specific local OQ
        # first to validate the path
        if not os.path.isdir(scen_info['EqRupture'].get('OQLocal')):  # noqa: PTH112
            print(  # noqa: T201
                'FetchOpenQuake: Local OpenQuake instance {} not found.'.format(
                    scen_info['EqRupture'].get('OQLocal')
                )
            )
            return 0
        else:  # noqa: RET505
            # getting version
            try:
                oq_ver = version('openquake.engine')
                if oq_ver:
                    print(  # noqa: T201
                        f'FetchOpenQuake: Removing previous installation of OpenQuake {oq_ver}.'
                    )
                    sys.modules.pop('openquake')
                    subprocess.check_call(  # noqa: S603
                        [
                            sys.executable,
                            '-m',
                            'pip',
                            'uninstall',
                            '-y',
                            'openquake.engine',
                        ]
                    )
            except:  # noqa: E722
                # no installed OQ python package
                # do nothing
                print(  # noqa: T201
                    'FetchOpenQuake: No previous installation of OpenQuake python package found.'
                )
            # load the local OQ
            try:
                print('FetchOpenQuake: Setting up the user-specified local OQ.')  # noqa: T201
                sys.path.insert(
                    0,
                    os.path.dirname(scen_info['EqRupture'].get('OQLocal')),  # noqa: PTH120
                )
                # owd = os.getcwd()
                # os.chdir(os.path.dirname(scen_info['EqRupture'].get('OQLocal')))
                if 'openquake' in list(sys.modules.keys()):
                    sys.modules.pop('openquake')
                from openquake import baselib  # noqa: PLC0415

                oq_ver_loaded = baselib.__version__
                # sys.modules.pop('openquake')
                # os.chdir(owd)
            except:  # noqa: E722
                print(  # noqa: T201
                    'FetchOpenQuake: {} cannot be loaded.'.format(
                        scen_info['EqRupture'].get('OQLocal')
                    )
                )

    else:
        # using the official released OQ
        try:
            oq_ver = version('openquake.engine')
            if oq_ver != scen_info['EqRupture'].get('OQVersion', default_oq_version):
                print(  # noqa: T201
                    'FetchOpenQuake: Required OpenQuake version is not found and being installed now.'
                )
                if oq_ver:
                    # pop the old version first
                    sys.modules.pop('openquake')
                    subprocess.check_call(  # noqa: S603
                        [
                            sys.executable,
                            '-m',
                            'pip',
                            'uninstall',
                            '-y',
                            'openquake.engine',
                        ]
                    )

                # install the required version
                subprocess.check_call(  # noqa: S603
                    [
                        sys.executable,
                        '-m',
                        'pip',
                        'install',
                        'openquake.engine=='
                        + scen_info['EqRupture'].get(
                            'OQVersion', default_oq_version
                        ),
                        '--user',
                    ]
                )
                oq_ver_loaded = version('openquake.engine')

            else:
                oq_ver_loaded = oq_ver

        except:  # noqa: E722
            print(  # noqa: T201
                'FetchOpenQuake: No OpenQuake is not found and being installed now.'
            )
            try:
                subprocess.check_call(  # noqa: S603
                    [
                        sys.executable,
                        '-m',
                        'pip',
                        'install',
                        'openquake.engine=='
                        + scen_info['EqRupture'].get(
                            'OQVersion', default_oq_version
                        ),
                        '--user',
                    ]
                )
                oq_ver_loaded = version('openquake.engine')
            except:  # noqa: E722
                print(  # noqa: T201
                    'FetchOpenQuake: Install of OpenQuake {} failed - please check the version.'.format(
                        scen_info['EqRupture'].get('OQVersion', default_oq_version)
                    )
                )

    print('FetchOpenQuake: OpenQuake configured.')  # noqa: T201

    # return
    return filename_ini, oq_ver_loaded, event_info


# this function writes a openquake.cfg for setting global configurations
# tested while not used so far but might be useful in future if moving to
# other os...
"""
def get_cfg(job_ini):
    # writing an openquake.cfg
    import configparser
    cfg = configparser.ConfigParser()
    # distribution
    cfg['distribution'] = {"oq_distribute": "processpool",
                           "serialize_jobs": 1,
                           "log_level": "info"}
    # memory
    cfg['memory'] = {"limit": "1_000_000_000_000",
                     "soft_mem_limit": 90,
                     "hard_mem_limit": 99}
    # amqp
    cfg['amqp'] = {"host": "localhost",
                   "port": 5672,
                   "user": "openquake",
                   "password": "openquake",
                   "vhost": "openquake",
                   "celery_queue": "celery"}
    # dbserver
    cfg['dbserver'] = {"multi_user": "false",
                       "file": os.path.join(os.environ.get('OQ_DATADIR'),'db.sqlit3').replace('\\','/'),
                       "listen": "127.0.0.1",
                       "host": "localhost",
                       "port": 1908,
                       "receiver_ports": "1912-1920",
                       "authkey": "changeme"}
    # webapi
    cfg['webapi'] = {"server": "http://localhost:8800",
                     "username": "",
                     "password": ""}
    # zworkers
    cfg['zworkers'] = {"host_cores": "127.0.0.1 -1",
                       "ctrl_port": 1909,
                       "remote_python": "",
                       "remote_user": ""}
    # directory
    cfg['directory'] = {"shared_dir": "",
                        "custom_tmp": ""}
    # path
    oq_cfg = os.path.join(os.path.dirname(job_ini),'openquake.cfg')
    with open(oq_cfg, 'w') as configfile:
        cfg.write(configfile)
    # return
    return oq_cfg
"""


def oq_run_classical_psha(  # noqa: ANN201, C901
    job_ini,  # noqa: ANN001
    exports='csv',  # noqa: ANN001
    oq_version=default_oq_version,  # noqa: ANN001
    dir_info=None,  # noqa: ANN001
):
    """Run a classical PSHA by OpenQuake

    :param job_ini:
        Path to configuration file/archive or
        dictionary of parameters with at least a key "calculation_mode"
    """  # noqa: D400
    # the run() method has been turned into private since v3.11
    # the get_last_calc_id() and get_datadir() have been moved to commonlib.logs since v3.12
    # the datastore has been moved to commonlib since v3.12
    # Note: the extracting realizations method was kindly shared by Dr. Anne Husley
    vtag = int(oq_version.split('.')[1])
    if vtag <= 10:  # noqa: PLR2004
        try:
            print(f'FetchOpenQuake: running Version {oq_version}.')  # noqa: T201
            # reloading
            # run.main([job_ini], exports=exports)
            # invoke/modify deeper openquake commands here to make it compatible with
            # the pylauncher on stampede2 for parallel runs...
            from openquake.baselib import (  # noqa: PLC0415
                datastore,
                general,
                performance,
            )
            from openquake.calculators import base  # noqa: PLC0415
            from openquake.calculators.export.hazard import (  # noqa: PLC2701, PLC0415, RUF100
                export_realizations,
            )
            from openquake.commonlib import logs, readinput  # noqa: PLC0415
            from openquake.server import dbserver  # noqa: PLC0415

            dbserver.ensure_on()
            global calc_path
            loglevel = 'info'
            params = {}
            reuse_input = False
            concurrent_tasks = None
            pdb = None
            hc_id = None
            for i in range(1000):  # noqa: B007
                try:
                    calc_id = logs.init('nojob', getattr(logging, loglevel.upper()))
                except:  # noqa: PERF203, E722
                    time.sleep(0.01)
                    continue
                else:
                    print('FetchOpenQuake: log created.')  # noqa: T201
                    break
            # disable gzip_input
            base.BaseCalculator.gzip_inputs = lambda self: None  # noqa: ARG005
            with performance.Monitor('total runtime', measuremem=True) as monitor:
                if os.environ.get('OQ_DISTRIBUTE') not in ('no', 'processpool'):  # noqa: PLR6201
                    os.environ['OQ_DISTRIBUTE'] = 'processpool'
                oqparam = readinput.get_oqparam(job_ini, hc_id=hc_id)
                if hc_id and hc_id < 0:  # interpret negative calculation ids
                    calc_ids = datastore.get_calc_ids()
                    try:
                        hc_id = calc_ids[hc_id]
                    except IndexError:
                        raise SystemExit(  # noqa: B904, DOC501
                            'There are %d old calculations, cannot '
                            'retrieve the %s' % (len(calc_ids), hc_id)
                        )
                calc = base.calculators(oqparam, calc_id)
                calc.run(
                    concurrent_tasks=concurrent_tasks,
                    pdb=pdb,
                    exports=exports,
                    hazard_calculation_id=hc_id,
                    rlz_ids=(),
                )

            calc_id = datastore.get_last_calc_id()
            path = os.path.join(datastore.get_datadir(), 'calc_%d.hdf5' % calc_id)  # noqa: PTH118
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
        except:  # noqa: E722
            print('FetchOpenQuake: Classical PSHA failed.')  # noqa: T201
            return 1
    elif vtag == 11:  # noqa: PLR2004
        try:
            print(f'FetchOpenQuake: running Version {oq_version}.')  # noqa: T201
            # reloading
            # run.main([job_ini], exports=exports)
            # invoke/modify deeper openquake commands here to make it compatible with
            # the pylauncher on stampede2 for parallel runs...
            from openquake.baselib import (  # noqa: PLC0415
                datastore,
                general,
                performance,
            )
            from openquake.calculators import base  # noqa: PLC0415
            from openquake.calculators.export.hazard import (  # noqa: PLC0415
                export_realizations,
            )
            from openquake.commonlib import logs, readinput  # noqa: PLC0415
            from openquake.server import dbserver  # noqa: PLC0415

            dbserver.ensure_on()
            global calc_path
            loglevel = 'info'
            params = {}
            reuse_input = False
            concurrent_tasks = None
            pdb = False
            for i in range(1000):  # noqa: B007
                try:
                    calc_id = logs.init('nojob', getattr(logging, loglevel.upper()))
                except:  # noqa: PERF203, E722
                    time.sleep(0.01)
                    continue
                else:
                    print('FetchOpenQuake: log created.')  # noqa: T201
                    break
            # disable gzip_input
            base.BaseCalculator.gzip_inputs = lambda self: None  # noqa: ARG005
            with performance.Monitor('total runtime', measuremem=True) as monitor:
                if os.environ.get('OQ_DISTRIBUTE') not in ('no', 'processpool'):  # noqa: PLR6201
                    os.environ['OQ_DISTRIBUTE'] = 'processpool'
                if 'hazard_calculation_id' in params:
                    hc_id = int(params['hazard_calculation_id'])
                else:
                    hc_id = None
                if hc_id and hc_id < 0:  # interpret negative calculation ids
                    calc_ids = datastore.get_calc_ids()
                    try:
                        params['hazard_calculation_id'] = str(calc_ids[hc_id])
                    except IndexError:
                        raise SystemExit(  # noqa: B904, DOC501
                            'There are %d old calculations, cannot '
                            'retrieve the %s' % (len(calc_ids), hc_id)
                        )
                oqparam = readinput.get_oqparam(job_ini, kw=params)
                calc = base.calculators(oqparam, calc_id)
                if reuse_input:  # enable caching
                    oqparam.cachedir = datastore.get_datadir()
                calc.run(concurrent_tasks=concurrent_tasks, pdb=pdb, exports=exports)

            calc_id = datastore.get_last_calc_id()
            path = os.path.join(datastore.get_datadir(), 'calc_%d.hdf5' % calc_id)  # noqa: PTH118
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
        except:  # noqa: E722
            print('FetchOpenQuake: Classical PSHA failed.')  # noqa: T201
            return 1
    else:
        try:
            print(f'FetchOpenQuake: running Version {oq_version}.')  # noqa: T201
            # reloading
            # run.main([job_ini], exports=exports)
            # invoke/modify deeper openquake commands here to make it compatible with
            # the pylauncher on stampede2 for parallel runs...
            from openquake.baselib import general, performance  # noqa: PLC0415
            from openquake.calculators import base  # noqa: PLC0415
            from openquake.calculators.export.hazard import (  # noqa: PLC0415
                export_realizations,
            )
            from openquake.commonlib import datastore, logs  # noqa: PLC0415
            from openquake.server import dbserver  # noqa: PLC0415

            dbserver.ensure_on()
            global calc_path  # noqa: PLW0602
            loglevel = 'info'
            params = {}
            reuse_input = False
            concurrent_tasks = None
            pdb = False
            for i in range(1000):  # noqa: B007
                try:
                    log = logs.init(
                        'job', job_ini, getattr(logging, loglevel.upper())
                    )
                except:  # noqa: PERF203, E722
                    time.sleep(0.01)
                    continue
                else:
                    print('FetchOpenQuake: log created.')  # noqa: T201
                    break
            log.params.update(params)
            base.BaseCalculator.gzip_inputs = lambda self: None  # noqa: ARG005
            with log, performance.Monitor(
                'total runtime', measuremem=True
            ) as monitor:
                calc = base.calculators(log.get_oqparam(), log.calc_id)
                if reuse_input:  # enable caching
                    calc.oqparam.cachedir = datastore.get_datadir()
                calc.run(concurrent_tasks=concurrent_tasks, pdb=pdb, exports=exports)

            logging.info('Total time spent: %s s', monitor.duration)
            logging.info('Memory allocated: %s', general.humansize(monitor.mem))
            print('See the output with silx view %s' % calc.datastore.filename)  # noqa: T201, UP031

            calc_id = logs.get_last_calc_id()
            path = os.path.join(logs.get_datadir(), 'calc_%d.hdf5' % calc_id)  # noqa: PTH118
            dstore = datastore.read(path)
            export_realizations('realizations', dstore)
        except:  # noqa: E722
            print('FetchOpenQuake: Classical PSHA failed.')  # noqa: T201
            return 1

    # h5 clear for stampede2 (this is somewhat inelegant...)
    if 'stampede2' in socket.gethostname():
        # h5clear
        if oq_h5clear(path) == 0:
            print('FetchOpenQuake.oq_run_classical_psha: h5clear completed')  # noqa: T201
        else:
            print('FetchOpenQuake.oq_run_classical_psha: h5clear failed')  # noqa: T201

    # copy the calc file to output directory
    if dir_info:
        dir_output = dir_info['Output']
        try:
            shutil.copy2(path, dir_output)
            print('FetchOpenQuake: calc hdf file saved.')  # noqa: T201
        except:  # noqa: E722
            print('FetchOpenQuake: failed to copy calc hdf file.')  # noqa: T201

    return 0


def oq_h5clear(hdf5_file):  # noqa: ANN001, ANN201, D103
    # h5clear = os.path.join(os.path.dirname(os.path.abspath(__file__)),'lib/hdf5/bin/h5clear')
    # print(h5clear)
    print(hdf5_file)  # noqa: T201
    # subprocess.run(["chmod", "a+rx", h5clear])
    subprocess.run(['chmod', 'a+rx', hdf5_file], check=False)  # noqa: S603, S607
    tmp = subprocess.run(['h5clear', '-s', hdf5_file], check=False)  # noqa: S603, S607
    print(tmp)  # noqa: T201
    run_flag = tmp.returncode
    return run_flag  # noqa: RET504


def oq_read_uhs_classical_psha(scen_info, event_info, dir_info):  # noqa: ANN001, ANN201
    """Collect the UHS from a classical PSHA by OpenQuake"""  # noqa: D400
    import glob  # noqa: PLC0415
    import random  # noqa: PLC0415

    # number of scenario
    num_scen = scen_info['Number']
    if num_scen > 1:
        print('FetchOpenQuake: currently only supporting a single scenario for PHSA')  # noqa: T201
        num_scen = 1
    # number of realizations per site
    num_rlz = event_info['NumberPerSite']
    # directory of the UHS
    res_dir = dir_info['Output']
    # mean UHS
    cur_uhs_file = glob.glob(os.path.join(res_dir, 'hazard_uhs-mean_*.csv'))[0]  # noqa: PTH118, PTH207
    print(cur_uhs_file)  # noqa: T201
    # read csv
    tmp = pd.read_csv(cur_uhs_file, skiprows=1)
    # number of stations
    num_stn = len(tmp.index)
    # number of IMs
    num_IMs = len(tmp.columns) - 2  # noqa: N806
    # IM list
    list_IMs = tmp.columns.tolist()[2:]  # noqa: N806
    im_list = [x.split('~')[1] for x in list_IMs]
    ln_psa_mr = []
    mag_maf = []
    for i in range(num_scen):
        # initialization
        ln_psa = np.zeros((num_stn, num_IMs, num_rlz))
        # collecting UHS
        if num_rlz == 1:
            ln_psa[:, :, 0] = np.log(tmp.iloc[:, 2:])
        else:
            num_r1 = np.min(
                [
                    len(glob.glob(os.path.join(res_dir, 'hazard_uhs-rlz-*.csv'))),  # noqa: PTH118, PTH207
                    num_rlz,
                ]
            )
            for i in range(num_r1):  # noqa: PLW2901
                cur_uhs_file = glob.glob(  # noqa: PTH207
                    os.path.join(res_dir, 'hazard_uhs-rlz-*.csv')  # noqa: PTH118
                )[i]
                tmp = pd.read_csv(cur_uhs_file, skiprows=1)
                ln_psa[:, :, i] = np.log(tmp.iloc[:, 2:])
            if num_rlz > num_r1:
                # randomly resampling available spectra
                for i in range(num_rlz - num_r1):  # noqa: PLW2901
                    rnd_tag = random.randrange(num_r1)
                    print(int(rnd_tag))  # noqa: T201
                    cur_uhs_file = glob.glob(  # noqa: PTH207
                        os.path.join(res_dir, 'hazard_uhs-rlz-*.csv')  # noqa: PTH118
                    )[int(rnd_tag)]
                    tmp = pd.read_csv(cur_uhs_file, skiprows=1)
                    ln_psa[:, :, i] = np.log(tmp.iloc[:, 2:])
        ln_psa_mr.append(ln_psa)
        mag_maf.append([0.0, float(list_IMs[0].split('~')[0]), 0.0])

    # return
    return ln_psa_mr, mag_maf, im_list


class OpenQuakeHazardCalc:  # noqa: D101
    def __init__(  # noqa: ANN204, C901
        self,
        job_ini,  # noqa: ANN001
        event_info,  # noqa: ANN001
        oq_version,  # noqa: ANN001
        dir_info=None,  # noqa: ANN001
        no_distribute=False,  # noqa: ANN001, FBT002
    ):
        """Initialize a calculation (reinvented from openquake.engine.engine)

        :param job_ini:
            Path to configuration file/archive or
            dictionary of parameters with at least a key "calculation_mode"
        """  # noqa: D400
        self.vtag = int(oq_version.split('.')[1])
        self.dir_info = dir_info

        from openquake.baselib import (  # noqa: PLC0415
            config,
        )
        from openquake.commonlib import logs, readinput  # noqa: PLC0415

        if self.vtag >= 12:  # noqa: PLR2004
            from openquake.commonlib import datastore  # noqa: PLC0415
        else:
            from openquake.baselib import datastore  # noqa: PLC0415
        from openquake.calculators import base  # noqa: PLC0415
        from openquake.server import dbserver  # noqa: PLC0415

        user_name = getpass.getuser()  # noqa: F841

        if no_distribute:
            os.environ['OQ_DISTRIBUTE'] = 'no'

        # check if the datadir exists
        datadir = datastore.get_datadir()
        if not os.path.exists(datadir):  # noqa: PTH110
            os.makedirs(datadir)  # noqa: PTH103

        # dbserver.ensure_on()
        if dbserver.get_status() == 'not-running':
            if config.dbserver.multi_user:
                sys.exit(
                    'Please start the DbServer: ' 'see the documentation for details'
                )
            # otherwise start the DbServer automatically; NB: I tried to use
            # multiprocessing.Process(target=run_server).start() and apparently
            # it works, but then run-demos.sh hangs after the end of the first
            # calculation, but only if the DbServer is started by oq engine (!?)
            # Here is a trick to activate OpenQuake's dbserver
            # We first cd to the openquake directory and invoke subprocess to open/hold on dbserver
            # Then, we cd back to the original working directory
            owd = os.getcwd()  # noqa: PTH109
            os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120
            self.prc = subprocess.Popen(  # noqa: S603
                [sys.executable, '-m', 'openquake.commands', 'dbserver', 'start']
            )
            os.chdir(owd)

            # wait for the dbserver to start
            waiting_seconds = 30
            while dbserver.get_status() == 'not-running':
                if waiting_seconds == 0:
                    sys.exit(
                        'The DbServer cannot be started after 30 seconds. '
                        'Please check the configuration'
                    )
                time.sleep(1)
                waiting_seconds -= 1
        else:
            self.prc = False

        # check if we are talking to the right server
        err = dbserver.check_foreign()
        if err:
            sys.exit(err)

        # Copy the event_info
        self.event_info = event_info

        # Create a job
        # self.job = logs.init("job", job_ini, logging.INFO, None, None, None)
        if self.vtag >= 11:  # noqa: PLR2004
            dic = readinput.get_params(job_ini)
        else:
            dic = readinput.get_params([job_ini])
        # dic['hazard_calculation_id'] = self.job.calc_id

        if self.vtag >= 12:  # noqa: PLR2004
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

        print('FetchOpenQuake: OpenQuake Hazard Calculator initiated.')  # noqa: T201

    def run_calc(self):  # noqa: ANN201, C901
        """Run a calculation and return results (reinvented from openquake.calculators.base)"""  # noqa: D400
        from openquake.baselib import config, performance, zeromq  # noqa: PLC0415
        from openquake.calculators import base, getters  # noqa: PLC0415

        if self.vtag >= 11:  # noqa: PLR2004
            from openquake.baselib import version  # noqa: PLC0415
        else:
            from openquake.baselib import __version__ as version  # noqa: PLC0415

        with self.calculator._monitor:  # noqa: SLF001
            self.calculator._monitor.username = ''  # noqa: SLF001
            try:
                # Pre-execute setups
                self.calculator.pre_execute()

                # self.calculator.datastore.swmr_on()
                oq = self.calculator.oqparam
                dstore = self.calculator.datastore
                self.calculator.set_param()
                self.calculator.offset = 0

                # Source model
                # print('self.__dict__ = ')
                # print(self.calculator.__dict__)
                if oq.hazard_calculation_id:  # from ruptures
                    dstore.parent = self.calculator.datastore.read(
                        oq.hazard_calculation_id
                    )
                elif hasattr(self.calculator, 'csm'):  # from sources
                    self.calculator_build_events_from_sources()
                    # self.calculator.build_events_from_sources()
                    if (
                        oq.ground_motion_fields is False
                        and oq.hazard_curves_from_gmfs is False
                    ):
                        return {}
                elif 'rupture_model' not in oq.inputs:
                    logging.warning(
                        'There is no rupture_model, the calculator will just '
                        'import data without performing any calculation'
                    )
                    fake = logictree.FullLogicTree.fake()  # noqa: F821
                    dstore['full_lt'] = fake  # needed to expose the outputs
                    dstore['weights'] = [1.0]
                    return {}
                else:  # scenario
                    self.calculator._read_scenario_ruptures()  # noqa: SLF001
                    if (
                        oq.ground_motion_fields is False
                        and oq.hazard_curves_from_gmfs is False
                    ):
                        return {}

                # Intensity measure models
                if oq.ground_motion_fields:
                    if self.vtag >= 12:  # noqa: PLR2004
                        imts = oq.get_primary_imtls()
                        nrups = len(dstore['ruptures'])
                        base.create_gmf_data(dstore, imts, oq.get_sec_imts())
                        dstore.create_dset(
                            'gmf_data/sigma_epsilon', getters.sig_eps_dt(oq.imtls)
                        )
                        dstore.create_dset(
                            'gmf_data/time_by_rup',
                            getters.time_dt,
                            (nrups,),
                            fillvalue=None,
                        )
                    elif self.vtag == 11:  # noqa: PLR2004
                        imts = oq.get_primary_imtls()
                        nrups = len(dstore['ruptures'])
                        base.create_gmf_data(dstore, len(imts), oq.get_sec_imts())
                        dstore.create_dset(
                            'gmf_data/sigma_epsilon', getters.sig_eps_dt(oq.imtls)
                        )
                        dstore.create_dset(
                            'gmf_data/time_by_rup',
                            getters.time_dt,
                            (nrups,),
                            fillvalue=None,
                        )
                    else:
                        pass

                # Prepare inputs for GmfGetter
                nr = len(dstore['ruptures'])
                logging.info(f'Reading {nr:_d} ruptures')
                if self.vtag >= 12:  # noqa: PLR2004
                    rgetters = getters.get_rupture_getters(
                        dstore,
                        oq.concurrent_tasks * 1.25,
                        srcfilter=self.calculator.srcfilter,
                    )
                elif self.vtag == 11:  # noqa: PLR2004
                    rgetters = getters.gen_rupture_getters(
                        dstore, oq.concurrent_tasks
                    )
                else:
                    rgetters = getters.gen_rupture_getters(
                        dstore, self.calculator.srcfilter, oq.concurrent_tasks
                    )

                args = [(rgetter, self.calculator.param) for rgetter in rgetters]
                mon = performance.Monitor()
                mon.version = version
                mon.config = config
                rcvr = 'tcp://%s:%s' % (  # noqa: UP031
                    config.dbserver.listen,
                    config.dbserver.receiver_ports,
                )
                skt = zeromq.Socket(rcvr, zeromq.zmq.PULL, 'bind').__enter__()  # noqa: PLC2801
                mon.backurl = 'tcp://%s:%s' % (config.dbserver.host, skt.port)  # noqa: UP031
                mon = mon.new(
                    operation='total ' + self.calculator.core_task.__func__.__name__,
                    measuremem=True,
                )
                mon.weight = getattr(args[0], 'weight', 1.0)  # used in task_info
                mon.task_no = 1  # initialize the task number
                args += (mon,)

                self.args = args
                self.mon = mon
                self.dstore = dstore

            finally:
                print('FetchOpenQuake: OpenQuake Hazard Calculator defined.')  # noqa: T201
                # parallel.Starmap.shutdown()

    def eval_calc(self):  # noqa: ANN201, C901, PLR0912, PLR0915
        """Evaluate each calculators for different IMs"""  # noqa: D400
        # Define the GmfGetter

        # for args_tag in range(len(self.args)-1):
        # Looping over all source models (Note: the last attribute in self.args is a monitor - so skipping it)

        from openquake.baselib import general  # noqa: PLC0415
        from openquake.calculators import getters  # noqa: PLC0415
        from openquake.commands import dbserver as cdbs  # noqa: PLC0415
        from openquake.hazardlib import calc, const, gsim  # noqa: PLC0415

        if self.vtag >= 12:  # noqa: PLR2004
            from openquake.hazardlib.const import StdDev  # noqa: PLC0415
        if self.vtag >= 12:  # noqa: PLR2004
            from openquake.commonlib import datastore  # noqa: PLC0415
        else:
            from openquake.baselib import datastore  # noqa: PLC0415

        cur_getter = getters.GmfGetter(
            self.args[0][0],
            calc.filters.SourceFilter(
                self.dstore['sitecol'], self.dstore['oqparam'].maximum_distance
            ),
            self.calculator.param['oqparam'],
            self.calculator.param['amplifier'],
            self.calculator.param['sec_perils'],
        )

        # Evaluate each computer
        print('FetchOpenQuake: Evaluating ground motion models.')  # noqa: T201
        for computer in cur_getter.gen_computers(self.mon):  # noqa: PLR1702
            # Looping over rupture(s) in the current realization
            sids = computer.sids  # noqa: F841
            # print('eval_calc: site ID sids = ')
            # print(sids)
            eids_by_rlz = computer.ebrupture.get_eids_by_rlz(cur_getter.rlzs_by_gsim)
            mag = computer.ebrupture.rupture.mag
            im_list = []
            data = general.AccumDict(accum=[])  # noqa: F841
            cur_T = self.event_info['IntensityMeasure'].get('Periods', None)  # noqa: N806
            for cur_gs, rlzs in cur_getter.rlzs_by_gsim.items():
                # Looping over GMPE(s)
                # print('eval_calc: cur_gs = ')
                # print(cur_gs)
                num_events = sum(len(eids_by_rlz[rlz]) for rlz in rlzs)
                if num_events == 0:  # it may happen
                    continue
                    # NB: the trick for performance is to keep the call to
                    # .compute outside of the loop over the realizations;
                    # it is better to have few calls producing big arrays
                tmpMean = []  # noqa: N806
                tmpstdtot = []
                tmpstdinter = []
                tmpstdintra = []
                if self.vtag >= 12:  # noqa: PLR2004
                    mean_stds_all = computer.cmaker.get_mean_stds(
                        [computer.ctx], StdDev.EVENT
                    )[0]
                for imti, imt in enumerate(computer.imts):
                    # Looping over IM(s)
                    # print('eval_calc: imt = ', imt)
                    if str(imt) in ['PGA', 'PGV', 'PGD']:  # noqa: PLR6201
                        cur_T = [0.0]  # noqa: N806
                        im_list.append(str(imt))
                        imTag = 'ln' + str(imt)  # noqa: N806
                    else:
                        if 'SA' not in im_list:
                            im_list.append('SA')
                        imTag = 'lnSA'  # noqa: N806
                    if isinstance(cur_gs, gsim.multi.MultiGMPE):
                        gs = cur_gs[str(imt)]  # MultiGMPE
                    else:
                        gs = cur_gs  # regular GMPE
                    try:
                        if self.vtag >= 12:  # noqa: PLR2004
                            mean_stds = mean_stds_all[:, imti]
                            num_sids = len(computer.sids)
                            num_stds = len(mean_stds)
                            if num_stds == 1:
                                # no standard deviation is available
                                # for truncation_level = 0 there is only mean, no stds
                                if computer.correlation_model:
                                    raise ValueError(  # noqa: DOC501, TRY003, TRY301
                                        'truncation_level=0 requires '  # noqa: EM101
                                        'no correlation model'
                                    )
                                mean = mean_stds[0]
                                stddev_intra = 0
                                stddev_inter = 0
                                stddev_total = 0
                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = np.concatenate(
                                        (tmpstdtot, stddev_total), axis=0
                                    )
                            elif num_stds == 2:  # noqa: PLR2004
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                # By default, we evaluate stddev_inter as the stddev_total

                                if self.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(  # noqa: DOC501, TRY301
                                        self.correlation_model, gsim
                                    )

                                mean, stddev_total = mean_stds
                                stddev_total = stddev_total.reshape(
                                    stddev_total.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005
                                stddev_inter = stddev_total
                                stddev_intra = 0
                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = np.concatenate(
                                        (tmpstdtot, stddev_total), axis=0
                                    )
                            else:
                                mean, stddev_inter, stddev_intra = mean_stds
                                stddev_intra = stddev_intra.reshape(
                                    stddev_intra.shape + (1,)  # noqa: RUF005
                                )
                                stddev_inter = stddev_inter.reshape(
                                    stddev_inter.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005
                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(
                                        stddev_inter * stddev_inter
                                        + stddev_intra * stddev_intra
                                    )
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = np.concatenate(
                                        (
                                            tmpstdtot,
                                            np.sqrt(
                                                stddev_inter * stddev_inter
                                                + stddev_intra * stddev_intra
                                            ),
                                        ),
                                        axis=1,
                                    )

                        elif self.vtag == 11:  # noqa: PLR2004
                            # v11
                            dctx = computer.dctx.roundup(cur_gs.minimum_distance)
                            if computer.distribution is None:
                                if computer.correlation_model:
                                    raise ValueError(  # noqa: DOC501, TRY003, TRY301
                                        'truncation_level=0 requires '  # noqa: EM101
                                        'no correlation model'
                                    )
                                mean, _stddevs = cur_gs.get_mean_and_stddevs(
                                    computer.sctx,
                                    computer.rctx,
                                    dctx,
                                    imt,
                                    stddev_types=[],
                                )
                            num_sids = len(computer.sids)
                            if {
                                const.StdDev.TOTAL
                            } == cur_gs.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                if computer.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(  # noqa: DOC501, TRY301
                                        computer.correlation_model, cur_gs
                                    )

                                mean, [stddev_total] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx,
                                    computer.rctx,
                                    dctx,
                                    imt,
                                    [const.StdDev.TOTAL],
                                )
                                stddev_total = stddev_total.reshape(
                                    stddev_total.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005
                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)  # noqa: N806
                                    tmpstdtot = np.concatenate(
                                        (tmpstdtot, stddev_total), axis=0
                                    )
                            else:
                                mean, [stddev_inter, stddev_intra] = (
                                    cur_gs.get_mean_and_stddevs(
                                        computer.sctx,
                                        computer.rctx,
                                        dctx,
                                        imt,
                                        [
                                            const.StdDev.INTER_EVENT,
                                            const.StdDev.INTRA_EVENT,
                                        ],
                                    )
                                )
                                stddev_intra = stddev_intra.reshape(
                                    stddev_intra.shape + (1,)  # noqa: RUF005
                                )
                                stddev_inter = stddev_inter.reshape(
                                    stddev_inter.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005

                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(
                                        stddev_inter * stddev_inter
                                        + stddev_intra * stddev_intra
                                    )
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = np.concatenate(
                                        (
                                            tmpstdtot,
                                            np.sqrt(
                                                stddev_inter * stddev_inter
                                                + stddev_intra * stddev_intra
                                            ),
                                        ),
                                        axis=1,
                                    )

                        else:
                            # v10
                            dctx = computer.dctx.roundup(cur_gs.minimum_distance)
                            if computer.truncation_level == 0:
                                if computer.correlation_model:
                                    raise ValueError(  # noqa: DOC501, TRY003, TRY301
                                        'truncation_level=0 requires '  # noqa: EM101
                                        'no correlation model'
                                    )
                                mean, _stddevs = cur_gs.get_mean_and_stddevs(
                                    computer.sctx,
                                    computer.rctx,
                                    dctx,
                                    imt,
                                    stddev_types=[],
                                )
                            num_sids = len(computer.sids)  # noqa: F841
                            if {
                                const.StdDev.TOTAL
                            } == cur_gs.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
                                # If the GSIM provides only total standard deviation, we need
                                # to compute mean and total standard deviation at the sites
                                # of interest.
                                # In this case, we also assume no correlation model is used.
                                if computer.correlation_model:
                                    raise CorrelationButNoInterIntraStdDevs(  # noqa: DOC501, TRY301
                                        computer.correlation_model, cur_gs
                                    )

                                mean, [stddev_total] = cur_gs.get_mean_and_stddevs(
                                    computer.sctx,
                                    computer.rctx,
                                    dctx,
                                    imt,
                                    [const.StdDev.TOTAL],
                                )
                                stddev_total = stddev_total.reshape(
                                    stddev_total.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005
                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdtot = stddev_total
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=0)  # noqa: N806
                                    tmpstdtot = np.concatenate(
                                        (tmpstdtot, stddev_total), axis=0
                                    )
                            else:
                                mean, [stddev_inter, stddev_intra] = (
                                    cur_gs.get_mean_and_stddevs(
                                        computer.sctx,
                                        computer.rctx,
                                        dctx,
                                        imt,
                                        [
                                            const.StdDev.INTER_EVENT,
                                            const.StdDev.INTRA_EVENT,
                                        ],
                                    )
                                )
                                stddev_intra = stddev_intra.reshape(
                                    stddev_intra.shape + (1,)  # noqa: RUF005
                                )
                                stddev_inter = stddev_inter.reshape(
                                    stddev_inter.shape + (1,)  # noqa: RUF005
                                )
                                mean = mean.reshape(mean.shape + (1,))  # noqa: RUF005

                                if imti == 0:
                                    tmpMean = mean  # noqa: N806
                                    tmpstdinter = stddev_inter
                                    tmpstdintra = stddev_intra
                                    tmpstdtot = np.sqrt(
                                        stddev_inter * stddev_inter
                                        + stddev_intra * stddev_intra
                                    )
                                else:
                                    tmpMean = np.concatenate((tmpMean, mean), axis=1)  # noqa: N806
                                    tmpstdinter = np.concatenate(
                                        (tmpstdinter, stddev_inter), axis=1
                                    )
                                    tmpstdintra = np.concatenate(
                                        (tmpstdintra, stddev_intra), axis=1
                                    )
                                    tmpstdtot = np.concatenate(
                                        (
                                            tmpstdtot,
                                            np.sqrt(
                                                stddev_inter * stddev_inter
                                                + stddev_intra * stddev_intra
                                            ),
                                        ),
                                        axis=1,
                                    )

                    except Exception as exc:  # noqa: BLE001
                        raise RuntimeError(  # noqa: B904
                            '(%s, %s, source_id=%r) %s: %s'  # noqa: UP031
                            % (
                                gs,
                                imt,
                                computer.source_id.decode('utf8'),
                                exc.__class__.__name__,
                                exc,
                            )
                        ).with_traceback(exc.__traceback__)

                # initialize
                # NOTE: needs to be extended for gmpe logic tree
                gm_collector = []
                # collect data
                for k in range(tmpMean.shape[0]):
                    imResult = {}  # noqa: N806
                    if len(tmpMean):
                        imResult.update(
                            {'Mean': [float(x) for x in tmpMean[k].tolist()]}
                        )
                    if len(tmpstdtot):
                        imResult.update(
                            {
                                'TotalStdDev': [
                                    float(x) for x in tmpstdtot[k].tolist()
                                ]
                            }
                        )
                    if len(tmpstdinter):
                        imResult.update(
                            {
                                'InterEvStdDev': [
                                    float(x) for x in tmpstdinter[k].tolist()
                                ]
                            }
                        )
                    if len(tmpstdintra):
                        imResult.update(
                            {
                                'IntraEvStdDev': [
                                    float(x) for x in tmpstdintra[k].tolist()
                                ]
                            }
                        )
                    gm_collector.append({imTag: imResult})
                # print(gm_collector)

        # close datastore instance
        self.calculator.datastore.close()

        # stop dbserver
        if self.vtag >= 11:  # noqa: PLR2004
            cdbs.main('stop')
        else:
            cdbs.dbserver('stop')

        # terminate the subprocess
        if self.prc:
            self.prc.kill()

        # copy calc hdf file
        if self.vtag >= 11:  # noqa: PLR2004
            calc_id = datastore.get_last_calc_id()
            path = os.path.join(datastore.get_datadir(), 'calc_%d.hdf5' % calc_id)  # noqa: PTH118
        else:
            path = os.path.join(  # noqa: PTH118
                datastore.get_datadir(), 'calc_%d.hdf5' % self.calc_id
            )

        if self.dir_info:
            dir_output = self.dir_info['Output']
            try:
                shutil.copy2(path, dir_output)
                print('FetchOpenQuake: calc hdf file saved.')  # noqa: T201
            except:  # noqa: E722
                print('FetchOpenQuake: failed to copy calc hdf file.')  # noqa: T201

        # Final results
        res = {
            'Magnitude': mag,
            'Periods': cur_T,
            'IM': im_list,
            'GroundMotions': gm_collector,
        }

        # return
        return res  # noqa: RET504

    def calculator_build_events_from_sources(self):  # noqa: ANN201, C901
        """Prefilter the composite source model and store the source_info"""  # noqa: D400
        gsims_by_trt = self.calculator.csm.full_lt.get_gsims_by_trt()
        print('FetchOpenQuake: self.calculator.csm.src_groups = ')  # noqa: T201
        print(self.calculator.csm.src_groups)  # noqa: T201
        sources = self.calculator.csm.get_sources()
        print('FetchOpenQuake: sources = ')  # noqa: T201
        print(sources)  # noqa: T201
        for src in sources:
            src.nsites = 1  # avoid 0 weight
            src.num_ruptures = src.count_ruptures()
        maxweight = sum(sg.weight for sg in self.calculator.csm.src_groups) / (
            self.calculator.oqparam.concurrent_tasks or 1
        )
        print('FetchOpenQuake: weights = ')  # noqa: T201
        print([sg.weight for sg in self.calculator.csm.src_groups])  # noqa: T201
        print('FetchOpenQuake: maxweight = ')  # noqa: T201
        print(maxweight)  # noqa: T201
        # trt => potential ruptures
        eff_ruptures = general.AccumDict(accum=0)  # noqa: F821
        # nr, ns, dt
        calc_times = general.AccumDict(accum=np.zeros(3, np.float32))  # noqa: F821
        allargs = []
        if self.calculator.oqparam.is_ucerf():
            # manage the filtering in a special way
            for sg in self.calculator.csm.src_groups:
                for src in sg:
                    src.src_filter = self.calculator.srcfilter
            # otherwise it would be ultra-slow
            srcfilter = calc.filters.nofilter  # noqa: F821
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
                allargs.append((src_group, srcfilter, par))  # noqa: PERF401

        smap = []
        for curargs in allargs:
            smap.append(  # noqa: PERF401
                calc.stochastic.sample_ruptures(curargs[0], curargs[1], curargs[2])  # noqa: F821
            )

        print('smap = ')  # noqa: T201
        print(smap)  # noqa: T201
        self.calculator.nruptures = 0
        mon = self.calculator.monitor('saving ruptures')
        for tmp in smap:
            dic = next(tmp)
            print(dic)  # noqa: T201
            # NB: dic should be a dictionary, but when the calculation dies
            # for an OOM it can become None, thus giving a very confusing error
            if dic is None:
                raise MemoryError('You ran out of memory!')  # noqa: DOC501, EM101, TRY003
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
                    self.calculator.nruptures, self.calculator.nruptures + n
                )
                self.calculator.nruptures += n
                hdf5.extend(self.calculator.datastore['ruptures'], rup_array)  # noqa: F821
                hdf5.extend(self.calculator.datastore['rupgeoms'], rup_array.geom)  # noqa: F821

        if len(self.calculator.datastore['ruptures']) == 0:
            raise RuntimeError(  # noqa: DOC501, TRY003
                'No ruptures were generated, perhaps the '  # noqa: EM101
                'investigation time is too short'
            )

        # must be called before storing the events
        self.calculator.store_rlz_info(eff_ruptures)  # store full_lt
        self.calculator.store_source_info(calc_times)
        imp = commonlib.calc.RuptureImporter(self.calculator.datastore)  # noqa: F821
        print('self.calculator.datastore.getitem(ruptures)')  # noqa: T201
        print(self.calculator.datastore.getitem('ruptures'))  # noqa: T201
        with self.calculator.monitor('saving ruptures and events'):
            imp.import_rups_events(
                self.calculator.datastore.getitem('ruptures')[()],
                getters.get_rupture_getters,  # noqa: F821
            )


class CorrelationButNoInterIntraStdDevs(Exception):  # noqa: N818, D101
    def __init__(self, corr, gsim):  # noqa: ANN001, ANN204
        self.corr = corr
        self.gsim = gsim

    def __str__(self):  # noqa: D105, ANN204
        return (
            f'You cannot use the correlation model '
            f'{self.corr.__class__.__name__} with the '
            f'GSIM {self.gsim.__class__.__name__}, '
            f'that defines only the total standard deviation. '
            f'If you want to use a correlation model you '
            f'have to select a GMPE that provides the inter '
            f'and intra event standard deviations.'
        )


def to_imt_unit_values(vals, imt):  # noqa: ANN001, ANN201
    """Exponentiate the values unless the IMT is MMI"""  # noqa: D400
    if str(imt) == 'MMI':
        return vals
    return np.exp(vals)


def export_rupture_to_json(scenario_info, mlon, mlat, siteFile, work_dir):  # noqa: ANN001, ANN201, C901, N803, D103
    import json  # noqa: PLC0415

    from openquake.commonlib import readinput  # noqa: PLC0415
    from openquake.hazardlib import nrml, site, sourceconverter  # noqa: PLC0415
    from openquake.hazardlib.calc.filters import (  # noqa: PLC0415
        SourceFilter,
        get_distances,
    )
    from openquake.hazardlib.geo.mesh import Mesh, surface_to_arrays  # noqa: PLC0415
    from openquake.hazardlib.geo.surface.base import BaseSurface  # noqa: PLC0415

    in_dir = os.path.join(work_dir, 'Input')  # noqa: PTH118
    outfile = os.path.join(work_dir, 'Output', 'RupFile.geojson')  # noqa: PTH118
    erf_data = {'type': 'FeatureCollection'}
    oq = readinput.get_oqparam(
        dict(  # noqa: C408
            calculation_mode='classical',
            inputs={'site_model': [siteFile]},
            intensity_measure_types_and_levels="{'PGA': [0.1], 'SA(0.1)': [0.1]}",  # place holder for initiating oqparam. Not used in ERF
            investigation_time=str(
                scenario_info['EqRupture'].get('investigation_time', '50.0')
            ),
            gsim='AbrahamsonEtAl2014',  # place holder for initiating oqparam, not used in ERF
            truncation_level='99.0',  # place holder for initiating oqparam. not used in ERF
            maximum_distance=str(
                scenario_info['EqRupture'].get('maximum_distance', '2000')
            ),
            width_of_mfd_bin=str(
                scenario_info['EqRupture'].get('width_of_mfd_bin', '1.0')
            ),
            area_source_discretization=str(
                scenario_info['EqRupture'].get('area_source_discretization', '10')
            ),
        )
    )
    rupture_mesh_spacing = scenario_info['EqRupture']['rupture_mesh_spacing']
    rupture_mesh_spacing = scenario_info['EqRupture']['rupture_mesh_spacing']
    [src_nrml] = nrml.read(
        os.path.join(in_dir, scenario_info['EqRupture']['sourceFile'])  # noqa: PTH118
    )
    conv = sourceconverter.SourceConverter(
        scenario_info['EqRupture']['investigation_time'],
        rupture_mesh_spacing,
        width_of_mfd_bin=scenario_info['EqRupture']['width_of_mfd_bin'],
        area_source_discretization=scenario_info['EqRupture'][
            'area_source_discretization'
        ],
    )
    src_raw = conv.convert_node(src_nrml)
    sources = []
    sources_dist = []
    sources_id = []
    id = 0  # noqa: A001
    siteMeanCol = site.SiteCollection.from_points([mlon], [mlat])  # noqa: N806
    srcfilter = SourceFilter(siteMeanCol, oq.maximum_distance)
    minMag = scenario_info['EqRupture']['min_mag']  # noqa: N806
    maxMag = scenario_info['EqRupture']['max_mag']  # noqa: N806
    for i in range(len(src_nrml)):
        subnode = src_nrml[i]
        subSrc = src_raw[i]  # noqa: N806
        tag = (
            subnode.tag.rsplit('}')[1]
            if subnode.tag.startswith('{')
            else subnode.tag
        )
        if tag == 'sourceGroup':
            for j in range(len(subnode)):
                subsubnode = subnode[j]
                subsubSrc = subSrc[j]  # noqa: N806
                subtag = (
                    subsubnode.tag.rsplit('}')[1]
                    if subsubnode.tag.startswith('{')
                    else subsubnode.tag
                )
                if (
                    subtag.endswith('Source')
                    and srcfilter.get_close_sites(subsubSrc) is not None
                ):
                    subsubSrc.id = id
                    sources_id.append(id)
                    id += 1  # noqa: A001
                    sources.append(subsubSrc)
                    sourceMesh = subsubSrc.polygon.discretize(rupture_mesh_spacing)  # noqa: N806
                    sourceSurface = BaseSurface(sourceMesh)  # noqa: N806
                    siteMesh = Mesh(siteMeanCol.lon, siteMeanCol.lat)  # noqa: N806
                    sources_dist.append(sourceSurface.get_min_distance(siteMesh))
        elif (
            tag.endswith('Source') and srcfilter.get_close_sites(subSrc) is not None
        ):
            subSrc.id = id
            sources_id.append(id)
            id += 1  # noqa: A001
            sources.append(subSrc)
            sourceMesh = subSrc.polygon.discretize(rupture_mesh_spacing)  # noqa: N806
            sourceSurface = BaseSurface(sourceMesh)  # noqa: N806
            siteMesh = Mesh(siteMeanCol.lon, siteMeanCol.lat)  # noqa: N806
            sources_dist.append(sourceSurface.get_min_distance(siteMesh))
    sources_df = pd.DataFrame.from_dict(
        {'source': sources, 'sourceDist': sources_dist, 'sourceID': sources_id}
    )
    sources_df = sources_df.sort_values(['sourceDist'], ascending=(True))
    sources_df = sources_df.set_index('sourceID')
    allrups = []
    allrups_rRup = []  # noqa: N806
    allrups_srcId = []  # noqa: N806
    for src in sources_df['source']:
        src_rups = list(src.iter_ruptures())
        for i, rup in enumerate(src_rups):
            rup.rup_id = src.offset + i
            allrups.append(rup)
            allrups_rRup.append(rup.surface.get_min_distance(siteMeanCol))
            allrups_srcId.append(src.id)
    rups_df = pd.DataFrame.from_dict(
        {'rups': allrups, 'rups_rRup': allrups_rRup, 'rups_srcId': allrups_srcId}
    )
    rups_df = rups_df.sort_values(['rups_rRup'], ascending=(True))
    feature_collection = []
    for ind in rups_df.index:
        cur_dict = {'type': 'Feature'}
        cur_dist = rups_df.loc[ind, 'rups_rRup']
        if cur_dist <= 0.0:
            # skipping ruptures with distance exceeding the maxDistance
            continue
        rup = rups_df.loc[ind, 'rups']
        # s0=number of multi surfaces, s1=number of rows, s2=number of columns
        arrays = surface_to_arrays(rup.surface)  # shape (s0, 3, s1, s2)
        src_id = rups_df.loc[ind, 'rups_srcId']
        maf = rup.occurrence_rate
        if maf <= 0.0:
            continue
        ruptureSurface = rup.surface  # noqa: N806, F841
        # Properties
        cur_dict['properties'] = dict()  # noqa: C408
        name = sources_df.loc[src_id, 'source'].name
        cur_dict['properties'].update({'Name': name})
        Mag = float(rup.mag)  # noqa: N806
        if (Mag < minMag) or (Mag > maxMag):
            continue
        cur_dict['properties'].update({'Magnitude': Mag})
        cur_dict['properties'].update({'Rupture': int(rup.rup_id)})
        cur_dict['properties'].update({'Source': int(src_id)})
        cur_dict['properties'].update({'Rake': rup.rake})
        cur_dict['properties'].update({'Lon': rup.hypocenter.x})
        cur_dict['properties'].update({'Lat': rup.hypocenter.y})
        cur_dict['properties'].update({'Depth': rup.hypocenter.z})
        cur_dict['properties'].update({'trt': rup.tectonic_region_type})
        cur_dict['properties'].update(
            {
                'mesh': json.dumps(
                    [
                        [[[round(float(z), 5) for z in y] for y in x] for x in array]
                        for array in arrays
                    ]
                )
            }
        )
        if hasattr(rup, 'probs_occur'):
            cur_dict['properties'].update({'Probability': rup.probs_occur})
        else:
            cur_dict['properties'].update({'MeanAnnualRate': rup.occurrence_rate})
        if hasattr(rup, 'weight'):
            cur_dict['properties'].update({'weight': rup.weight})
        cur_dict['properties'].update(
            {'Distance': get_distances(rup, siteMeanCol, 'rrup')[0]}
        )
        cur_dict['properties'].update(
            {'DistanceRup': get_distances(rup, siteMeanCol, 'rrup')[0]}
        )
        # cur_dict['properties'].update({'DistanceSeis': get_distances(rup, siteMeanCol, 'rrup')})
        cur_dict['properties'].update(
            {'DistanceJB': get_distances(rup, siteMeanCol, 'rjb')[0]}
        )
        cur_dict['properties'].update(
            {'DistanceX': get_distances(rup, siteMeanCol, 'rx')[0]}
        )
        cur_dict['geometry'] = dict()  # noqa: C408
        # if (len(arrays)==1 and arrays[0].shape[1]==1 and arrays[0].shape[2]==1):
        #     # Point Source
        #     cur_dict['geometry'].update({'type': 'Point'})
        #     cur_dict['geometry'].update({'coordinates': [arrays[0][0][0][0], arrays[0][1][0][0]]})
        # elif len(rup.surface.mesh.shape)==1:
        if len(rup.surface.mesh.shape) == 1:
            # Point Source or area source
            top_edge = (
                rup.surface.mesh
            )  # See the get_top_edge_depth method of the BaseSurface class
            coordinates = []
            for i in range(len(top_edge.lats)):
                coordinates.append([top_edge.lons[i], top_edge.lats[i]])  # noqa: PERF401
            cur_dict['geometry'].update({'type': 'LineString'})
            cur_dict['geometry'].update({'coordinates': coordinates})
        else:
            # Line source
            top_edge = rup.surface.mesh[
                0:1
            ]  # See the get_top_edge_depth method of the BaseSurface class
            coordinates = []
            for i in range(len(top_edge.lats[0])):
                coordinates.append([top_edge.lons[0][i], top_edge.lats[0][i]])
            cur_dict['geometry'].update({'type': 'LineString'})
            cur_dict['geometry'].update({'coordinates': coordinates})
        feature_collection.append(cur_dict)
    maf_list_n = [-x['properties']['MeanAnnualRate'] for x in feature_collection]
    sort_ids = np.argsort(maf_list_n)
    feature_collection_sorted = [feature_collection[i] for i in sort_ids]
    del feature_collection
    erf_data.update({'features': feature_collection_sorted})
    print(  # noqa: T201
        f'FetchOpenquake: total {len(feature_collection_sorted)} ruptures are collected.'
    )
    # Output
    if outfile is not None:
        print(  # noqa: T201
            f'The collected ruptures are sorted by MeanAnnualRate and saved in {outfile}'
        )
        with open(outfile, 'w') as f:  # noqa: PLW1514, PTH123
            json.dump(erf_data, f, indent=2)


def get_site_rup_info_oq(source_info, siteList):  # noqa: ANN001, ANN201, N803, D103
    from openquake.hazardlib import site  # noqa: PLC0415
    from openquake.hazardlib.calc.filters import get_distances  # noqa: PLC0415

    rup = source_info['rup']
    distToRupture = []  # noqa: N806, F841
    distJB = []  # noqa: N806, F841
    distX = []  # noqa: N806, F841
    for i in range(len(siteList)):
        siteMeanCol = site.SiteCollection.from_points(  # noqa: N806
            [siteList[i]['lon']], [siteList[i]['lat']]
        )
        siteList[i].update({'rRup': get_distances(rup, siteMeanCol, 'rrup')[0]})
        siteList[i].update({'rJB': get_distances(rup, siteMeanCol, 'rjb')[0]})
        siteList[i].update({'rX': get_distances(rup, siteMeanCol, 'rx')[0]})
    site_rup_info = {
        'dip': float(rup.surface.get_dip()),
        'width': float(rup.surface.get_width()),
        'zTop': float(rup.rake),
        'zHyp': float(rup.hypocenter.depth),
        'aveRake': float(rup.rake),
    }
    return site_rup_info, siteList
