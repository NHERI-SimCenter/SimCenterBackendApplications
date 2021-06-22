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

# handleError for permission denied
def handleError(func, path, exc_info):
    print('Handling Error for file ' , path)
    print(exc_info)
    # Check if file access issue
    if not os.access(path, os.W_OK):
       # Try to change the permision of file
       os.chmod(path, stat.S_IWUSR)
       # call the calling function again
       func(path)

# check that openquake module exists
if not os.path.isdir(os.path.dirname(os.path.realpath(__file__))+'/openquake'):
    if not os.path.isdir('./oq-engine'):
        try:
            os.system('git clone https://github.com/gem/oq-engine.git')
        except:
            print('FetchOpenQuake: could not clone https://github.com/gem/oq-engine.git')
    shutil.copytree('./oq-engine/openquake',os.path.dirname(os.path.realpath(__file__))+'/openquake')
    shutil.rmtree('oq-engine', onerror=handleError)

from openquake.baselib import config, version, performance, general, zeromq, hdf5, parallel
from openquake.hazardlib import const, calc, gsim
from openquake import commonlib
from openquake.commonlib import readinput, logictree, logs, datastore
from openquake.calculators import base, getters
from openquake.server import dbserver
from openquake.commands import dbserver as cdbs


def openquake_config(site_info, scen_info, event_info, dir_input):

    import configparser
    cfg = configparser.ConfigParser()
    # general section
    if scen_info['EqRupture']['Type'] == 'OpenQuakeScenario':
        cfg['general'] = {'description': 'Scenario Hazard Config File',
                          'calculation_mode': 'scenario'}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
        cfg['general'] = {'description': 'Scenario Hazard Config File',
                          'calculation_mode': 'event_based',
                          'ses_seed': 24}
        cfg['logic_tree'] = {'number_of_logic_tree_samples': 0}
    else:
        print('FetchOpenQuake: please specify Scenario[\'Generator\'], options: OpenQuakeScenario or OpenQuakeEventBased.')
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
                              'trucation_level': 3.0, 
                              'maximum_distance': scen_info['EqRupture'].get('max_Dist', 500.0),
                              'number_of_ground_motion_fields': event_info['NumberPerSite']}
    elif scen_info['EqRupture']['Type'] == 'OpenQuakeEventBased':
        imt = ''
        if event_info['IntensityMeasure']['Type'] == 'SA':
            for curT in event_info['IntensityMeasure']['Periods']:
                imt = imt + '"SA(' + str(curT) + ')": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], '
            imt = imt[:-2]
        elif event_info['IntensityMeasure']['Type'] == 'PGA':
            imt = '"PGA": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]'
        else:
            imt = event_info['IntensityMeasure']['Type'] + ': logscale(1, 200, 45)'
        cfg['calculation'] = {'source_model_logic_tree_file': scen_info['EqRupture']['Filename'],
                              'gsim_logic_tree_file': event_info['GMPE']['Parameters'],
                              'investigation_time': scen_info['EqRupture']['TimeSpan'],
                              'intensity_measure_types_and_levels': '{' + imt + '}', 
                              'random_seed': 42, 
                              'trucation_level': 3.0, 
                              'maximum_distance': 500.0,
                              'number_of_ground_motion_fields': event_info['NumberPerSite']}
    else:
        print('FetchOpenQuake: please specify Scenario[\'Generator\'], options: OpenQuakeScenario or OpenQuakeEventBased.')
        return 0
    # Write the ini
    filename_ini = os.path.join(dir_input, 'oq_job.ini')
    with open(filename_ini, 'w') as configfile:
        cfg.write(configfile)

    print('FetchOpenQuake: OpenQuake configured.')

    # return
    return filename_ini  
    

class OpenQuakeHazardCalc:

    def __init__(self, job_ini, event_info, no_distribute=False):
        """
        Initialize a calculation (reinvented from openquake.engine.engine)

        :param job_ini:
            Path to configuration file/archive or
            dictionary of parameters with at least a key "calculation_mode"
        """

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
            subprocess.Popen([sys.executable, '-m', 'openquake.commands', 'dbserver', 'start'])
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
        dic = readinput.get_params(job_ini)
        #dic['hazard_calculation_id'] = self.job.calc_id

        # Create the job log
        self.log = logs.init('job', dic, logging.INFO, None, None, None)

        # Get openquake parameters
        self.oqparam = self.log.get_oqparam()

        # Create the calculator
        self.calculator = base.calculators(self.oqparam, self.log.calc_id)
        self.calculator.from_engine = True

        print('FetchOpenQuake: OpenQuake Hazard Calculator initiated.')

    def run_calc(self):
        """
        Run a calculation and return results (reinvented from openquake.calculators.base)
        """

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
                print('self.__dict__ = ')
                print(self.calculator.__dict__)
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
                    imts = oq.get_primary_imtls()
                    nrups = len(dstore['ruptures'])
                    base.create_gmf_data(dstore, imts, oq.get_sec_imts())
                    dstore.create_dset('gmf_data/sigma_epsilon',
                                    getters.sig_eps_dt(oq.imtls))
                    dstore.create_dset('gmf_data/time_by_rup',
                                    getters.time_dt, (nrups,), fillvalue=None)

                # Prepare inputs for GmfGetter
                nr = len(dstore['ruptures'])
                logging.info('Reading {:_d} ruptures'.format(nr))
                rgetters = getters.get_rupture_getters(dstore, oq.concurrent_tasks * 1.25,
                                                    srcfilter=self.calculator.srcfilter)
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

        cur_getter = getters.GmfGetter(self.args[0][0], calc.filters.SourceFilter(
            self.dstore['sitecol'], self.dstore['oqparam'].maximum_distance), 
            self.calculator.param['oqparam'], self.calculator.param['amplifier'], 
            self.calculator.param['sec_perils'])

        print('eval_calc: cur_getter = ')
        print(cur_getter)

        # Evaluate each computer
        print('FetchOpenQuake: Evaluting ground motion models.')
        for computer in cur_getter.gen_computers(self.mon):
            # Looping over rupture(s) in the current realization
            sids = computer.sids
            print('eval_calc: site ID sids = ')
            print(sids)
            eids_by_rlz = computer.ebrupture.get_eids_by_rlz(
                cur_getter.rlzs_by_gsim)
            mag = computer.ebrupture.rupture.mag
            data = general.AccumDict(accum=[])
            cur_T = self.event_info['IntensityMeasure'].get('Periods', None)
            for cur_gs, rlzs in cur_getter.rlzs_by_gsim.items():
                # Looping over GMPE(s)
                print('eval_calc: cur_gs = ')
                print(cur_gs)
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
                for imti, imt in enumerate(computer.imts): 
                    # Looping over IM(s)
                    #print('eval_calc: imt = ')
                    #print(imt)
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

                    except Exception as exc:
                        raise RuntimeError(
                            '(%s, %s, source_id=%r) %s: %s' %
                            (gs, imt, computer.source_id.decode('utf8'),
                            exc.__class__.__name__, exc)
                        ).with_traceback(exc.__traceback__)

                # initialize
                gm_collector = []
                # transpose
                """
                if len(tmpMean):
                    tmpMean = tmpMean.transpose()
                if len(tmpstdinter):
                    tmpstdinter = tmpstdinter.transpose()
                if len(tmpstdintra):
                    tmpstdintra = tmpstdintra.transpose()
                if len(tmpstdtot):
                    tmpstdtot = tmpstdtot.transpose()
                """
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
                print(gm_collector)
        
        # close datastore instance
        self.calculator.datastore.close()
        
        # stop dbserver
        cdbs.main('stop')


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
