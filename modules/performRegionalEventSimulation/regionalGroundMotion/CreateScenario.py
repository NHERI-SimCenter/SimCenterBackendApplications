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

import json
import os
import random
import socket
import sys
import time

import numpy as np
import pandas as pd

if 'stampede2' not in socket.gethostname():
    from FetchOpenSHA import *  # noqa: F403


def get_rups_to_run(scenario_info, user_scenarios, num_scenarios):  # noqa: ANN001, ANN201, C901, D103, PLR0912
    # If there is a filter
    if scenario_info['Generator'].get('method', None) == 'ScenarioSpecific':
        SourceIndex = scenario_info['Generator'].get('SourceIndex', None)  # noqa: N806
        RupIndex = scenario_info['Generator'].get('RuptureIndex', None)  # noqa: N806
        if (SourceIndex is None) or (RupIndex is None):
            print(  # noqa: T201
                'Both SourceIndex and RuptureIndex are needed for'
                'ScenarioSpecific analysis'
            )
            return None
        rups_to_run = []
        for ind in range(len(user_scenarios.get('features'))):
            cur_rup = user_scenarios.get('features')[ind]
            cur_id_source = cur_rup.get('properties').get('Source', None)
            if cur_id_source != int(SourceIndex):
                continue
            cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
            if cur_id_rupture == int(RupIndex):
                rups_to_run.append(ind)
                break
    elif scenario_info['Generator'].get('method', None) == 'MonteCarlo':
        rup_filter = scenario_info['Generator'].get('RuptureFilter', None)
        if rup_filter is None or len(rup_filter) == 0:
            rups_to_run = list(range(num_scenarios))
        else:
            rups_requested = []
            for rups in rup_filter.split(','):
                if '-' in rups:
                    asset_low, asset_high = rups.split('-')
                    rups_requested += list(
                        range(int(asset_low), int(asset_high) + 1)
                    )
                else:
                    rups_requested.append(int(rups))
            rups_requested = np.array(rups_requested)
            rups_requested = (
                rups_requested - 1
            )  # The input index starts from 1, not 0
            rups_available = list(range(num_scenarios))
            rups_to_run = rups_requested[
                np.where(np.isin(rups_requested, rups_available))[0]
            ]
        # Select all
    elif scenario_info['Generator'].get('method', None) == 'Subsampling':
        rups_to_run = list(range(num_scenarios))
    else:
        sys.exit(
            f'The scenario selection method {scenario_info["Generator"].get("method", None)} is not available'  # noqa: E501
        )
    return rups_to_run


def load_earthquake_rupFile(scenario_info, rupFilePath):  # noqa: ANN001, ANN201, N802, N803, D103
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    try:
        with open(rupFilePath) as f:  # noqa: PTH123
            user_scenarios = json.load(f)
    except:  # noqa: E722
        sys.exit(f'CreateScenario: source file {rupFilePath} not found.')
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features', []))
    if num_scenarios < 1:
        sys.exit('CreateScenario: source file is empty.')
    rups_to_run = get_rups_to_run(scenario_info, user_scenarios, num_scenarios)
    # get rupture and source ids
    scenario_data = {}
    if source_type == 'ERF':
        # source model
        source_model = scenario_info['EqRupture']['Model']
        for rup_tag in rups_to_run:
            cur_rup = user_scenarios.get('features')[rup_tag]
            cur_id_source = cur_rup.get('properties').get('Source', None)
            cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
            scenario_data.update(
                {
                    rup_tag: {
                        'Type': source_type,
                        'RuptureForecast': source_model,
                        'Name': cur_rup.get('properties').get('Name', ''),
                        'Magnitude': cur_rup.get('properties').get(
                            'Magnitude', None
                        ),
                        'MeanAnnualRate': cur_rup.get('properties').get(
                            'MeanAnnualRate', None
                        ),
                        'SourceIndex': cur_id_source,
                        'RuptureIndex': cur_id_rupture,
                        'SiteSourceDistance': cur_rup.get('properties').get(
                            'Distance', None
                        ),
                        'SiteRuptureDistance': cur_rup.get('properties').get(
                            'DistanceRup', None
                        ),
                    }
                }
            )
    elif source_type == 'PointSource':
        sourceID = 0  # noqa: N806
        rupID = 0  # noqa: N806
        for rup_tag in rups_to_run:
            try:
                cur_rup = user_scenarios.get('features')[rup_tag]
                magnitude = cur_rup.get('properties')['Magnitude']
                location = cur_rup.get('properties')['Location']
                average_rake = cur_rup.get('properties')['AverageRake']
                average_dip = cur_rup.get('properties')['AverageDip']
                scenario_data.update(
                    {
                        0: {
                            'Type': source_type,
                            'Magnitude': magnitude,
                            'Location': location,
                            'AverageRake': average_rake,
                            'AverageDip': average_dip,
                            'SourceIndex': sourceID,
                            'RuptureIndex': rupID,
                        }
                    }
                )
                rupID = rupID + 1  # noqa: N806
            except:  # noqa: PERF203, E722
                print('Please check point-source inputs.')  # noqa: T201
    # return  # noqa: ERA001
    return scenario_data


def load_ruptures_openquake(scenario_info, stations, work_dir, siteFile, rupFile):  # noqa: ANN001, ANN201, C901, N803, D103, PLR0915
    # Collecting all possible earthquake scenarios
    lat = []
    lon = []
    for s in stations:
        lat.append(s['lat'])
        lon.append(s['lon'])
    # Reference location
    mlat = np.mean(lat)
    mlon = np.mean(lon)
    import json

    from openquake.commonlib import readinput
    from openquake.hazardlib import nrml, site, sourceconverter
    from openquake.hazardlib.calc.filters import SourceFilter, get_distances
    from openquake.hazardlib.geo.mesh import Mesh
    from openquake.hazardlib.geo.surface.base import BaseSurface

    try:
        with open(rupFile) as f:  # noqa: PTH123
            user_scenarios = json.load(f)
    except:  # noqa: E722
        sys.exit(f'CreateScenario: source file {rupFile} not found.')
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features', []))
    if num_scenarios < 1:
        sys.exit('CreateScenario: source file is empty.')
    rups_to_run = get_rups_to_run(scenario_info, user_scenarios, num_scenarios)
    in_dir = os.path.join(work_dir, 'Input')  # noqa: PTH118
    oq = readinput.get_oqparam(
        dict(  # noqa: C408
            calculation_mode='classical',
            inputs={'site_model': [siteFile]},
            intensity_measure_types_and_levels="{'PGA': [0.1], 'SA(0.1)': [0.1]}",  # place holder for initiating oqparam. Not used in ERF  # noqa: E501
            investigation_time=str(
                scenario_info['EqRupture'].get('investigation_time', '50.0')
            ),
            gsim='AbrahamsonEtAl2014',  # place holder for initiating oqparam, not used in ERF  # noqa: E501
            truncation_level='99.0',  # place holder for initiating oqparam. not used in ERF  # noqa: E501
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
    allrups_mar = []
    for src in sources_df['source']:
        src_rups = list(src.iter_ruptures())
        for i, rup in enumerate(src_rups):
            rup.rup_id = src.offset + i
            allrups.append(rup)
            allrups_rRup.append(rup.surface.get_min_distance(siteMeanCol))
            allrups_srcId.append(src.id)
            allrups_mar.append(rup.occurrence_rate)
    rups_df = pd.DataFrame.from_dict(
        {
            'rups': allrups,
            'rups_rRup': allrups_rRup,
            'rups_srcId': allrups_srcId,
            'MeanAnnualRate': allrups_mar,
        }
    )
    rups_df = rups_df.sort_values(['rups_rRup'], ascending=(True))
    rups_df = rups_df[rups_df['rups_rRup'] > 0]
    maf_list_n = [-x for x in rups_df['MeanAnnualRate']]
    sort_ids = np.argsort(maf_list_n)
    rups_df = rups_df.iloc[sort_ids]
    rups_df.reset_index(drop=True, inplace=True)  # noqa: PD002
    # rups_df = rups_df = rups_df.sort_values(['MeanAnnualRate'], ascending = (False))  # noqa: ERA001, E501
    rups_df = rups_df.loc[rups_to_run, :]
    scenario_data = {}
    for ind in rups_df.index:
        src_id = int(rups_df.loc[ind, 'rups_srcId'])
        name = sources_df.loc[src_id, 'source'].name
        rup = rups_df.loc[ind, 'rups']
        scenario_data.update(
            {
                ind: {
                    'Type': 'oqSourceXML',
                    'RuptureForecast': 'oqERF',
                    'Name': name,
                    'Magnitude': float(rup.mag),
                    'MeanAnnualRate': getattr(rup, 'occurrence_rate', None),
                    'SourceIndex': src_id,
                    'RuptureIndex': int(rup.rup_id),
                    'SiteSourceDistance': sources_df.loc[src_id, 'sourceDist'][0],
                    'SiteRuptureDistance': get_distances(rup, siteMeanCol, 'rrup')[
                        0
                    ],
                    'rup': rup,
                }
            }
        )
    return scenario_data


def load_earthquake_scenarios(scenario_info, stations, dir_info):  # noqa: ANN001, ANN201, D103
    # Number of scenarios
    source_num = scenario_info.get('Number', 1)  # noqa: F841
    # sampling method
    samp_method = scenario_info['EqRupture'].get('Sampling', 'Random')  # noqa: F841
    # source model
    source_model = scenario_info['EqRupture']['Model']
    eq_source = getERF(scenario_info)  # noqa: F405
    # Getting earthquake rupture forecast data
    source_type = scenario_info['EqRupture']['Type']
    # Collecting all sites
    lat = []
    lon = []
    for s in stations['Stations']:
        lat.append(s['Latitude'])
        lon.append(s['Longitude'])
    # load scenario file
    user_scenario_file = os.path.join(  # noqa: PTH118
        dir_info.get('Input'), scenario_info.get('EqRupture').get('UserScenarioFile')
    )
    try:
        with open(user_scenario_file) as f:  # noqa: PTH123
            user_scenarios = json.load(f)
    except:  # noqa: E722
        print(f'CreateScenario: source file {user_scenario_file} not found.')  # noqa: T201
        return {}
    # number of features (i.e., ruptures)
    num_scenarios = len(user_scenarios.get('features', []))
    if num_scenarios < 1:
        print('CreateScenario: source file is empty.')  # noqa: T201
        return {}
    # get rupture and source ids
    scenario_data = {}
    for rup_tag in range(num_scenarios):
        cur_rup = user_scenarios.get('features')[rup_tag]
        cur_id_source = cur_rup.get('properties').get('Source', None)
        cur_id_rupture = cur_rup.get('properties').get('Rupture', None)
        if cur_id_rupture is None or cur_id_source is None:
            print(  # noqa: T201
                f'CreateScenario: rupture #{rup_tag} does not have valid source/rupture ID - skipped.'  # noqa: E501
            )
            continue
        cur_source, cur_rupture = get_source_rupture(  # noqa: F405
            eq_source, cur_id_source, cur_id_rupture
        )
        scenario_data.update(
            {
                rup_tag: {
                    'Type': source_type,
                    'RuptureForecast': source_model,
                    'Name': str(cur_source.getName()),
                    'Magnitude': float(cur_rupture.getMag()),
                    'MeanAnnualRate': float(
                        cur_rupture.getMeanAnnualRate(
                            eq_source.getTimeSpan().getDuration()
                        )
                    ),
                    'SourceIndex': cur_id_source,
                    'RuptureIndex': cur_id_rupture,
                    'SiteSourceDistance': get_source_distance(  # noqa: F405
                        eq_source, cur_id_source, lat, lon
                    ),
                    'SiteRuptureDistance': get_rupture_distance(  # noqa: F405
                        eq_source, cur_id_source, cur_id_rupture, lat, lon
                    ),
                }
            }
        )

    # return  # noqa: ERA001
    return scenario_data


def create_earthquake_scenarios(  # noqa: ANN201, C901, D103, PLR0912, PLR0915
    scenario_info, stations, work_dir, openquakeSiteFile=None  # noqa: ANN001, N803
):
    # # Number of scenarios
    # source_num = scenario_info.get('Number', 1)  # noqa: ERA001
    # if source_num == 'All':
    #     # Large number to consider all sources in the ERF
    #     source_num = 10000000  # noqa: ERA001
    out_dir = os.path.join(work_dir, 'Output')  # noqa: PTH118
    if scenario_info['Generator'] == 'Simulation':
        # TODO:  # noqa: FIX002, TD002, TD003, TD005
        print('Physics-based earthquake simulation is under development.')  # noqa: T201
        return 1
    # Searching earthquake ruptures that fulfill the request
    elif scenario_info['Generator'] == 'Selection':  # noqa: RET505
        # Collecting all possible earthquake scenarios
        lat = []
        lon = []
        for s in stations['Stations']:
            lat.append(s['Latitude'])
            lon.append(s['Longitude'])
        # Reference location
        mlat = np.mean(lat)
        mlon = np.mean(lon)
        ref_station = [mlat, mlon]
        # Getting earthquake rupture forecast data
        source_type = scenario_info['EqRupture']['Type']
        t_start = time.time()
        if source_type == 'ERF':
            if (
                'SourceIndex' in scenario_info['EqRupture'].keys()  # noqa: SIM118
                and 'RuptureIndex' in scenario_info['EqRupture'].keys()  # noqa: SIM118
            ):
                source_model = scenario_info['EqRupture']['Model']
                eq_source = getERF(scenario_info)  # noqa: F405
                # check source index list and rupture index list
                if type(scenario_info['EqRupture']['SourceIndex']) == int:  # noqa: E721
                    source_index_list = [scenario_info['EqRupture']['SourceIndex']]
                else:
                    source_index_list = scenario_info['EqRupture']['SourceIndex']
                if type(scenario_info['EqRupture']['RuptureIndex']) == int:  # noqa: E721
                    rup_index_list = [scenario_info['EqRupture']['RuptureIndex']]
                else:
                    rup_index_list = scenario_info['EqRupture']['RuptureIndex']
                if len(source_index_list) != len(rup_index_list):
                    print(  # noqa: T201
                        f'CreateScenario: source number {len(source_index_list)} should be matched by rupture number {len(rup_index_list)}'  # noqa: E501
                    )
                    return dict()  # noqa: C408
                # loop over all scenarios
                scenario_data = dict()  # noqa: C408
                for i in range(len(source_index_list)):
                    cur_source_index = source_index_list[i]
                    cur_rup_index = rup_index_list[i]
                    distToSource = get_source_distance(  # noqa: N806, F405
                        eq_source, cur_source_index, lat, lon
                    )
                    scenario_data.update(
                        {
                            i: {
                                'Type': source_type,
                                'RuptureForecast': source_model,
                                'SourceIndex': cur_source_index,
                                'RuptureIndex': cur_rup_index,
                                'SiteSourceDistance': distToSource,
                                'SiteRuptureDistance': get_rupture_distance(  # noqa: F405
                                    eq_source,
                                    cur_source_index,
                                    cur_rup_index,
                                    lat,
                                    lon,
                                ),
                            }
                        }
                    )
                return scenario_data
            else:  # noqa: RET505
                source_model = scenario_info['EqRupture']['Model']
                source_name = scenario_info['EqRupture'].get('Name', None)
                min_M = scenario_info['EqRupture'].get('min_Mag', 5.0)  # noqa: N806
                max_M = scenario_info['EqRupture'].get('max_Mag', 9.0)  # noqa: N806
                max_R = scenario_info['EqRupture'].get('max_Dist', 1000.0)  # noqa: N806
                eq_source = getERF(scenario_info)  # noqa: F405
                erf_data = export_to_json(  # noqa: F405, F841
                    eq_source,
                    ref_station,
                    outfile=os.path.join(out_dir, 'RupFile.geojson'),  # noqa: PTH118
                    EqName=source_name,
                    minMag=min_M,
                    maxMag=max_M,
                    maxDistance=max_R,
                )
                # Parsing data
                # feat = erf_data['features']  # noqa: ERA001
                # """
                # tag = []  # noqa: ERA001
                # for i, cur_f in enumerate(feat):
                #     if source_name and (source_name not in cur_f['properties']['Name']):  # noqa: E501
                #         continue  # noqa: ERA001
                #     if min_M > cur_f['properties']['Magnitude']:
                #         continue  # noqa: ERA001
                #     tag.append(i)  # noqa: ERA001
                # # Abstracting desired ruptures
                # s_tag = random.sample(tag, min(source_num, len(tag)))  # noqa: ERA001
                # """
                # t_start = time.time()  # noqa: ERA001
                # s_tag = sample_scenarios(rup_info=feat, sample_num=source_num, sample_type=samp_method, source_name=source_name, min_M=min_M)  # noqa: ERA001, E501
                # print('CreateScenario: scenarios sampled {0} sec'.format(time.time() - t_start))  # noqa: ERA001, E501
                # #erf_data['features'] = list(feat[i] for i in s_tag)  # noqa: ERA001
                # erf_data['features'] = [feat[i] for i in range(source_num)]  # noqa: ERA001
                # scenario_data = dict()  # noqa: ERA001
                # t_start = time.time()  # noqa: ERA001
                # for i, rup in enumerate(erf_data['features']):
                #     scenario_data.update({i: {
                #         'Type': source_type,  # noqa: ERA001
                #         'RuptureForecast': source_model,  # noqa: ERA001
                #         'Name': rup['properties']['Name'],  # noqa: ERA001
                #         'Magnitude': rup['properties']['Magnitude'],  # noqa: ERA001
                #         'MeanAnnualRate': rup['properties']['MeanAnnualRate'],  # noqa: ERA001
                #         'SourceIndex': rup['properties']['Source'],  # noqa: ERA001
                #         'RuptureIndex': rup['properties']['Rupture'],  # noqa: ERA001
                #         'SiteSourceDistance': get_source_distance(eq_source, rup['properties']['Source'], lat, lon),  # noqa: ERA001, E501
                #         'SiteRuptureDistance': get_rupture_distance(eq_source, rup['properties']['Source'], rup['properties']['Rupture'], lat, lon)  # noqa: E501
                #     }})  # noqa: ERA001
                # print('CreateScenario: scenarios collected {0} sec'.format(time.time() - t_start))  # noqa: ERA001, E501
                # # Cleaning tmp outputs
                # del erf_data
        elif source_type == 'PointSource':
            # Export to a geojson format RupFile.json
            outfile = os.path.join(out_dir, 'RupFile.geojson')  # noqa: PTH118
            pointSource_data = {'type': 'FeatureCollection'}  # noqa: N806
            feature_collection = []
            newRup = {  # noqa: N806
                'type': 'Feature',
                'properties': {
                    'Type': source_type,
                    'Magnitude': scenario_info['EqRupture']['Magnitude'],
                    'Location': scenario_info['EqRupture']['Location'],
                    'AverageRake': scenario_info['EqRupture']['AverageRake'],
                    'AverageDip': scenario_info['EqRupture']['AverageDip'],
                    'Source': 0,
                    'Rupture': 0,
                },
            }
            newRup['geometry'] = dict()  # noqa: C408
            newRup['geometry'].update({'type': 'Point'})
            newRup['geometry'].update(
                {
                    'coordinates': [
                        scenario_info['EqRupture']['Location']['Longitude'],
                        scenario_info['EqRupture']['Location']['Latitude'],
                    ]
                }
            )
            feature_collection.append(newRup)
            pointSource_data.update({'features': feature_collection})
            if outfile is not None:
                print(f'The collected point source ruptures are saved in {outfile}')  # noqa: T201
                with open(outfile, 'w') as f:  # noqa: PTH123
                    json.dump(pointSource_data, f, indent=2)
        elif source_type == 'oqSourceXML':
            import FetchOpenQuake

            siteFile = os.path.join(work_dir, 'Input', openquakeSiteFile)  # noqa: PTH118, N806
            FetchOpenQuake.export_rupture_to_json(
                scenario_info, mlon, mlat, siteFile, work_dir
            )
        print(  # noqa: T201
            f'CreateScenario: all scenarios configured {time.time() - t_start} sec'
        )
    # return  # noqa: ERA001
    return None


def sample_scenarios(  # noqa: ANN201, D103
    rup_info=[], sample_num=1, sample_type='Random', source_name=None, min_M=0.0  # noqa: ANN001, B006, N803
):
    if len(rup_info) == 0:
        print(  # noqa: T201
            'CreateScenario.sample_scenarios: no available scenario provided - please relax earthquake filters.'  # noqa: E501
        )
        return []

    feat = rup_info
    tag = []
    for i, cur_f in enumerate(feat):
        if source_name and (source_name not in cur_f['properties']['Name']):
            continue
        if min_M > cur_f['properties']['Magnitude']:
            continue
        tag.append(i)

    if sample_type == 'Random':
        s_tag = random.sample(tag, min(sample_num, len(tag)))

    elif sample_type == 'MAF':
        # maf list
        maf_list = [feat[x]['properties']['MeanAnnualRate'] for x in tag]
        # normalize maf list
        sum_maf = np.sum(maf_list)
        maf_list_n = [x / sum_maf for x in maf_list]
        # get sample
        s_tag = np.random.choice(tag, sample_num, p=maf_list_n).tolist()  # noqa: NPY002

    else:
        print('CreateScenario.sample_scenarios: please specify a sampling method.')  # noqa: T201
        s_tag = []

    # return  # noqa: ERA001
    return s_tag


def create_wind_scenarios(scenario_info, stations, data_dir):  # noqa: ANN001, ANN201, D103
    # Number of scenarios
    source_num = scenario_info.get('Number', 1)  # noqa: F841
    # Directly defining earthquake ruptures
    if scenario_info['Generator'] == 'Simulation':
        # Collecting site locations
        lat = []
        lon = []
        for s in stations['Stations']:
            lat.append(s['Latitude'])
            lon.append(s['Longitude'])
        # Save Stations.csv
        df = pd.DataFrame({'lat': lat, 'lon': lon})  # noqa: PD901
        df.to_csv(data_dir + 'Stations.csv', index=False, header=False)
        # Save Lat_w.csv
        lat_w = np.linspace(min(lat) - 0.5, max(lat) + 0.5, 100)
        df = pd.DataFrame({'lat_w': lat_w})  # noqa: PD901
        df.to_csv(data_dir + 'Lat_w.csv', index=False, header=False)
        # Parsing Terrain info
        df = pd.read_csv(  # noqa: PD901
            data_dir + scenario_info['Terrain']['Longitude'],
            header=None,
            index_col=None,
        )
        df.to_csv(data_dir + 'Long_wr.csv', header=False, index=False)
        df = pd.read_csv(  # noqa: PD901
            data_dir + scenario_info['Terrain']['Latitude'],
            header=None,
            index_col=None,
        )
        df.to_csv(data_dir + 'Lat_wr.csv', header=False, index=False)
        df = pd.read_csv(  # noqa: PD901
            data_dir + scenario_info['Terrain']['Size'], header=None, index_col=None
        )
        df.to_csv(data_dir + 'wr_sizes.csv', header=False, index=False)
        df = pd.read_csv(  # noqa: PD901
            data_dir + scenario_info['Terrain']['z0'], header=None, index_col=None
        )
        df.to_csv(data_dir + 'z0r.csv', header=False, index=False)
        # Parsing storm properties
        param = []
        param.append(scenario_info['Storm']['Landfall']['Latitude'])
        param.append(scenario_info['Storm']['Landfall']['Longitude'])
        param.append(scenario_info['Storm']['LandingAngle'])
        param.append(scenario_info['Storm']['Pressure'])
        param.append(scenario_info['Storm']['Speed'])
        param.append(scenario_info['Storm']['Radius'])
        df = pd.DataFrame({'param': param})  # noqa: PD901
        df.to_csv(data_dir + 'param.csv', index=False, header=False)
        df = pd.read_csv(  # noqa: PD901
            data_dir + scenario_info['Storm']['Track'], header=None, index_col=None
        )
        df.to_csv(data_dir + 'Track.csv', header=False, index=False)
        # Saving del_par.csv
        del_par = [0, 0, 0]  # default
        df = pd.DataFrame({'del_par': del_par})  # noqa: PD901
        df.to_csv(data_dir + 'del_par.csv', header=False, index=False)
        # Parsing resolution data
        delta_p = [1000.0, scenario_info['Resolution']['DivRad'], 1000000.0]
        delta_p.extend([0.0, scenario_info['Resolution']['DivDeg'], 360.0])
        delta_p.extend(
            [scenario_info['MeasureHeight'], 10, scenario_info['MeasureHeight']]
        )
        df = pd.DataFrame({'delta_p': delta_p})  # noqa: PD901
        df.to_csv(data_dir + 'delta_p.csv', header=False, index=False)
    else:
        print('Currently only supporting Simulation generator.')  # noqa: T201
