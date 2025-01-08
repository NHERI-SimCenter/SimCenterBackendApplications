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
# The computation method of ground motion intensity map using Markhivida et al. and
# the Baker-Jayaram correlation models is contributed by Dr. Anne Husley's
# seaturtles package (https://github.com/annehulsey/seaturtles).
#
# Contributors:
# Anne Husley
# Kuanshi Zhong
# Jinyan Zhao
import sys  # noqa: I001
import time
import warnings

import h5py
import numpy as np
import ujson
import geopandas as gpd
from scipy.spatial.distance import cdist
from gmpe import CorrelationModel
from tqdm import tqdm

IM_CORR_INTER = {
    'Baker & Jayaram (2008)': ['SA', 'PGA'],
    'Baker & Bradley (2017)': ['SA', 'PGA', 'PGV', 'DS575H', 'DS595H'],
}

IM_CORR_INTRA = {
    'Jayaram & Baker (2009)': ['SA', 'PGA'],
    'Loth & Baker (2013)': ['SA', 'PGA'],
    'Markhvida et al. (2017)': ['SA', 'PGA'],
    'Du & Ning (2021)': ['SA', 'PGA', 'PGV', 'Ia', 'CAV', 'DS575H', 'DS595H'],
}

IM_CORR = {'INTER': IM_CORR_INTER, 'INTRA': IM_CORR_INTRA}


def simulate_ground_motion(  # noqa: D103
    stations,
    im_raw_path,
    im_list,
    scenarios,
    num_simu,
    correlation_info,
    im_info,
    eq_ids,
):
    # create a ground motion simulator
    ln_im_mr = []
    mag_maf = []
    t_start = time.time()
    im_sampled = dict()  # noqa: C408
    if im_raw_path.endswith('.json'):
        with open(im_raw_path) as f:  # noqa: PTH123
            im_raw = ujson.load(f)
        for i in eq_ids:
            im_sampled.update({i: im_raw[str(i)]})
        gm_simulator = GM_Simulator(
            site_info=stations,
            im_list=im_list,
            num_simu=num_simu,
            correlation_info=correlation_info,
            im_info=im_info,
        )
    elif im_raw_path.endswith('.hdf5'):
        with h5py.File(im_raw_path, 'r') as f:
            for i in eq_ids:
                sample = dict()  # noqa: C408
                sample.update({'Mean': f[str(i)]['Mean'][()]})
                sample.update({'InterEvStdDev': f[str(i)]['InterEvStdDev'][()]})
                sample.update({'IntraEvStdDev': f[str(i)]['IntraEvStdDev'][()]})
                im_sampled.update({i: sample})
        gm_simulator = GM_Simulator_hdf5(
            site_info=stations,
            im_list=im_list,
            num_simu=num_simu,
            correlation_info=correlation_info,
            im_info=im_info,
        )
    else:
        SystemError(f'Unrecognized IM mean and stddev file format in {im_raw_path}')  # noqa: PLW0133
    im_raw = im_sampled
    for scen_i in tqdm(
        range(len(eq_ids)),
        desc=f'ComputeIntensityMeasure for {len(eq_ids)} scenarios',
    ):
        # for i, cur_im_raw in enumerate(im_raw):
        # print('ComputeIntensityMeasure: Scenario #{}/{}'.format(i+1,len(im_raw)))
        cur_im_raw = im_raw[eq_ids[scen_i]]
        # set im_raw
        gm_simulator.set_im_raw(cur_im_raw, im_list)
        # Computing inter event residuals
        # t_start = time.time()
        epsilon = gm_simulator.compute_inter_event_residual()
        # print('ComputeIntensityMeasure: inter-event correlation {0} sec'.format(time.time() - t_start))
        # Computing intra event residuals
        # t_start = time.time()
        eta = gm_simulator.compute_intra_event_residual()
        # print('ComputeIntensityMeasure: intra-event correlation {0} sec'.format(time.time() - t_start))
        ln_im_all = np.zeros((gm_simulator.num_sites, gm_simulator.num_im, num_simu))
        for i in range(num_simu):
            epsilon_m = np.array(
                [epsilon[:, i] for j in range(gm_simulator.num_sites)]
            )
            ln_im_all[:, :, i] = (
                gm_simulator.get_ln_im()
                + gm_simulator.get_inter_sigma_im() * epsilon_m
                + gm_simulator.get_intra_sigma_im() * eta[:, :, i]
            )

        ln_im_mr.append(ln_im_all)
        scenario = scenarios[eq_ids[scen_i]]
        mag_maf.append(
            [
                scenario['Magnitude'],
                scenario.get('MeanAnnualRate', None),
                scenario.get('SiteSourceDistance', None),
                scenario.get('SiteRuptureDistance', None),
            ]
        )

    print(  # noqa: T201
        f'ComputeIntensityMeasure: all inter- and intra-event correlation {time.time() - t_start} sec'
    )
    # return
    return ln_im_mr, mag_maf


class GM_Simulator:  # noqa: D101
    def __init__(
        self,
        site_info=[],  # noqa: B006
        im_list=[],  # noqa: B006
        im_raw=dict(),  # noqa: B006, C408
        num_simu=0,
        correlation_info=None,
        im_info=None,
    ):
        self.set_sites(site_info)
        self.set_num_simu(num_simu)
        self.parse_correlation_info(correlation_info, im_info)
        self.set_im_raw(im_raw, im_list)

    def set_sites(self, site_info):  # noqa: D102
        # set sites
        self.sites = site_info.copy()
        self.num_sites = len(self.sites)
        if self.num_sites < 2:  # noqa: PLR2004
            self.stn_dist = None
            print(  # noqa: T201
                'GM_Simulator: Only one site is defined, spatial correlation models ignored.'
            )
            return
        self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        # site number check
        if self.num_sites < 2:  # noqa: PLR2004
            print('GM_Simulator: error - please give at least two sites.')  # noqa: T201
            self.stn_dist = None
            return
        # compute the distance matrix
        # tmp = np.zeros((self.num_sites, self.num_sites))
        # # for i in tqdm(range(self.num_sites)):
        #     loc_i = np.array([self.sites[i]['lat'], self.sites[i]['lon']])
        #     for j in range(self.num_sites):
        #         loc_j = np.array([self.sites[j]['lat'], self.sites[j]['lon']])
        #         # Computing station-wise distances
        #         tmp[i, j] = CorrelationModel.get_distance_from_lat_lon(loc_i, loc_j)
        # self.stn_dist = tmp
        loc_i = np.array(
            [
                [self.sites[i]['lat'], self.sites[i]['lon']]
                for i in range(self.num_sites)
            ]
        )
        loc_i_gdf = gpd.GeoDataFrame(
            {'geometry': gpd.points_from_xy(loc_i[:, 1], loc_i[:, 0])},
            crs='EPSG:4326',
        ).to_crs('EPSG:6500')
        lat = loc_i_gdf.geometry.y
        lon = loc_i_gdf.geometry.x
        loc_i = np.array([[lon[i], lat[i]] for i in range(self.num_sites)])
        loc_j = np.array([[lon[i], lat[i]] for i in range(self.num_sites)])
        distances = cdist(loc_i, loc_j, 'euclidean') / 1000  # in km
        self.stn_dist = distances

    def set_num_simu(self, num_simu):  # noqa: D102
        # set simulation number
        self.num_simu = num_simu

    def set_im_raw(self, im_raw, im_list):  # noqa: D102
        # get IM type list
        self.im_type_list = im_raw.get('IM', [])
        # get im_data
        self.im_data = im_raw.get('GroundMotions', [])
        # get period
        self.periods = [x for x in im_raw.get('Periods', []) if x is not None]
        # im name list
        self.im_name_list = im_list
        # set IM size
        self.num_im = len(self.im_name_list)
        self.cross_check_im_correlation()

    def get_ln_im(self):  # noqa: D102
        ln_im = []
        for i in range(self.num_sites):
            tmp_im_data = []
            for cur_im_type in self.im_type_list:
                tmp_im_data = (
                    tmp_im_data + self.im_data[i][f'ln{cur_im_type}']['Mean']
                )
            ln_im.append(tmp_im_data)
        return ln_im

    def get_inter_sigma_im(self):  # noqa: D102
        inter_sigma_im = []
        for i in range(self.num_sites):
            tmp_im_data = []
            for cur_im_type in self.im_type_list:
                tmp_im_data = (
                    tmp_im_data
                    + self.im_data[i][f'ln{cur_im_type}']['InterEvStdDev']
                )
            inter_sigma_im.append(tmp_im_data)
        return inter_sigma_im

    def get_intra_sigma_im(self):  # noqa: D102
        intra_sigma_im = []
        for i in range(self.num_sites):
            tmp_im_data = []
            for cur_im_type in self.im_type_list:
                tmp_im_data = (
                    tmp_im_data
                    + self.im_data[i][f'ln{cur_im_type}']['IntraEvStdDev']
                )
            intra_sigma_im.append(tmp_im_data)
        return intra_sigma_im

    def parse_correlation_info(self, correlation_info, im_info):  # noqa: C901, D102
        # default is no correlation model and uncorrelated motions if generated
        self.inter_cm = None
        self.intra_cm = None
        # parse correlation information if any
        if correlation_info is None:
            print(  # noqa: T201
                'GM_Simulator: warning - correlation information not found - results will be uncorrelated motions.'
            )
            return
        if correlation_info.get('Type', None) == 'Vector':
            inter_cm = dict()  # noqa: C408
            im_info.pop('Type')
            for im, item in im_info.items():
                # for im in self.im_type_list:
                inter_cm.update({im: item['InterEventCorr']})
            inter_cm_unique = list(set([item for _, item in inter_cm.items()]))  # noqa: C403
            if len(inter_cm_unique) == 1:
                inter_cm = inter_cm_unique[0]
            self.inter_cm = inter_cm
            intra_cm = dict()  # noqa: C408
            for im, item in im_info.items():
                # for im in self.im_type_list:
                intra_cm.update({im: item['IntraEventCorr']})
            intra_cm_unique = list(set([item for _, item in intra_cm.items()]))  # noqa: C403
            if len(intra_cm_unique) == 1:
                intra_cm = intra_cm_unique[0]
            self.intra_cm = intra_cm
            return

        # inter-event model
        if correlation_info.get('InterEvent', None):
            self.inter_cm = correlation_info['InterEvent']
        elif correlation_info.get('SaInterEvent', None):
            # back compatibility
            self.inter_cm = correlation_info['SaInterEvent']
        else:
            print(  # noqa: T201
                'GM_Simulator: no inter-event correlation information not found - results will be uncorrelated motions.'
            )
        # intra-event model
        if correlation_info.get('IntraEvent', None):
            self.intra_cm = correlation_info['IntraEvent']
        if correlation_info.get('SaIntraEvent', None):
            # back compatibility
            self.intra_cm = correlation_info['SaIntraEvent']
        else:
            print(  # noqa: T201
                'GM_Simulator: no intra-event correlation information not found - results will be uncorrelated motions.'
            )

    def cross_check_im_correlation(self):  # noqa: C901, D102
        # because each correlation model only applies to certain intensity measure
        # so here we check if the correlation models are applicable for the required intensity measures
        self.im_cm_inter_flag = True
        self.im_cm_intra_flag = True
        if type(self.inter_cm) == dict:  # noqa: E721
            for cur_im in self.im_type_list:
                avail_im_inter_cm = IM_CORR_INTER.get(self.inter_cm[cur_im])
                if (avail_im_inter_cm is None) or (cur_im not in avail_im_inter_cm):
                    print(  # noqa: T201
                        f'GM_Simulator.cross_check_im_correlation: warning - {cur_im} is not available in {self.inter_cm}'
                    )
                    self.im_cm_inter_flag = False
                    continue
        else:
            avail_im_inter_cm = IM_CORR_INTER.get(self.inter_cm)
            if avail_im_inter_cm is not None:
                for cur_im in self.im_type_list:
                    if cur_im not in avail_im_inter_cm:
                        print(  # noqa: T201
                            f'GM_Simulator.cross_check_im_correlation: warning - {cur_im} is not available in {self.inter_cm}'
                        )
                        self.im_cm_inter_flag = False
                        continue
        if type(self.intra_cm) == dict:  # noqa: E721
            for cur_im in self.im_type_list:
                avail_im_intra_cm = IM_CORR_INTRA.get(self.intra_cm[cur_im])
                if cur_im not in avail_im_intra_cm:
                    print(  # noqa: T201
                        f'GM_Simulator.cross_check_im_correlation: warning - {cur_im} is not available in {self.intra_cm}'
                    )
                    self.im_cm_intra_flag = False
                    continue
        else:
            avail_im_intra_cm = IM_CORR_INTRA.get(self.intra_cm)
            if avail_im_intra_cm is not None:
                for cur_im in self.im_type_list:
                    if cur_im not in avail_im_intra_cm:
                        print(  # noqa: T201
                            f'GM_Simulator.cross_check_im_correlation: warning - {cur_im} is not available in {self.intra_cm}'
                        )
                        self.im_cm_intra_flag = False
                        continue

    def compute_inter_event_residual_ij(self, cm, im_name_list_1, im_name_list_2):  # noqa: D102
        if cm == 'Baker & Jayaram (2008)':
            rho = np.array(
                [
                    CorrelationModel.baker_jayaram_correlation_2008(im1, im2)
                    for im1 in im_name_list_1
                    for im2 in im_name_list_2
                ]
            ).reshape([len(im_name_list_1), len(im_name_list_2)])
        elif cm == 'Baker & Bradley (2017)':
            rho = np.array(
                [
                    CorrelationModel.baker_bradley_correlation_2017(im1, im2)
                    for im1 in im_name_list_1
                    for im2 in im_name_list_2
                ]
            ).reshape([len(im_name_list_1), len(im_name_list_2)])
        else:
            # TODO: extending this to more inter-event correlation models  # noqa: TD002
            sys.exit(
                'GM_Simulator.compute_inter_event_residual: currently supporting Baker & Jayaram (2008), Baker & Bradley (2017)'
            )
        return rho

    def replace_submatrix(self, mat, ind1, ind2, mat_replace):  # noqa: D102
        for i, index in enumerate(ind1):
            mat[index, ind2] = mat_replace[i, :]
        return mat

    def compute_inter_event_residual(self):  # noqa: D102
        if type(self.inter_cm) == dict:  # noqa: E721
            rho = np.zeros([self.num_im, self.num_im])
            im_types = list(self.inter_cm.keys())
            for i in range(len(im_types)):
                for j in range(i + 1):
                    im_type_i = im_types[i]
                    im_type_j = im_types[j]
                    im_name_list_i = [
                        im_name
                        for im_name in self.im_name_list
                        if im_name.startswith(im_type_i)
                    ]
                    im_indices_i = [
                        index
                        for index, element in enumerate(self.im_name_list)
                        if element.startswith(im_type_i)
                    ]
                    im_name_list_j = [
                        im_name
                        for im_name in self.im_name_list
                        if im_name.startswith(im_type_j)
                    ]
                    im_indices_j = [
                        index
                        for index, element in enumerate(self.im_name_list)
                        if element.startswith(im_type_j)
                    ]
                    # In R2D, use SA(0.01) to approximate PGA
                    im_name_list_i = [
                        'SA(0.01)' if x == 'PGA' else x for x in im_name_list_i
                    ]
                    im_name_list_j = [
                        'SA(0.01)' if x == 'PGA' else x for x in im_name_list_j
                    ]
                    rho_ij = self.compute_inter_event_residual_ij(
                        self.inter_cm[im_types[i]], im_name_list_i, im_name_list_j
                    )
                    rho = self.replace_submatrix(
                        rho, im_indices_i, im_indices_j, rho_ij
                    )
                    if i != j:
                        rho = self.replace_submatrix(
                            rho, im_indices_j, im_indices_i, rho_ij.T
                        )
        else:
            rho = self.compute_inter_event_residual_ij(
                self.inter_cm, self.im_name_list, self.im_name_list
            )
        # Simulating residuals
        with warnings.catch_warnings():
            # The intra-event models produce rho with tiny negative eigen values
            # This warning is suppressed
            warnings.filterwarnings(
                'ignore',
                message='covariance is not symmetric positive-semidefinite.',
            )
            residuals = np.random.multivariate_normal(
                np.zeros(self.num_im), rho, self.num_simu
            ).T
        # return
        return residuals  # noqa: RET504

    def compute_intra_event_residual_i(self, cm, im_name_list, num_simu):  # noqa: D102
        if cm == 'Jayaram & Baker (2009)':
            rho = np.zeros((self.num_sites, self.num_sites, len(im_name_list)))
            for i in range(self.num_sites):
                for j in range(self.num_sites):
                    cur_stn_dist = self.stn_dist[i, j]
                    for k in range(len(im_name_list)):
                        rho[i, j, k] = (
                            CorrelationModel.jayaram_baker_correlation_2009(
                                im_name_list[k], cur_stn_dist, flag_clustering=False
                            )
                        )
            # Simulating residuals
            residuals = np.zeros((self.num_sites, len(im_name_list), num_simu))
            for k in range(self.num_im):
                residuals[:, k, :] = np.random.multivariate_normal(
                    np.zeros(self.num_sites), rho[:, :, k], num_simu
                ).T
        elif cm == 'Loth & Baker (2013)':
            residuals = CorrelationModel.loth_baker_correlation_2013(
                self.sites, im_name_list, num_simu, self.stn_dist
            )
        elif cm == 'Markhvida et al. (2017)':
            num_pc = 19
            residuals = CorrelationModel.markhvida_ceferino_baker_correlation_2017(
                self.sites, im_name_list, num_simu, self.stn_dist, num_pc
            )
        elif cm == 'Du & Ning (2021)':
            num_pc = 23
            residuals = CorrelationModel.du_ning_correlation_2021(
                self.sites, im_name_list, num_simu, self.stn_dist, num_pc
            )
        else:
            # TODO: extending this to more inter-event correlation models  # noqa: TD002
            sys.exit(
                'GM_Simulator.compute_intra_event_residual: currently supporting Jayaram & Baker (2009), Loth & Baker (2013),Markhvida et al. (2017), Du & Ning (2021)'
            )
        return residuals

    def compute_intra_event_residual(self):  # noqa: D102
        if type(self.intra_cm) == dict:  # noqa: E721
            cm_groups = dict()  # noqa: C408
            # Group the IMs using the same cm
            for key, item in self.intra_cm.items():
                if item not in cm_groups:
                    cm_groups.update({item: [key]})
                else:
                    cm_groups[item].append(key)
            residuals = np.zeros((self.num_sites, self.num_im, self.num_simu))
            for cm, im_types in cm_groups.items():
                # im_type_list = [im_name.split('(')[0] for im_name in self.im_name_list]
                im_name_list = [
                    im_name
                    for im_name in self.im_name_list
                    if im_name.split('(')[0] in im_types
                ]
                im_indices = [
                    index
                    for index, element in enumerate(self.im_name_list)
                    if element.split('(')[0] in im_types
                ]
                residuals_i = self.compute_intra_event_residual_i(
                    cm, im_name_list, self.num_simu
                )
                for i, ind in enumerate(im_indices):
                    residuals[:, ind, :] = residuals_i[:, i, :]
        else:
            residuals = self.compute_intra_event_residual_i(
                self.intra_cm, self.im_name_list, self.num_simu
            )
        # return
        return residuals


class GM_Simulator_hdf5(GM_Simulator):  # noqa: D101
    def __init__(
        self,
        site_info=[],  # noqa: B006
        im_list=[],  # noqa: B006
        num_simu=0,
        correlation_info=None,
        im_info=None,
    ):
        self.set_im_type(im_list)
        self.set_sites(site_info)
        self.set_num_simu(num_simu)
        self.parse_correlation_info(correlation_info, im_info)
        self.cross_check_im_correlation()

    def set_im_type(self, im_list):  # noqa: D102
        self.im_name_list = im_list
        im_types = set()
        for im in im_list:
            if im.startswith('PGA'):
                im_types.add('PGA')
            elif im.startswith('SA'):
                im_types.add('SA')
            elif im.startswith('PGV'):
                im_types.add('PGV')
            else:
                SyntaxError(f'Unrecognized im type: {im}')  # noqa: PLW0133
        # Add ims one by one because the order is important
        self.im_type_list = []
        if ('PGA') in im_types:
            self.im_type_list.append('PGA')
        if ('SA') in im_types:
            self.im_type_list.append('SA')
        if ('PGV') in im_types:
            self.im_type_list.append('PGV')

    def set_im_raw(self, im_raw, im_list):  # noqa: D102
        self.im_name_list = im_list
        self.num_im = len(im_list)
        self.im_data = im_raw
        self.cross_check_im_correlation()

    def get_ln_im(self):  # noqa: D102
        ln_im = []
        for i in range(self.num_sites):
            tmp_im_data = self.im_data['Mean'][i, :].tolist()
            ln_im.append(tmp_im_data)
        return ln_im

    def get_inter_sigma_im(self):  # noqa: D102
        inter_sigma_im = []
        for i in range(self.num_sites):
            tmp_im_data = self.im_data['InterEvStdDev'][i, :].tolist()
            inter_sigma_im.append(tmp_im_data)
        return inter_sigma_im

    def get_intra_sigma_im(self):  # noqa: D102
        intra_sigma_im = []
        for i in range(self.num_sites):
            tmp_im_data = self.im_data['IntraEvStdDev'][i, :].tolist()
            intra_sigma_im.append(tmp_im_data)
        return intra_sigma_im
