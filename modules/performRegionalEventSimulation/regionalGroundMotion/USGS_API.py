# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Leland Stanford Junior University
# Copyright (c) 2022 The Regents of the University of California
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

import os, requests, json
import numpy as np

class USGS_HazardCurve:

    def __init__(self,
                 longitude=None,
                 latitude=None,
                 vs30=None,
                 edition='E2014',
                 imt='PGA',
                 tag=None):

        if self._load_config():
            print('USGS_HazardCurve.__init__: configuration loaded.')
        else:
            print('USGS_HazardCurve.__init__: error in loading configuration file.')
            return

        if self._check_edition(edition):
            self.edition = self._check_edition(edition)
        else:
            print('USGS_HazardCurve.__init__: edition {} is not supported by USGS.'.format(edition))
            return

        query_region = self._get_region(longitude,latitude)
        if query_region is None:
            print('USGS_HazardCurve.__init__: site (lon, lat) = ({},{}) is not supported.'.format(longitude,latitude))
            return
        else:
            self.longitude = longitude
            self.latitude = latitude
            self.region = query_region
            print('USGS_HazardCurve.__init__: site (lon, lat) = ({},{}) is found in USGS region {}.'.format(longitude,latitude,self.region))
        
        if self._check_region(self.region):
            print('USGS_HazardCurve.__init__: region {} is set up.'.format(self.region))
        else:
            print('USGS_HazardCurve.__init__: region {} is not supported by edition {}.'.format(self.region,self.edition))
            return

        if self._check_vs30(vs30):
            self.vs30 = self._check_vs30(vs30)
        else:
            print('USGS_HazardCurve.__init__: vs30 {} is not supported by edition {} and reigon {}.'.format(vs30,self.edition,self.region))
            return

        if self._check_imt(imt):
            self.imt = imt
        else:
            print('USGS_HazardCurve.__init__: imt {} is not supported.'.format(imt))
            return
        
        self.tag = tag
        # return
        print('USGS_HazardCurve.__init__: configuration done.')
        return

    def _load_config(self):

        cur_path = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(cur_path,'lib','USGS_HazardCurveConfig.json')
        try:
            with open(config_file,'r') as f:
                self.config = json.load(f)
            return True
        except:
            self.config = {}
            return False

    def _check_edition(self, edition, auto_correction=True):

        # available editions
        ed_list = self.config.get('parameters').get('edition').get('values')
        self.avail_editions = [x.get('value') for x in ed_list]
        print('USGS_HazardCurve._check_edition: available editions: {}'.format(self.avail_editions))

        # check
        if edition in self.avail_editions:
            return edition
        else:
            if auto_correction:
                edition = self.avail_editions[0]
                return edition
            else:
                return False

    def _get_region(self, long, lat):

        self.all_regions = [x['value'] for x in self.config.get('parameters').get('region').get('values')]
        for i in range(len(self.config.get('parameters').get('region').get('values'))):
            cur_region = self.config.get('parameters').get('region').get('values')[i]
            if long >= cur_region.get('minlongitude') and long <= cur_region.get('maxlongitude'):
                if lat >= cur_region.get('minlatitude') and lat <= cur_region.get('maxlatitude'):
                    return self.all_regions[i]
        # return empty
        return None

    def _check_region(self, region):

        # available regions
        self.avail_regions = self.config.get('parameters').get('edition').get('values')[self.avail_editions.index(self.edition)].get('supports').get('region')
        
        # check
        if region in self.avail_regions:
            return True
        else:
            return False

    def _check_vs30(self, vs30):

        # get edition supported vs30
        vs30_avail_ed = [int(x) for x in self.config.get('parameters').get('edition').get('values')[self.avail_editions.index(self.edition)].get('supports').get('vs30')]

        # get region supported vs30
        #vs30_avail_rg = [int(x) for x in self.config.get('parameters').get('region').get('values')[self.avail_regions.index(self.region)].get('supports').get('vs30')]

        vs30_avail_all = vs30_avail_ed
        vs30_id = np.argmin(np.abs([vs30-x for x in vs30_avail_all]))
        return str(vs30_avail_all[vs30_id])

        return False

    def _check_imt(self, imt):

        # get supported imt:
        imt_available = self.config.get('parameters').get('region').get('values')[self.avail_regions.index(self.region)].get('supports').get('imt')
        # get period in a double list:
        period_available = [float(x.replace('P','.')[2:]) for x in imt_available if x.startswith('SA')]
        print('Periods available = ',period_available)
        if imt in imt_available:
            self.imt_list = [imt]
            return True
        else:
            cur_period = float(imt.replace('P','.')[2:])
            if cur_period < np.min(period_available) or cur_period > np.max(period_available):
                return False
            else:
                # interpolate periods
                self.period_list = []
                for i, p in enumerate(period_available):
                    if p > cur_period:
                        self.period_list.append(period_available[i-1])
                        self.period_list.append(p)
                        break
                self.imt_list = ['SA'+str(x).replace('.','P') for x in self.period_list]
                #print('self.imt_list = ',self.imt_list)
                return True
                        

    def fetch_url(self):

        self.res_json = []
        
        for cur_imt in self.imt_list:

            # set url
            usgs_url = 'https://earthquake.usgs.gov/nshmp-haz-ws/hazard/{}/{}/{}/{}/{}/{}'.format(self.edition,
                                                                                                  self.region,
                                                                                                  self.longitude,
                                                                                                  self.latitude,
                                                                                                  cur_imt,
                                                                                                  self.vs30)
            
            print('USGS_HazardCurve.fetch_url: {}.\n'.format(usgs_url))
            # request
            res = requests.get(usgs_url)
            if res.status_code == 200:
                self.res_json.append(res.json())
                #print('USGS_HazardCurve.fetch_url: {}'.format(self.res_json))
            else:
                # try 10 more times to overcome the api traffic issue
                for i in range(10):
                    res = requests.get(usgs_url)
                    if res.status_code == 200:
                        self.res_json.append(res.json())
                        return True
                else:
                    self.res_json.append(None)
                    print('USGS_HazardCurve.fetch_url: cannot get the data')
                    return False
        
        return True

            

    def get_hazard_curve(self):

        cur_ims = []
        cur_mafs = []
        cur_rps = []
        
        for cur_res_json in self.res_json:
        
            tmp_x = cur_res_json.get('response')[0].get('metadata').get('xvalues')
            tmp_y = cur_res_json.get('response')[0].get('data')[0].get('yvalues')
            cur_ims.append([tmp_x[i] for i in range(len(tmp_x)) if tmp_y[i]>0])
            cur_mafs.append([x for x in tmp_y if x > 0])
            cur_rps.append([1.0/x for x in cur_mafs[-1]])

        if len(self.res_json)==1:
            self.ims = cur_ims[0]
            self.mafs = cur_mafs[0]
            self.rps = cur_rps[0]
        else:
            num_levels = np.min([len(cur_mafs[0]),len(cur_mafs[1])])
            self.ims = cur_ims[0][0:num_levels]
            self.mafs = [np.interp(self.imt.replace('P','.')[2:],self.period_list,[cur_mafs[0][x],cur_mafs[1][x]]) for x in range(num_levels)]
            self.mafs = [x for x in self.mafs if x > 0]
            self.rps = [1.0/x for x in self.mafs]

        dict_hc = {
            "SiteID": self.tag,
            "ReturnPeriod": self.rps,
            "IM": self.ims
        }

        return dict_hc






