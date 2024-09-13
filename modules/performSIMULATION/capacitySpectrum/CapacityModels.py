#  # noqa: N999, D100
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
# Jinyan Zhao
# 
# References:
# 1. Cao, T., & Petersen, M. D. (2006). Uncertainty of earthquake losses due to 
# model uncertainty of input ground motions in the Los Angeles area. Bulletin of 
# the Seismological Society of America, 96(2), 365-376.
# 2. Steelman, J., & Hajjar, J. F. (2008). Systemic validation of consequence-based
# risk management for seismic regional losses.
# 3. Newmark, N. M., & Hall, W. J. (1982). Earthquake spectra and design. 
# Engineering monographs on earthquake criteria.
# 4. FEMA (2022), HAZUS – Multi-hazard Loss Estimation Methodology 5.0, 
# Earthquake Model Technical Manual, Federal Emergency Management Agency, Washington D.C.



import os
import sys
import time

import numpy as np
import pandas as pd

ap_DesignLevel = {1940: 'LC', 1975: 'MC', 2100: 'HC'}
# original:
# ap_DesignLevel = {1940: 'PC', 1940: 'LC', 1975: 'MC', 2100: 'HC'}
# Note that the duplicated key is ignored, and Python keeps the last
# entry.

ap_DesignLevel_W1 = {0: 'LC', 1975: 'MC', 2100: 'HC'}
# original:
# ap_DesignLevel_W1 = {0: 'PC', 0: 'LC', 1975: 'MC', 2100: 'HC'}
# same thing applies

def convert_story_rise(structureType, stories):
    if structureType in ['W1', 'W2', 'S3', 'PC1', 'MH']:
        # These archetypes have no rise information in their IDs
        rise = None

    else:
        # First, check if we have valid story information
        try:
            stories = int(stories)

        except (ValueError, TypeError):
            raise ValueError(
                'Missing "NumberOfStories" information, '
                'cannot infer `rise` attribute of archetype'
            )

        if structureType == 'RM1':
            if stories <= 3:
                rise = "L"

            else:
                rise = "M"

        elif structureType == 'URM':
            if stories <= 2:
                rise = "L"

            else:
                rise = "M"

        elif structureType in [
            'S1',
            'S2',
            'S4',
            'S5',
            'C1',
            'C2',
            'C3',
            'PC2',
            'RM2',
        ]:
            if stories <= 3:
                rise = "L"

            elif stories <= 7:
                rise = "M"

            else:
                rise = "H"

    return rise

def auto_populate_hazus(GI):
    # get the building parameters
    bt = GI['StructureType']  # building type

        # get the design level
    dl = GI.get('DesignLevel', None)
    if dl is None:
        # If there is no DesignLevel provided, we assume that the YearBuilt is
        # available
        year_built = GI['YearBuilt']

        if 'W1' in bt:
            DesignL = ap_DesignLevel_W1
        else:
            DesignL = ap_DesignLevel
        
        for year in sorted(DesignL.keys()):
            if year_built <= year:
                dl = DesignL[year]
                break

    # get the number of stories / height
    stories = GI.get('NumberOfStories', None)

    # We assume that the structure type does not include height information
    # and we append it here based on the number of story information
    rise = convert_story_rise(bt, stories)

    if rise is not None:
        LF = f'{bt}{rise}'    
    else:
        LF = f'{bt}'
    return LF, dl


class capacity_model_base:
    """
    A class to represent the base of capacity models.
    
    Attributes:
    ----------
    
    Methods:
    -------
    """
    def __init__(self):
        pass
    def name(self):
        return 'capacity_model_base'

class cao_peterson_2006(capacity_model_base):
    """
    A class to represent the capacity model in Cao and Peterson 2006.
    
    Attributes:
    ----------
    Dy : float
        Yield displacement. In the unit of (inch)
    Ay : float
        Yield acceleration. In the unit of (g)
    Du : float
        Ultimate displacement. In the unit of (inch)
    Au : float
        Ultimate acceleration. In the unit of (g)
    Ax : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (g)
    B : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (g)
    C : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (inch)
    
    Methods:
    -------
    """
    def __init__(self, Dy, Ay, Du, Au, dD = 0.001):
        # region between elastic and perfectly plastic
        sd_elpl = np.arange(Dy,Du,dD)
        # Eq. B3 in Steelman & Hajjar 2008
        Ax = (Au**2*Dy - Ay**2*Du)/(2*Au*Dy - Ay*Dy - Ay*Du)
        # Eq. B4 in Steelman & Hajjar 2008
        B  = Au - Ax
        # Eq. B5 in Steelman & Hajjar 2008
        C  = (Dy*B**2*(Du-Dy)/(Ay*(Ay-Ax)))**0.5
        # Eq. B1 in Steelman & Hajjar 2008
        sa_elpl = Ax + B*(1 - ((sd_elpl-Du)/C)**2)**0.5
        # elastic and perfectly plastic regions
        sd_el = np.arange(0,Dy,dD)
        sd_pl = np.arange(Du,4*Du,dD)

        sa_el = sd_el*Ay/Dy
        sa_pl = Au*np.ones(len(sd_pl))

        self.sd = np.concatenate((sd_el,sd_elpl,sd_pl))
        self.sa = np.concatenate((sa_el,sa_elpl,sa_pl))
        self.Ax = Ax
        self.B = B
        self.C = C
        self.Du = Du
        self.Ay = Ay
        self.Dy = Dy
    
    def name(self):
        return 'cao_peterson_2006'
    
class HAZUS_cao_peterson_2006(capacity_model_base):
    """
    A class to represent the capacity model in Cao and Peterson 2006.
    
    Attributes:
    ----------
    Dy : float
        Yield displacement. In the unit of (inch)
    Ay : float
        Yield acceleration. In the unit of (g)
    Du : float
        Ultimate displacement. In the unit of (inch)
    Au : float
        Ultimate acceleration. In the unit of (g)
    Ax : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (g)
    B : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (g)
    C : float
        Parameter in Eq. A5 of Cao and Peterson 2006. In the unit of (inch)
    
    Methods:
    -------
    """
    def __init__(self, general_info, dD = 0.001):
        # HAZUS capacity data: Table 5-7 to Tabl 5-10 in HAZUS 5.1
        self.capacity_data = dict()
        self.capacity_data['HC'] = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'HC_capacity_data.csv'),
                                                   index_col=0).to_dict(orient='index')
        self.capacity_data['MC'] = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'MC_capacity_data.csv'),
                                                   index_col=0).to_dict(orient='index')
        self.capacity_data['LC'] = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'LC_capacity_data.csv'),
                                                   index_col=0).to_dict(orient='index')
        self.capacity_data['PC'] = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'PC_capacity_data.csv'),
                                                   index_col=0).to_dict(orient='index')
        self.capacity_data['alpha2'] = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'hazus_capacity_alpha2.csv'),
                                                   index_col=0).to_dict(orient='index')
        self.capacity_data['roof_height'] = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                                    'hazus_typical_roof_height.csv'),
                                                    index_col=0).to_dict(orient='index')
        # auto populate to get the parameters
        self.HAZUS_type, self.design_level = auto_populate_hazus(general_info)
        try:
            self.Du = self.capacity_data[self.design_level][self.HAZUS_type]['Du']
            self.Au = self.capacity_data[self.design_level][self.HAZUS_type]['Au']
            self.Dy = self.capacity_data[self.design_level][self.HAZUS_type]['Dy']
            self.Ay = self.capacity_data[self.design_level][self.HAZUS_type]['Ay']
        except KeyError:
            raise KeyError(f'No capacity data for {self.HAZUS_type} and {self.design_level}')
        self.cao_peterson_2006 = cao_peterson_2006(self.Dy, self.Ay, self.Du, self.Au, dD)
        self.Ax = self.cao_peterson_2006.Ax
        self.B = self.cao_peterson_2006.B
        self.C = self.cao_peterson_2006.C
    
    def get_capacity_curve(self, sd_max):
        sd = self.cao_peterson_2006.sd
        sa = self.cao_peterson_2006.sa
        if sd_max > sd[-1]:
            num_points = min(500, int((sd_max - self.cao_peterson_2006.sd[-1])/0.001))
            sd = np.concatenate((sd,np.linspace(
                self.cao_peterson_2006.sd[-1], sd_max, num_points)))
            sa = np.concatenate((sa, sa[-1]*np.ones(num_points)))
        return sd, sa

    # def get_capacity_curve(self):
    #     return self.cao_peterson_2006.sd, self.cao_peterson_2006.sa
    
    def get_hazus_alpha2(self):
        return self.capacity_data['alpha2'][self.HAZUS_type]['alpha2']
    
    def get_hazus_roof_height(self):
        return self.capacity_data['roof_height'][self.HAZUS_type]['roof_height_ft']

    def name(self):
        return 'HAZUS_cao_peterson_2006'
         