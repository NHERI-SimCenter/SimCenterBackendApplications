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
# Tamika Bassman
# Adam Zsarnóczay
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

class damping_model_base:
    """
    A class to represent the base of damping models.
    
    Attributes:
    ----------
    
    Methods:
    -------

    """
    def __init__(self):
        pass
    def name(self):
        return 'damping_model_base'

class damping_model_hazus(damping_model_base):
    """
    A class to represent the hazus damping models.
    
    Attributes:
    ----------
    beta_elastic_map : dict
        The damping ratio is suggested by FEMA HAZUS Below Eq. 5-9, which in turn
        is based on Newmark and Hall 1982.
        The median value of the dampling ratio in Table 3 of Newmark and Hall 1982
        is used. E.g. For steel buildings, the damping ratio is assumed as 
        (6+12.5)/2=9.25%, which is the average of welded steel and bolted steel.
        Masonry buildings are assumed to have a damping ratio similar to reinforced
        concrete buildings. Mobile homes are assumed to have a damping ratio similar
        to steel buildings.

    
    Methods:
    -------
    get_beta_elastic : Calculate the elastic damping ratio beta.
    """
    def __init__(self):
        self.beta_elastic_map = {
            'W1': 15,
            'W2': 15,
            'S1L': 10,
            'S1M': 7,
            'S1H': 5,
            'S2L': 10,
            'S2M': 7,
            'S2H': 5,
            'S3': 7,
            'S4L': 10,
            'S4M': 7,
            'S4H': 5,
            'S5L': 10,
            'S5M': 7,
            'S5H': 5,
            'C1L': 10, 
            'C1M': 8.5,
            'C1H': 7,
            'C2L': 10,
            'C2M': 8.5,
            'C2H': 7,
            'C3L': 10,
            'C3M': 8.5,
            'C3H': 7,
            'PC1': 8.5,
            'PC2L': 10,
            'PC2M': 8.5,
            'PC2H': 7,
            'RM1L': 10,
            'RM1M': 8.5,
            'RM2L': 10,
            'RM2M': 8.5,
            'RM2H': 7,
            'URML': 8.5,
            'URMM': 8.5,
            'MH': 9.25
        }
        self.kappa_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 
                                                   'hazus_kappa_data.csv'),
                                                   index_col=0, header=None)
        self.kappa_col_map = {'HC':{'S':1, 'M':2, 'L':3},
                              'MC':{'S':4, 'M':5, 'L':6},
                              'LC':{'S':7, 'M':8, 'L':9},
                              'PC':{'S':10, 'M':11, 'L':12},
        }

    def get_beta_elastic(self, HAZUS_bldg_type):
        """
        Calculate the elastic damping ratio beta.
        
        Parameters:
        -----------
        HAZUS_bldg_type : str
            The HAZUS building type.
        
        Returns:
        --------
        beta : float
            The elastic damping ratio beta.
        """
        if HAZUS_bldg_type not in self.beta_elastic_map.keys():
            sys.exit(f'The building type {HAZUS_bldg_type} is not in the damping'
                     'model.')
        beta = self.beta_elastic_map[HAZUS_bldg_type]
        return beta
    def get_kappa(self, HAZUS_bldg_type, design_level, Mw):
        """
        Calculate the kappa in Table 5-33 of FEMA HAZUS 2022.
        
        Parameters:
        -----------
        HAZUS_bldg_type : str
            The HAZUS building type.
        
        Returns:
        --------
        kappa : float
            The kappa in Table 5-33 of FEMA HAZUS 2022.
        """
        if HAZUS_bldg_type not in self.beta_elastic_map.keys():
            sys.exit(f'The building type {HAZUS_bldg_type} is not in the damping'
                     'model.')
        # Infer duration according to HAZUS 2022 below Table 5-33
        if Mw <= 5.5:
            duration = 'S'
        elif Mw < 7.5:
            duration = 'M'
        else:
            duration = 'L'
        col = self.kappa_col_map[design_level][duration]
        kappa = self.kappa_data.loc[HAZUS_bldg_type, col]
        return kappa
    
    def get_name(self):
        return 'damping_model_hazus'

class HAZUS_cao_peterson_2006(damping_model_base):
    """
    A class to represent the damping model in Cao and Peterson 2006.
    
    Attributes:
    ----------
    
    Methods:
    -------
    """
    def __init__(self, demand, capacity, base_model = damping_model_hazus()):
        self.supported_capacity_model = ['HAZUS_cao_peterson_2006']
        self.supported_demand_model = ['HAZUS', 'HAZUS_lin_chang_2003']
        self.base_model = base_model
        if capacity.name() not in self.supported_capacity_model:
            sys.exit(f'The capacity model {capacity.name()} is not compatible'
                     'with the damping model: cao_peterson_2006.')
        if demand.name() not in self.supported_demand_model:
            sys.exit(f'The demand model {demand.name()} is not compatible'
                     'with the damping model: cao_peterson_2006.')
        self.capacity = capacity
        self.HAZUS_type = capacity.HAZUS_type
        self.design_level = capacity.design_level
        self.Mw = demand.Mw
    

    def get_beta(self, Dp, Ap):
        """
        Equation B.44-B.45 in Steelman & Hajjar (2010), which are originally published
        in Cao and Peterson 2006
        """
        try:
            beta_elastic = self.base_model.get_beta_elastic(self.HAZUS_type)
        except: # noqa: E722
            sys.exit(f'The base model {self.base_model} does not have a useful'
                     'get_beta_elastic method.')
        try:
            kappa = self.base_model.get_kappa(self.HAZUS_type, self.design_level, self.Mw)
        except: # noqa: E722
            sys.exit(f'The base model {self.base_model} does not have a useful'
                     'get_kappa method.')
        Du = self.capacity.Du
        Ax = self.capacity.Ax
        B = self.capacity.B
        C = self.capacity.C
        Kt = (Du-Dp)/(Ap-Ax)*(B/C)**2 # Eq B.46
        Ke = self.capacity.Ay/self.capacity.Dy # Eq B.47
        area_h = max(0,4*(Ap-Dp*Ke)*(Dp*Kt-Ap)/(Ke-Kt)) # Eq. B.45
        # beta is in the unit of percentage
        # beta_h = kappa*area_h/(2*3.1416*Dp*Ap) * 100# Eq. B.44
        beta_h = kappa*area_h/(2*3.1416*Dp*Ap)# Eq. B.44
        return beta_elastic + beta_h

    def get_beta_elastic(self):
        return self.base_model.get_beta_elastic(self.HAZUS_type)



        

    
         