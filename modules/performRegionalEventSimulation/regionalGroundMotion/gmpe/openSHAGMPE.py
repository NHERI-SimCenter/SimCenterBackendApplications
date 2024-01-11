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
# Jinyan Zhao
# Transfered from openSHA to achieve better performance in r2d

import numpy as np
import pandas as pd
import os
import time
############### Chiou and Young (2014)
class chiou_youngs_2013():
    timeSetImt = 0
    timeCalc = 0
    supportedImt = None
    def __init__(self):
        self.coeff = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','CY14.csv'))
        self.coeff.iloc[:-2,0] = self.coeff.iloc[:-2,0].apply(lambda x: float(x))
        self.coeff = self.coeff.set_index('T')
        self.supportedImt = list(self.coeff.index)
        self.coeff = self.coeff.to_dict()
        
        # Constants same for all periods
        self.C2 = self.coeff["c2"]["PGA"]
        self.C4 = self.coeff["c4"]["PGA"]
        self.C4a = self.coeff["c4a"]["PGA"]
        self.dC4 = self.C4a - self.C4
        self.C11 = self.coeff["c11"]["PGA"]
        self.CRB = self.coeff["cRB"]["PGA"]
        self.PHI6 = self.coeff["phi6"]["PGA"]
        self.A = np.power(571, 4)
        self.B = np.power(1360, 4) + self.A
        self.CRBsq = self.CRB * self.CRB
        
    def setIMT(self, imt):
        if imt not in self.supportedImt:
            print(f"The imt {imt} is not supported by Chiou and Young (2014)")
            return None
        self.c1 = self.coeff["c1"][imt]
        self.c1a = self.coeff["c1a"][imt]
        self.c1b = self.coeff["c1b"][imt]
        self.c1c = self.coeff["c1c"][imt]
        self.c1d = self.coeff["c1d"][imt]
        self.c3 =  self.coeff["c3"][imt]
        self.c5 =  self.coeff["c5"][imt]
        self.c6 =  self.coeff["c6"][imt]
        self.c7 =  self.coeff["c7"][imt]
        self.c7b = self.coeff["c7b"][imt]
        self.c8b = self.coeff["c8b"][imt]
        self.c9 = self.coeff["c9"][imt]
        self.c9a = self.coeff["c9a"][imt]
        self.c9b = self.coeff["c9b"][imt]
        self.c11b = self.coeff["c11b"][imt]
        self.cn = self.coeff["cn"][imt]
        self.cM = self.coeff["cM"][imt]
        self.cHM = self.coeff["cHM"][imt]
        self.cgamma1 = self.coeff["cgamma1"][imt]
        self.cgamma2 = self.coeff["cgamma2"][imt]
        self.cgamma3 = self.coeff["cgamma3"][imt]
        self.phi1 = self.coeff["phi1"][imt]
        self.phi2 = self.coeff["phi2"][imt]
        self.phi3 = self.coeff["phi3"][imt]
        self.phi4 = self.coeff["phi4"][imt]
        self.phi5 = self.coeff["phi5"][imt]
        self.tau1 = self.coeff["tau1"][imt]
        self.tau2 = self.coeff["tau2"][imt]
        self.sigma1 = self.coeff["sigma1"][imt]
        self.sigma2 = self.coeff["sigma2"][imt]
        self.sigma3 = self.coeff["sigma3"][imt]

    # Center zTop on the zTop-M relation -- Equations 4, 5
    def calcMwZtop(self,style, Mw):
        mzTop = 0.0
        if style == "REVERSE":
            if Mw<=5.849:
                mzTop = 2.704
            else:
                mzTop = max(2.704 - 1.226 * (Mw - 5.849), 0)   
        else:
             mzTop =  2.673 if (Mw <= 4.970) else max(2.673 - 1.136 * (Mw - 4.970), 0)
        return mzTop * mzTop
	
    def calcSAref(self, Mw, rJB, rRup, rX, dip, zTop, style):
        # Magnitude scaling
        r1 = self.c1 + self.C2 * (Mw - 6.0) + ((self.C2 - self.c3) / self.cn) * np.log(1.0 + np.exp(self.cn * (self.cM - Mw)))
        # Near-field magnitude and distance scaling
        r2 = self.C4 * np.log(rRup + self.c5 * np.cosh(self.c6 * max(Mw - self.cHM, 0.0)))
        # Far-field distance scaling
        gamma = (self.cgamma1 + self.cgamma2 / np.cosh(max(Mw - self.cgamma3, 0.0)))
        r3 = self.dC4 * np.log(np.sqrt(rRup * rRup + self.CRBsq)) + rRup * gamma
        # Scaling with other source variables
        coshM = np.cosh(2 * max(Mw - 4.5, 0))
        cosDelta = np.cos(dip * np.pi/180.0)
        # Center zTop on the zTop-M relation
        deltaZtop = zTop - self.calcMwZtop(style, Mw)
        r4 = (self.c7 + self.c7b / coshM) * deltaZtop + (self.C11 + self.c11b / coshM) * cosDelta * cosDelta
        if style == "REVERSE":
            r4 += self.c1a + self.c1c / coshM
        elif style == "NORMAL":
            r4 += self.c1b + self.c1d / coshM
        else:
            r4 += 0.0
        # Hanging-wall effect
        r5 = 0.0
        if rX>=0.0:
            r5 = self.c9 * np.cos(dip * np.pi/180.0) * (self.c9a + (1.0 - self.c9a) * np.tanh(rX / self.c9b)) * (1 - np.sqrt(rJB * rJB + zTop * zTop) / (rRup + 1.0))
        return np.exp(r1 + r2 + r3 + r4 + r5)
    
    def calcSoilNonLin(self, vs30):
        exp1 = np.exp(self.phi3 * (min(vs30, 1130.0) - 360.0))
        exp2 = np.exp(self.phi3 * (1130.0 - 360.0))
        return self.phi2 * (exp1 - exp2)
    
    def calcZ1ref(self, vs30):
		# -- Equation 18
        vsPow4 = vs30 * vs30 * vs30 * vs30
        return np.exp(-7.15 / 4 * np.log((vsPow4 + self.A) / self.B)) / 1000.0 # km
    
    def calcDeltaZ1(self, z1p0, vs30):
        if (np.isnan(z1p0)):
            return 0.0
        return 1000.0 * (z1p0 - self.calcZ1ref(vs30))

    # Mean ground motion model -- Equation 12
    def calcMean(self, vs30, z1p0, snl, saRef):
        # Soil effect: linear response
        sl = self.phi1 * min(np.log(vs30 / 1130.0), 0.0)
        # Soil effect: nonlinear response (base passed in)
        snl *= np.log((saRef + self.phi4) / self.phi4)
        # Soil effect: sediment thickness
        dZ1 = self.calcDeltaZ1(z1p0, vs30)
        rkdepth = self.phi5 * (1.0 - np.exp(-dZ1 / self.PHI6))
        return np.log(saRef) + sl + snl + rkdepth
    
    def calcNLOsq(self, snl, saRef):
        NL0 = snl * saRef / (saRef + self.phi4)
        NL0sq = (1 + NL0) * (1 + NL0)
        return NL0sq
    
    def calcTauSq(self, NL0sq, mTest):
        tau = self.tau1 + (self.tau2 - self.tau1) / 1.5 * mTest
        tauSq = tau * tau * NL0sq
        return tauSq
    
    def calcPhiSq(self, vsInf, NL0sq, mTest):
        sigmaNL0 = self.sigma1 + (self.sigma2 - self.sigma1) / 1.5 * mTest
        vsTerm = self.sigma3 if vsInf else 0.7
        sigmaNL0 *= np.sqrt(vsTerm + NL0sq)
        phiSq = sigmaNL0 * sigmaNL0
        return phiSq
        
    def calc(self, Mw, rJB, rRup, rX, dip, zTop, vs30, vsInf, 
			z1p0, style):
        """
        Preliminary implementation of the Chiou & Youngs (2013) next generation
        attenuation relationship developed as part of NGA West II.
        Input
        imt intensity measure type ['PGA', 'PGV',1.0]
        Mw moment magnitude
        rJB Joyner-Boore distance to rupture (in km)
        rRup 3D distance to rupture plane (in km)
        rX distance X (in km)
        dip of rupture (in degrees)
        zTop depth to the top of the rupture (in km)
        vs30 average shear wave velocity in top 30 m (in m/sec)
        vsInferred whether vs30 is an inferred or measured value
        z1p0 depth to V<sub>s</sub>=1.0 km/sec (in km)
        style of faulting
        Output:
        Mean
        TotalStdDev
        InterEvStdDev
        IntraEvStdDev
        """
        saRef = self.calcSAref(Mw, rJB, rRup, rX, dip, zTop, style)
        soilNonLin = self.calcSoilNonLin(vs30)
        mean = self.calcMean(vs30, z1p0, soilNonLin, saRef)
        # Aleatory uncertainty model -- Equation 3.9
        # Response Term - linear vs. non-linear
        NL0sq = self.calcNLOsq(soilNonLin, saRef)
        # Magnitude thresholds
        mTest = min(max(Mw, 5.0), 6.5) - 5.0
        # Inter-event Term
        tauSq = self.calcTauSq(NL0sq, mTest)
        # Intra-event term
        phiSq = self.calcPhiSq(vsInf, NL0sq, mTest)

        stdDev = np.sqrt(tauSq + phiSq)

        return mean, stdDev, np.sqrt(tauSq), np.sqrt(phiSq)
    #https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/sha/imr/attenRelImpl/ngaw2/NGAW2_Wrapper.java#L220
    def getFaultFromRake(self,rake):
        if(rake >= 135 or rake <= -135):
            return "STRIKE_SLIP"
        elif rake>=-45 and rake <=45:
            return "STRIKE_SLIP"
        elif rake>=45 and rake <=135:
            return "REVERSE"
        else:
            return "NORMAL"
        
    def get_IM(self, Mw, site_rup_dict, site_info, im_info):
        vsInf = bool(site_info["vsInferred"])
        style = self.getFaultFromRake(site_rup_dict["aveRake"])
        if 'SA' in im_info['Type']:
            cur_T = im_info.get('Periods', None)
        elif im_info['Type'] == 'PGA':
            cur_T = ['PGA']
        elif im_info['Type'] == 'PGV':
            cur_T = ['PGV']
        else:
            print(f'The IM type {im_info["Type"]} is not supported')
        meanList = []
        stdDevList = []
        InterEvStdDevList = []
        IntraEvStdDevList = []
        for Tj in cur_T:
            start = time.process_time_ns()
            self.setIMT(Tj)
            self.timeSetImt += time.process_time_ns() - start
            start = time.process_time_ns()
            mean, stdDev, InterEvStdDev, IntraEvStdDev = self.calc(Mw, site_info["rJB"], site_info["rRup"], site_info["rX"], site_rup_dict["dip"], site_rup_dict["zTop"], site_info["vs30"], vsInf, site_info["z1pt0"]/1000.0, style)
            self.timeCalc += time.process_time_ns() - start
            meanList.append(mean)
            stdDevList.append(stdDev)
            InterEvStdDevList.append(InterEvStdDev)
            IntraEvStdDevList.append(IntraEvStdDev)
        saResult = {'Mean': meanList,
                'TotalStdDev': stdDevList,
                'InterEvStdDev': InterEvStdDevList,
                'IntraEvStdDev': IntraEvStdDevList}
        return saResult
        
        # Station
        # if station_info['Type'] == 'SiteList':
        #     siteSpec = station_info['SiteList']
        # for i in range(len(site_list)):

############## Abrahamson, Silva, and Kamai (2014)
class abrahamson_silva_kamai_2014():
    timeSetImt = 0
    timeCalc = 0
    supportedImt = None
    def __init__(self):
        self.coeff = pd.read_csv(os.path.join(os.path.dirname(__file__),'data','ASK14.csv'))
        self.coeff.iloc[:-2,0] = self.coeff.iloc[:-2,0].apply(lambda x: float(x))
        self.coeff = self.coeff.set_index('T')
        self.supportedImt = list(self.coeff.index)
        self.coeff = self.coeff.to_dict()
        
        # Authors declared constants
        self.A3 = 0.275
        self.A4 = -0.1
        self.A5 = -0.41
        self.M2 = 5.0
        self.N  = 1.5
        self.C4 = 4.5
        
        # implementation constants
        self.A = np.power(610, 4)
        self.B = np.power(1360, 4) + self.A
        self.VS_RK = 1180.0
        self.A2_HW = 0.2
        self.H1 = 0.25
        self.H2 = 1.5
        self.H3 = -0.75
        self.PHI_AMP_SQ = 0.16

    def setIMT(self, imt):
        if imt not in self.supportedImt:
            print(f"The imt {imt} is not supported by Abrahamson, Silva, and Kamai (2014)")
            return None
        self.imt = imt
        self.a1  = self.coeff['a1'][imt]        
        self.a2  = self.coeff['a2'][imt]
        self.a6  = self.coeff['a6'][imt]
        self.a8  = self.coeff['a8'][imt]
        self.a10 = self.coeff['a10'][imt]        
        self.a12 = self.coeff['a12'][imt]
        self.a13 = self.coeff['a13'][imt]
        self.a14 = self.coeff['a14'][imt]
        self.a15 = self.coeff['a15'][imt]        
        self.a17 = self.coeff['a17'][imt]
        self.a43 = self.coeff['a43'][imt]
        self.a44 = self.coeff['a44'][imt]
        self.a45 = self.coeff['a45'][imt]        
        self.a46 = self.coeff['a46'][imt]
        self.b   = self.coeff['b'][imt]
        self.c   = self.coeff['c'][imt]
        self.s1e = self.coeff['s1e'][imt]
        self.s2e = self.coeff['s2e'][imt]
        self.s3  = self.coeff['s3'][imt]        
        self.s4  = self.coeff['s4'][imt]
        self.s1m = self.coeff['s1m'][imt]
        self.s2m = self.coeff['s2m'][imt]
        self.s5  = self.coeff['s5'][imt]        
        self.s6  = self.coeff['s6'][imt]
        self.M1  = self.coeff['M1'][imt]        
        self.Vlin= self.coeff['Vlin'][imt]

    def getV1(self):
        try:
            if self.imt == "PGA" or self.imt == "PGV":
                return 1500.0
            if self.imt >= 3.0:
                return 800.0
            if self.imt > 0.5:
                return np.exp( -0.35 * np.log(self.imt / 0.5) + np.log(1500.0))
            return 1500.0
        except:
            return 1500.0
    def calcZ1ref(self, vs30):
        vsPow4 = vs30 * vs30 * vs30 * vs30
        return np.exp(-7.67 / 4.0 * np.log((vsPow4 + self.A) / self.B)) / 1000.0

    def calcSoilTerm(self, vs30, z1p0):
        if np.isnan(z1p0):
            return 0.0
        z1ref = self.calcZ1ref(vs30)
        vsCoeff = np.array([self.a43, self.a44, self.a45, self.a46, self.a46])
        VS_BINS = np.array([150.0, 250.0, 400.0, 700.0, 1000.0])
        z1c = np.interp(vs30, VS_BINS, vsCoeff)
        return z1c * np.log((z1p0 + 0.01) / (z1ref + 0.01))

    def getPhiA(self, Mw, s1, s2):
        if Mw < 4.0:
            return s1
        if Mw > 6.0:
            return s2
        else:
            return s1 + ((s2 - s1) / 2) * (Mw - 4.0)

    def getTauA(self, Mw, s3, s4):
        if Mw < 5.0:
            return s3
        if Mw > 7.0:
            return s4
        return s3 + ((s4 - s3) / 2) * (Mw - 5.0)
    
    def get_dAmp(self, b, c, vLin, vs30, saRock):
        if (vs30 >= vLin):
            return 0.0
        return (-b * saRock) / (saRock + c) + (b * saRock) / (saRock + c * np.power(vs30 / vLin, self.N))
        
    def calcValues(self, Mw, rJB, rRup, rX, rY0, dip, width, zTop, vs30, vsInferred, z1p0, style):
        if Mw > 5:
            c4mag = self.C4
        elif Mw > 4:
            c4mag = self.C4 - (self.C4 - 1.0) * (5.0 - Mw)
        else:
            c4mag = 1.0
        # -- Equation 3
        R = np.sqrt(rRup * rRup + c4mag * c4mag)
        # -- Equation 2
        MaxMwSq = (8.5 - Mw) * (8.5 - Mw)
        MwM1 = Mw - self.M1

        f1 = self.a1 + self.a17 * rRup
        if Mw > self.M1:
            f1 += self.A5 * MwM1 + self.a8 * MaxMwSq + (self.a2 + self.A3 * MwM1) * np.log(R)
        elif Mw >= self.M2:
            f1 += self.A4 * MwM1 + self.a8 * MaxMwSq + (self.a2 + self.A3 * MwM1) * np.log(R)
        else:
            M2M1 = self.M2 - self.M1
            MaxM2Sq = (8.5 - self.M2) * (8.5 - self.M2)
            MwM2 = Mw - self.M2
            f1 += self.A4 * M2M1 + self.a8 * MaxM2Sq + self.a6 * MwM2 + (self.a2 + self.A3 * M2M1) * np.log(R)
        
        # Hanging Wall Model
        f4 = 0.0
        if rJB < 30 and rX >= 0.0 and Mw > 5.5 and zTop <= 10.0:
            T1 = ((90.0 - dip) / 45 if (dip > 30.0) else 1.33333333)
            dM = Mw - 6.5
            T2 = (1 + self.A2_HW * dM if Mw>=6.5 else 1 + self.A2_HW * dM - (1 - self.A2_HW) * dM * dM)
            T3 = 0.0
            r1 = width * np.cos(dip * np.pi/180.0)
            r2 = 3 * r1
            if rX <= r1:
                rXr1 = rX / r1
                T3 = self.H1 + self.H2 * rXr1 + self.H3 * rXr1 * rXr1
            elif rX <= r2:
                T3 = 1-(rX-r1)/(r2-r1)
            T4 = 1 - (zTop * zTop) / 100.0
            T5 = (1.0 if rJB == 0.0 else 1-rJB/30.0)
            f4 = self.a13 * T1 * T2 * T3 * T4 * T5
        f6 = self.a15
        if zTop < 20.0:
            f6 *= zTop / 20.0
        if style == "NORMAL":
            if Mw > 5.0:
                f78 = self.a12
            else:
                if Mw >= 4.0:
                    f78 = self.a12 * (Mw - 4)
                else:
                    f78 = 0.0
        else:
            f78 = 0.0
        # -- Equation 17
        f10 = self.calcSoilTerm(vs30, z1p0)

        # Site Response Model
        f5 = 0.0
        v1 = self.getV1() # -- Equation 9
        vs30s = (vs30 if (vs30 < v1) else v1) # -- Equation 8

        # Site term -- Equation 7
        saRock = 0.0; # calc Sa1180 (rock reference) if necessary 
        if vs30 < self.Vlin:
            if self.VS_RK < v1:
                vs30s_rk = self.VS_RK
            else:
                vs30s_rk = v1
            f5_rk = (self.a10 + self.b * self.N) * np.log(vs30s_rk / self.Vlin)
            saRock = np.exp(f1 + f78 + f5_rk + f4 + f6)
            f5 = self.a10 * np.log(vs30s / self.Vlin) - self.b * np.log(saRock + self.c) + self.b * np.log(saRock + self.c * pow(vs30s / self.Vlin, self.N))
        else:
            f5 = (self.a10 + self.b * self.N) * np.log(vs30s / self.Vlin)
        # total model (no aftershock f11) -- Equation 1
        mean = f1 + f78 + f5 + f4 + f6 + f10

        # ****** Aleatory uncertainty model ******
        # Intra-event term -- Equation 24
        if vsInferred:
            phiAsq = self.getPhiA(Mw, self.s1e, self.s2e)
        else:
            phiAsq = self.getPhiA(Mw, self.s1m, self.s2m)
        phiAsq *= phiAsq
        # Inter-event term -- Equation 25
        tauB = self.getTauA(Mw, self.s3, self.s4)
        # Intra-event term with site amp variability removed -- Equation 27
        phiBsq = phiAsq - self.PHI_AMP_SQ
        # Parital deriv. of ln(soil amp) w.r.t. ln(SA1180) -- Equation 30
		# saRock subject to same vs30 < Vlin test as in mean model
        dAmp_p1 = self.get_dAmp(self.b, self.c, self.Vlin, vs30, saRock) + 1.0
        # phi squared, with non-linear effects -- Equation 28
        phiSq = phiBsq * dAmp_p1 * dAmp_p1 + self.PHI_AMP_SQ
        #  tau squared, with non-linear effects -- Equation 29
        tau = tauB * dAmp_p1
        # total std dev
        stdDev = np.sqrt(phiSq + tau * tau)

        return mean, stdDev, np.sqrt(phiSq), tau
    
    def getFaultFromRake(self,rake):
        if(rake >= 135 or rake <= -135):
            return "STRIKE_SLIP"
        elif rake>=-45 and rake <=45:
            return "STRIKE_SLIP"
        elif rake>=45 and rake <=135:
            return "REVERSE"
        else:
            return "NORMAL"
    
    def get_IM(self, Mw, site_rup_dict, site_info, im_info):
        vsInf = bool(site_info["vsInferred"])
        style = self.getFaultFromRake(site_rup_dict["aveRake"])
        if 'SA' in im_info['Type']:
            cur_T = im_info.get('Periods', None)
        elif im_info['Type'] == 'PGA':
            cur_T = ['PGA']
        elif im_info['Type'] == 'PGV':
            cur_T = ['PGV']
        else:
            print(f'The IM type {im_info["Type"]} is not supported')
        meanList = []
        stdDevList = []
        InterEvStdDevList = []
        IntraEvStdDevList = []
        for Tj in cur_T:
            start = time.process_time_ns()
            self.setIMT(Tj)
            self.timeSetImt += time.process_time_ns() - start
            start = time.process_time_ns()
            mean, stdDev, InterEvStdDev, IntraEvStdDev = self.calcValues(Mw, site_info["rJB"], site_info["rRup"], site_info["rX"], -1, site_rup_dict["dip"], site_rup_dict["width"], site_rup_dict["zTop"], site_info["vs30"], vsInf, site_info["z1pt0"]/1000.0, style)
            self.timeCalc += time.process_time_ns() - start
            meanList.append(mean)
            stdDevList.append(stdDev)
            InterEvStdDevList.append(InterEvStdDev)
            IntraEvStdDevList.append(IntraEvStdDev)
        saResult = {'Mean': meanList,
                'TotalStdDev': stdDevList,
                'InterEvStdDev': InterEvStdDevList,
                'IntraEvStdDev': IntraEvStdDevList}
        return saResult

