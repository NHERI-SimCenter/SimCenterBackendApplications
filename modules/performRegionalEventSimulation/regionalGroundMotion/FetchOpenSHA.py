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

import numpy as np
import pandas as pd
import ujson
import socket
import subprocess
import importlib
import sys
import psutil
import GlobalVariable

if 'stampede2' not in socket.gethostname():
    import GlobalVariable
    if GlobalVariable.JVM_started is False:
        GlobalVariable.JVM_started = True
        if importlib.util.find_spec('jpype') is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'JPype1'])  # noqa: S603
        import jpype
        # from jpype import imports
        import jpype.imports
        from jpype.types import *  # noqa: F403
        memory_total = psutil.virtual_memory().total / (1024.0**3)
        memory_request = int(memory_total * 0.75)
        jpype.addClassPath('./lib/OpenSHA-1.5.2.jar')
        jpype.startJVM(f'-Xmx{memory_request}G', convertStrings=False)
from java.io import *  # noqa: F403
from java.lang import *  # noqa: F403
from java.lang.reflect import *  # noqa: F403
from java.util import *  # noqa: F403
from org.opensha.commons.data import *  # noqa: F403
from org.opensha.commons.data.function import *  # noqa: F403
from org.opensha.commons.data.siteData import *  # noqa: F403
from org.opensha.commons.geo import *  # noqa: F403
from org.opensha.commons.param import *  # noqa: F403
from org.opensha.commons.param.constraint import *  # noqa: F403
from org.opensha.commons.param.event import *  # noqa: F403
from org.opensha.sha.calc import *  # noqa: F403
from org.opensha.sha.earthquake import *  # noqa: F403
from org.opensha.sha.earthquake.param import *  # noqa: F403
from org.opensha.sha.earthquake.rupForecastImpl.Frankel02 import (
    Frankel02_AdjustableEqkRupForecast,
)
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF1 import (
    WGCEP_UCERF1_EqkRupForecast,
)
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final import UCERF2
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final.MeanUCERF2 import (
    MeanUCERF2,
)
from org.opensha.sha.faultSurface import *  # noqa: F403
from org.opensha.sha.faultSurface.utils import PtSrcDistCorr
from org.opensha.sha.imr import *  # noqa: F403
from org.opensha.sha.imr.attenRelImpl import *  # noqa: F403
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *  # noqa: F403
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *  # noqa: F403
from org.opensha.sha.imr.param.IntensityMeasureParams import *  # noqa: F403
from org.opensha.sha.imr.param.OtherParams import *  # noqa: F403
from org.opensha.sha.util import *  # noqa: F403
from tqdm import tqdm

try:
    from scratch.UCERF3.erf.mean import MeanUCERF3
except ModuleNotFoundError:
    MeanUCERF3 = jpype.JClass('scratch.UCERF3.erf.mean.MeanUCERF3')  # noqa: F405

from org.opensha.sha.gcim.calc import *  # noqa: F403
from org.opensha.sha.gcim.imr.attenRelImpl import *  # noqa: F403
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *  # noqa: F403
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *  # noqa: F403


def getERF(scenario_info, update_flag=True):  # noqa: FBT002, C901, N802, D103
    # Initialization
    erf = None
    erf_name = scenario_info['EqRupture']['Model']
    erf_selection = scenario_info['EqRupture']['ModelParameters']
    # ERF model options
    if erf_name == 'WGCEP (2007) UCERF2 - Single Branch':
        erf = MeanUCERF2()
        if (erf_selection.get('Background Seismicity', None) == 'Exclude') and (
            'Treat Background Seismicity As' in erf_selection.keys()  # noqa: SIM118
        ):
            value = erf_selection.pop('Treat Background Seismicity As')
            print(  # noqa: T201
                f'Background Seismicvity is set as Excluded, Treat Background Seismicity As: {value} is ignored'
            )
        for key, value in erf_selection.items():
            if type(value) is int:
                value = float(value)  # noqa: PLW2901
            erf.setParameter(key, value)
            # erf.getParameter(key).setValue(value)
    elif erf_name == 'USGS/CGS 2002 Adj. Cal. ERF':
        erf = Frankel02_AdjustableEqkRupForecast()
    elif erf_name == 'WGCEP UCERF 1.0 (2005)':
        erf = WGCEP_UCERF1_EqkRupForecast()
    elif erf_name == 'Mean UCERF3':
        tmp = MeanUCERF3()
        if (
            erf_selection.get('preset', None)
            == '(POISSON ONLY) Both FM Branch Averaged'
        ):
            tmp.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
            if (erf_selection.get('Background Seismicity', None) == 'Exclude') and (
                'Treat Background Seismicity As' in erf_selection.keys()  # noqa: SIM118
            ):
                value = erf_selection.pop('Treat Background Seismicity As')
                print(  # noqa: T201
                    f'Background Seismicvity is set as Excluded, Treat Background Seismicity As: {value} is ignored'
                )
            # Some parameters in MeanUCERF3 have overloaded setValue() Need to set one by one
            # Set Apply Aftershock Filter
            if erf_selection.get('Apply Aftershock Filter', None):
                tmp.setParameter(
                    'Apply Aftershock Filter',
                    erf_selection['Apply Aftershock Filter'],
                )
            # Set Aleatoiry mag-area stdDev
            if erf_selection.get('Aleatory Mag-Area StdDev', None):
                tmp.setParameter(
                    'Aleatory Mag-Area StdDev',
                    erf_selection['Aleatory Mag-Area StdDev'],
                )
            # Set IncludeBackgroundOpetion
            setERFbackgroundOptions(tmp, erf_selection)
            # Set Treat Background Seismicity As Option
            setERFtreatBackgroundOptions(tmp, erf_selection)
        elif erf_selection.get('preset', None) == 'FM3.1 Branch Averaged':
            tmp.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
            if (erf_selection.get('Background Seismicity', None) == 'Exclude') and (
                'Treat Background Seismicity As' in erf_selection.keys()  # noqa: SIM118
            ):
                value = erf_selection.pop('Treat Background Seismicity As')
                print(  # noqa: T201
                    f'Background Seismicvity is set as Excluded, Treat Background Seismicity As: {value} is ignored'
                )
            # Some parameters in MeanUCERF3 have overloaded setValue() Need to set one by one
            # Set Apply Aftershock Filter
            if erf_selection.get('Apply Aftershock Filter', None):
                tmp.setParameter(
                    'Apply Aftershock Filter',
                    erf_selection['Apply Aftershock Filter'],
                )
            # Set Aleatoiry mag-area stdDev
            if erf_selection.get('Aleatory Mag-Area StdDev', None):
                tmp.setParameter(
                    'Aleatory Mag-Area StdDev',
                    erf_selection['Aleatory Mag-Area StdDev'],
                )
            # Set IncludeBackgroundOpetion
            setERFbackgroundOptions(tmp, erf_selection)
            # Set Treat Background Seismicity As Option
            setERFtreatBackgroundOptions(tmp, erf_selection)
            # Set Probability Model Option
            setERFProbabilityModelOptions(tmp, erf_selection)
        elif erf_selection.get('preset', None) == 'FM3.2 Branch Averaged':
            tmp.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
            if (erf_selection.get('Background Seismicity', None) == 'Exclude') and (
                'Treat Background Seismicity As' in erf_selection.keys()  # noqa: SIM118
            ):
                value = erf_selection.pop('Treat Background Seismicity As')
                print(  # noqa: T201
                    f'Background Seismicvity is set as Excluded, Treat Background Seismicity As: {value} is ignored'
                )
            # Some parameters in MeanUCERF3 have overloaded setValue() Need to set one by one
            # Set Apply Aftershock Filter
            if erf_selection.get('Apply Aftershock Filter', None):
                tmp.setParameter(
                    'Apply Aftershock Filter',
                    erf_selection['Apply Aftershock Filter'],
                )
            # Set Aleatoiry mag-area stdDev
            if erf_selection.get('Aleatory Mag-Area StdDev', None):
                tmp.setParameter(
                    'Aleatory Mag-Area StdDev',
                    erf_selection['Aleatory Mag-Area StdDev'],
                )
            # Set IncludeBackgroundOpetion
            setERFbackgroundOptions(tmp, erf_selection)
            # Set Treat Background Seismicity As Option
            setERFtreatBackgroundOptions(tmp, erf_selection)
            # Set Probability Model Option
            setERFProbabilityModelOptions(tmp, erf_selection)
        else:
            print(  # noqa: T201
                f"""The specified Mean UCERF3 preset {erf_selection.get("preset", None)} is not implemented"""
            )
        erf = tmp
        del tmp
    elif erf_name == 'WGCEP Eqk Rate Model 2 ERF':
        erf = UCERF2()
    else:
        print('Please check the ERF model name.')  # noqa: T201

    if erf_name and update_flag:
        erf.updateForecast()
    # return
    return erf


def setERFbackgroundOptions(erf, selection):  # noqa: N802, D103
    option = selection.get('Background Seismicity', None)
    if option == 'Include':
        erf.setParameter('Background Seismicity', IncludeBackgroundOption.INCLUDE)  # noqa: F405
    elif option == 'Exclude':
        erf.setParameter('Background Seismicity', IncludeBackgroundOption.EXCLUDE)  # noqa: F405
    elif option == 'Only':
        erf.setParameter('Background Seismicity', IncludeBackgroundOption.ONLY)  # noqa: F405


def setERFtreatBackgroundOptions(erf, selection):  # noqa: N802, D103
    option = selection.get('Treat Background Seismicity As', None)
    if option is None:
        pass
    elif option == 'Point Sources':
        erf.setParameter('Treat Background Seismicity As', BackgroundRupType.POINT)  # noqa: F405
    elif option == 'Single Random Strike Faults':
        erf.setParameter('Treat Background Seismicity As', BackgroundRupType.FINITE)  # noqa: F405
    elif option == 'Two Perpendicular Faults':
        erf.setParameter(
            'Treat Background Seismicity As',
            BackgroundRupType.CROSSHAIR,  # noqa: F405
        )


def setERFProbabilityModelOptions(erf, selection):  # noqa: N802, D103
    option = selection.get('Probability Model', None)
    if option is None:
        pass
    elif option == 'Poisson':
        erf.setParameter('Probability Model', ProbabilityModelOptions.POISSON)  # noqa: F405
    elif option == 'UCERF3 BPT':
        erf.setParameter('Probability Model', ProbabilityModelOptions.U3_BPT)  # noqa: F405
        erf.setParameter(
            'Historic Open Interval', selection.get('Historic Open Interval')
        )
        setERFMagDependentAperiodicityOptions(erf, selection)
        setERFBPTAveragingTypeOptions(erf, selection)
    elif option == 'UCERF3 Preferred Blend':
        erf.setParameter('Probability Model', ProbabilityModelOptions.U3_PREF_BLEND)  # noqa: F405
        erf.setParameter(
            'Historic Open Interval', selection.get('Historic Open Interval')
        )
        setERFBPTAveragingTypeOptions(erf, selection)
    elif option == 'WG02 BPT':
        erf.setParameter('Probability Model', ProbabilityModelOptions.WG02_BPT)  # noqa: F405
        erf.setParameter(
            'Historic Open Interval', selection.get('Historic Open Interval')
        )
        setERFMagDependentAperiodicityOptions(erf, selection)


def setERFMagDependentAperiodicityOptions(erf, selection):  # noqa: C901, N802, D103
    option = selection.get('Aperiodicity', None)
    if option is None:
        pass
    elif option == '0.4,0.3,0.2,0.1':
        erf.setParameter('Aperiodicity', MagDependentAperiodicityOptions.LOW_VALUES)  # noqa: F405
    elif option == '0.5,0.4,0.3,0.2':
        erf.setParameter('Aperiodicity', MagDependentAperiodicityOptions.MID_VALUES)  # noqa: F405
    elif option == '0.6,0.5,0.4,0.3':
        erf.setParameter('Aperiodicity', MagDependentAperiodicityOptions.HIGH_VALUES)  # noqa: F405
    elif option == 'All 0.1':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT1_VALUES,  # noqa: F405
        )
    elif option == 'All 0.2':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT2_VALUES,  # noqa: F405
        )
    elif option == 'All 0.3':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT3_VALUES,  # noqa: F405
        )
    elif option == 'All 0.4':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT4_VALUES,  # noqa: F405
        )
    elif option == 'All 0.5':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT5_VALUES,  # noqa: F405
        )
    elif option == 'All 0.6':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT6_VALUES,  # noqa: F405
        )
    elif option == 'All 0.7':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT7_VALUES,  # noqa: F405
        )
    elif option == 'All 0.8':
        erf.setParameter(
            'Aperiodicity',
            MagDependentAperiodicityOptions.ALL_PT8_VALUES,  # noqa: F405
        )


def setERFBPTAveragingTypeOptions(erf, selection):  # noqa: N802, D103
    option = selection.get('BPT Averaging Type', None)
    if option is None:
        pass
    elif option == 'AveRI and AveTimeSince':
        erf.setParameter(
            'BPT Averaging Type',
            BPTAveragingTypeOptions.AVE_RI_AVE_TIME_SINCE,  # noqa: F405
        )
    elif option == 'AveRI and AveNormTimeSince':
        erf.setParameter(
            'BPT Averaging Type',
            BPTAveragingTypeOptions.AVE_RI_AVE_NORM_TIME_SINCE,  # noqa: F405
        )
    elif option == 'AveRate and AveNormTimeSince':
        erf.setParameter(
            'BPT Averaging Type',
            BPTAveragingTypeOptions.AVE_RATE_AVE_NORM_TIME_SINCE,  # noqa: F405
        )


def get_source_rupture(erf, source_index, rupture_index):  # noqa: D103
    rupSource = erf.getSource(source_index)  # noqa: N806
    ruptures = rupSource.getRuptureList()
    rupture = ruptures.get(rupture_index)
    return rupSource, rupture


def get_source_distance(erf, source_index, lat, lon):  # noqa: D103
    rupSource = erf.getSource(source_index)  # noqa: N806
    sourceSurface = rupSource.getSourceSurface()  # noqa: N806
    # print(lon)
    # print(lat)
    distToSource = []  # noqa: N806
    for i in range(len(lat)):
        distToSource.append(  # noqa: PERF401
            float(sourceSurface.getDistanceRup(Location(lat[i], lon[i])))  # noqa: F405
        )

    return distToSource


def get_rupture_distance(erf, source_index, rupture_index, lat, lon):  # noqa: D103
    rupSource = erf.getSource(source_index)  # noqa: N806
    rupSurface = rupSource.getRupture(rupture_index).getRuptureSurface()  # noqa: N806
    distToRupture = []  # noqa: N806
    for i in range(len(lat)):
        distToRupture.append(  # noqa: PERF401
            float(rupSurface.getDistanceRup(Location(lat[i], lon[i])))  # noqa: F405
        )

    return distToRupture


def get_rupture_info_CY2014(erf, source_index, rupture_index, siteList):  # noqa: N802, N803, D103
    rupSource = erf.getSource(source_index)  # noqa: N806
    rupList = rupSource.getRuptureList()  # noqa: N806
    rupSurface = rupList.get(rupture_index).getRuptureSurface()  # noqa: N806
    if rupList.get(rupture_index).getHypocenterLocation() is None:
        # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/nshmp2/imr/ngaw2/NSHMP14_WUS_CB.java#L242
        dip = float(rupSurface.getAveDip())
        width = float(rupSurface.getAveWidth())
        zTop = float(rupSurface.getAveRupTopDepth())  # noqa: N806
        zHyp = zTop + np.sin(dip / 180.0 * np.pi) * width / 2.0  # noqa: N806
    else:
        zHyp = rupList.get(rupture_index).getHypocenterLocation().getDepth()  # noqa: N806
    for i in range(len(siteList)):
        siteList[i].update(
            {
                'rRup': float(
                    rupSurface.getDistanceRup(
                        Location(siteList[i]['lat'], siteList[i]['lon'])  # noqa: F405
                    )
                )
            }
        )
        siteList[i].update(
            {
                'rJB': float(
                    rupSurface.getDistanceJB(
                        Location(siteList[i]['lat'], siteList[i]['lon'])  # noqa: F405
                    )
                )
            }
        )
        siteList[i].update(
            {
                'rX': float(
                    rupSurface.getDistanceX(
                        Location(siteList[i]['lat'], siteList[i]['lon'])  # noqa: F405
                    )
                )
            }
        )
    site_rup_info = {
        'dip': float(rupSurface.getAveDip()),
        'width': float(rupSurface.getAveWidth()),
        'zTop': float(rupSurface.getAveRupTopDepth()),
        'aveRake': float(rupList.get(rupture_index).getAveRake()),
        'zHyp': zHyp,
    }
    return site_rup_info, siteList


def horzDistanceFast(lat1, lon1, lat2, lon2):  # noqa: N802, D103
    lat1 = lat1 / 180 * np.pi
    lon1 = lon1 / 180 * np.pi
    lat2 = lat2 / 180 * np.pi
    lon2 = lon2 / 180 * np.pi
    dlon = np.abs(lon2 - lon1)
    dlat = np.abs(lat2 - lat1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    EARTH_RADIUS_MEAN = 6371.0072  # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/commons/geo/GeoTools.java#L22  # noqa: N806
    # return EARTH_RADIUS_MEAN * np.sqrt((dLat * dLat) + (dLon * dLon))
    return EARTH_RADIUS_MEAN * c


def getPtSrcDistCorr(horzDist, mag, type):  # noqa: A002, N802, N803, D103
    # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/sha/faultSurface/utils/PtSrcDistCorr.java#L20
    if type == 'FIELD':
        rupLen = np.power(10.0, -3.22 + 0.69 * mag)  # noqa: N806
        return 0.7071 + (1.0 - 0.7071) / (
            1 + np.power(rupLen / (horzDist * 0.87), 1.1)
        )
    elif type == 'NSHMP08':  # noqa: RET505
        print(  # noqa: T201
            'The NSHMP08 rJB correction has not been implemented. corr=1.0 is used instead'
        )
        # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/sha/faultSurface/utils/PtSrcDistCorr.java#L20
        return 1.0
    else:
        return 1.0


def get_PointSource_info_CY2014(source_info, siteList):  # noqa: N802, N803, D103
    # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/sha/faultSurface/PointSurface.java#L118
    sourceLat = source_info['Location']['Latitude']  # noqa: N806
    sourceLon = source_info['Location']['Longitude']  # noqa: N806
    sourceDepth = source_info['Location']['Depth']  # noqa: N806
    for i in range(len(siteList)):
        siteLat = siteList[i]['lat']  # noqa: N806
        siteLon = siteList[i]['lon']  # noqa: N806
        horiD = horzDistanceFast(sourceLat, sourceLon, siteLat, siteLon)  # noqa: N806
        rJB = horiD * getPtSrcDistCorr(horiD, source_info['Magnitude'], 'NONE')  # noqa: N806
        rRup = np.sqrt(rJB**2 + sourceDepth**2)  # noqa: N806
        rX = 0.0  # noqa: N806
        siteList[i].update({'rRup': rRup})
        siteList[i].update({'rJB': rJB})
        siteList[i].update({'rX': rX})
    site_rup_info = {
        'dip': float(source_info['AverageDip']),
        'width': 0.0,  # https://github.com/opensha/opensha/blob/master/src/main/java/org/opensha/sha/faultSurface/PointSurface.java#L68
        'zTop': sourceDepth,
        'aveRake': float(source_info['AverageRake']),
    }
    return site_rup_info, siteList


def export_to_json(  # noqa: C901, D103
    erf,
    site_loc,
    outfile=None,
    EqName=None,  # noqa: N803
    minMag=0.0,  # noqa: N803
    maxMag=10.0,  # noqa: N803
    maxDistance=1000.0,  # noqa: N803
    use_hdf5=False,  # noqa: FBT002
):
    # Initializing
    erf_data = {'type': 'FeatureCollection'}
    site_loc = Location(site_loc[0], site_loc[1])  # type: ignore # noqa: F405
    site = Site(site_loc)  # type: ignore # noqa: F405
    # Total source number
    num_sources = erf.getNumSources()
    source_tag = []
    source_dist = []
    for i in tqdm(range(num_sources), desc=f'Find sources with in {maxDistance} km'):
        rup_source = erf.getSource(i)
        distance_to_source = rup_source.getMinDistance(site)
        # sourceSurface = rupSource.getSourceSurface()
        # distanceToSource = sourceSurface.getDistanceRup(site_loc)
        source_tag.append(i)
        source_dist.append(distance_to_source)
    df = pd.DataFrame.from_dict({'sourceID': source_tag, 'sourceDist': source_dist})  # noqa: PD901
    # Sorting sources
    source_collection = df.sort_values(['sourceDist'], ascending=(True))
    source_collection = source_collection[
        source_collection['sourceDist'] < maxDistance
    ]
    #Collecting source features
    if not use_hdf5:
        feature_collection = []
        for i in tqdm(range(source_collection.shape[0]), desc=f'Find ruptures with in {maxDistance} km'):
            source_index = source_collection.iloc[i, 0]
            # Getting rupture distances
            rupSource = erf.getSource(source_index)  # noqa: N806
            try:
                rupList = rupSource.getRuptureList()  # noqa: N806
            except:  # noqa: E722
                numOfRup = rupSource.getNumRuptures()  # noqa: N806
                rupList = []  # noqa: N806
                for n in range(numOfRup):
                    rupList.append(rupSource.getRupture(n))
                rupList = ArrayList(rupList)  # noqa: N806, F405
            rup_tag = []
            rup_dist = []
            for j in range(rupList.size()):
                ruptureSurface = rupList.get(j).getRuptureSurface()  # noqa: N806
                # If pointsource rupture distance correction
                if isinstance(ruptureSurface, PointSurface):  # noqa: F405
                    # or 'FIELD' or 'NSHMP08'
                    distCorrType = PtSrcDistCorr.Type.NONE  # noqa: N806
                    (PointSurface @ ruptureSurface).setDistCorrMagAndType(  # noqa: F405
                        rupList.get(j).getMag(), distCorrType
                    )
                cur_dist = ruptureSurface.getDistanceRup(site_loc)
                rup_tag.append(j)
                if cur_dist < maxDistance:
                    rup_dist.append(cur_dist)
                else:
                    # exceeding the maxDistance requirement
                    rup_dist.append(-1.0)
            df = pd.DataFrame.from_dict({'rupID': rup_tag, 'rupDist': rup_dist})  # noqa: PD901
            # Sorting
            rup_collection = df.sort_values(['rupDist'], ascending=(True))
            # Preparing the dict of ruptures
            for j in range(rupList.size()):
                cur_dict = dict()  # noqa: C408
                cur_dict.update({'type': 'Feature'})
                rup_index = rup_collection.iloc[j, 0]
                cur_dist = rup_collection.iloc[j, 1]
                if cur_dist <= 0.0:
                    # skipping ruptures with distance exceeding the maxDistance
                    continue
                rupture = rupList.get(rup_index)
                maf = rupture.getMeanAnnualRate(erf.getTimeSpan().getDuration())
                if maf <= 0.0:
                    continue
                ruptureSurface = rupture.getRuptureSurface()  # noqa: N806
                # Properties
                cur_dict['properties'] = dict()  # noqa: C408
                name = str(rupSource.getName())
                if EqName is not None:
                    if EqName not in name:
                        continue
                cur_dict['properties'].update({'Name': name})
                Mag = float(rupture.getMag())  # noqa: N806
                if (Mag < minMag) or (Mag > maxMag):
                    continue
                cur_dict['properties'].update({'Magnitude': Mag})
                cur_dict['properties'].update({'Rupture': int(rup_index)})
                cur_dict['properties'].update({'Source': int(source_index)})
                if outfile is not None:
                    # these calls are time-consuming, so only run them if one needs
                    # detailed outputs of the sources
                    cur_dict['properties'].update({'Distance': float(cur_dist)})
                    distanceRup = rupture.getRuptureSurface().getDistanceRup(site_loc)  # noqa: N806
                    cur_dict['properties'].update({'DistanceRup': float(distanceRup)})
                    distanceSeis = rupture.getRuptureSurface().getDistanceSeis(site_loc)  # noqa: N806
                    cur_dict['properties'].update({'DistanceSeis': float(distanceSeis)})
                    distanceJB = rupture.getRuptureSurface().getDistanceJB(site_loc)  # noqa: N806
                    cur_dict['properties'].update({'DistanceJB': float(distanceJB)})
                    distanceX = rupture.getRuptureSurface().getDistanceX(site_loc)  # noqa: N806
                    cur_dict['properties'].update({'DistanceX': float(distanceX)})
                    Prob = rupture.getProbability()  # noqa: N806
                    cur_dict['properties'].update({'Probability': float(Prob)})
                    maf = rupture.getMeanAnnualRate(erf.getTimeSpan().getDuration())
                    cur_dict['properties'].update({'MeanAnnualRate': abs(float(maf))})
                    # Geometry
                    cur_dict['geometry'] = dict()  # noqa: C408
                    if ruptureSurface.isPointSurface():
                        # Point source
                        pointSurface = ruptureSurface  # noqa: N806
                        location = pointSurface.getLocation()
                        cur_dict['geometry'].update({'type': 'Point'})
                        cur_dict['geometry'].update(
                            {
                                'coordinates': [
                                    float(location.getLongitude()),
                                    float(location.getLatitude()),
                                ]
                            }
                        )
                    else:
                        # Line source
                        try:
                            trace = ruptureSurface.getUpperEdge()
                        except:  # noqa: E722
                            trace = ruptureSurface.getEvenlyDiscritizedUpperEdge()
                        coordinates = []
                        for k in trace:
                            coordinates.append(  # noqa: PERF401
                                [float(k.getLongitude()), float(k.getLatitude())]
                            )
                        cur_dict['geometry'].update({'type': 'LineString'})
                        cur_dict['geometry'].update({'coordinates': coordinates})
                # Appending
                feature_collection.append(cur_dict)
        # sort the list
        maf_list_n = [-x['properties']['MeanAnnualRate'] for x in feature_collection]
        sort_ids = np.argsort(maf_list_n)
        feature_collection_sorted = [feature_collection[i] for i in sort_ids]
        del feature_collection
        erf_data.update({'features': feature_collection_sorted})
        print(  # noqa: T201
            f'FetchOpenSHA: total {len(feature_collection_sorted)} ruptures are collected.'
        )
        if outfile is not None:
            print(  # noqa: T201
                f'The collected ruptures are sorted by MeanAnnualRate and saved in {outfile}'
            )
            with open(outfile, 'w') as f:  # noqa: PTH123
                ujson.dump(erf_data, f, indent=2)
    else:
        import h5py
        with h5py.File(outfile, 'w') as h5file:
            # Store the geometry as a string array
            h5file.create_dataset('geometry', data=gdf.geometry.astype(str).values.astype('S'))
    # return
    return erf_data


def CreateIMRInstance(gmpe_name):  # noqa: N802, D103
    # GMPE name map
    gmpe_map = {
        str(ASK_2014.NAME): ASK_2014_Wrapper.class_.getName(),  # noqa: F405
        str(BSSA_2014.NAME): BSSA_2014_Wrapper.class_.getName(),  # noqa: F405
        str(CB_2014.NAME): CB_2014_Wrapper.class_.getName(),  # noqa: F405
        str(CY_2014.NAME): CY_2014_Wrapper.class_.getName(),  # noqa: F405
        str(KS_2006_AttenRel.NAME): KS_2006_AttenRel.class_.getName(),  # noqa: F405
        str(
            BommerEtAl_2009_AttenRel.NAME  # noqa: F405
        ): BommerEtAl_2009_AttenRel.class_.getName(),  # noqa: F405
        str(
            AfshariStewart_2016_AttenRel.NAME  # noqa: F405
        ): AfshariStewart_2016_AttenRel.class_.getName(),  # noqa: F405
    }
    # Mapping GMPE name
    imrClassName = gmpe_map.get(gmpe_name)  # noqa: N806
    if imrClassName is None:
        return imrClassName
    # Getting the java class
    imrClass = Class.forName(imrClassName)  # noqa: N806, F405
    ctor = imrClass.getConstructor()
    imr = ctor.newInstance()
    # Setting default parameters
    imr.setParamDefaults()
    # return
    return imr


def get_DataSource(paramName, siteData):  # noqa: N802, N803, D103
    typeMap = SiteTranslator.DATA_TYPE_PARAM_NAME_MAP  # noqa: N806, F405
    for dataType in typeMap.getTypesForParameterName(paramName):  # noqa: N806
        if dataType == SiteData.TYPE_VS30:  # noqa: F405
            for dataValue in siteData:  # noqa: N806
                if dataValue.getDataType() != dataType:
                    continue
                vs30 = Double(dataValue.getValue())  # noqa: F405
                if (not vs30.isNaN()) and (vs30 > 0.0):
                    return dataValue.getSourceName()
        elif (dataType == SiteData.TYPE_DEPTH_TO_1_0) or (  # noqa: F405, PLR1714
            dataType == SiteData.TYPE_DEPTH_TO_2_5  # noqa: F405
        ):
            for dataValue in siteData:  # noqa: N806
                if dataValue.getDataType() != dataType:
                    continue
                depth = Double(dataValue.getValue())  # noqa: F405
                if (not depth.isNaN()) and (depth > 0.0):
                    return dataValue.getSourceName()
    return 1


def get_site_prop(gmpe_name, siteSpec):  # noqa: C901, N803, D103
    # GMPE
    try:
        imr = CreateIMRInstance(gmpe_name)
    except:  # noqa: E722
        print('Please check GMPE name.')  # noqa: T201
        return 1
    # Site data
    sites = ArrayList()  # noqa: F405
    for cur_site in siteSpec:
        cur_loc = Location(  # noqa: F405
            cur_site['Location']['Latitude'], cur_site['Location']['Longitude']
        )
        sites.add(Site(cur_loc))  # noqa: F405
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()  # noqa: N806, F405
    try:
        availableSiteData = siteDataProviders.getAllAvailableData(sites)  # noqa: N806
    except:  # noqa: E722
        availableSiteData = []  # noqa: N806
        print(  # noqa: T201
            'remote getAllAvailableData is not available temporarily, will use site Vs30 in the site csv file.'
        )
        # return 1
    siteTrans = SiteTranslator()  # noqa: N806, F405
    # Looping over all sites
    site_prop = []
    for i in range(len(siteSpec)):
        site_tmp = dict()  # noqa: C408
        # Current site
        site = sites.get(i)
        # Location
        cur_site = siteSpec[i]
        locResults = {  # noqa: N806
            'Latitude': cur_site['Location']['Latitude'],
            'Longitude': cur_site['Location']['Longitude'],
        }
        cur_loc = Location(  # noqa: F405
            cur_site['Location']['Latitude'], cur_site['Location']['Longitude']
        )
        siteDataValues = ArrayList()  # noqa: N806, F405
        for j in range(len(availableSiteData)):
            siteDataValues.add(availableSiteData.get(j).getValue(i))
        imrSiteParams = imr.getSiteParams()  # noqa: N806
        siteDataResults = []  # noqa: N806
        # Setting site parameters
        for j in range(imrSiteParams.size()):
            siteParam = imrSiteParams.getByIndex(j)  # noqa: N806
            newParam = Parameter.clone(siteParam)  # noqa: N806, F405
            if siteDataValues.size() > 0:
                siteDataFound = siteTrans.setParameterValue(newParam, siteDataValues)  # noqa: N806
            else:
                siteDataFound = False  # noqa: N806
            if str(newParam.getName()) == 'Vs30' and bool(
                cur_site.get('Vs30', None)
            ):
                newParam.setValue(Double(cur_site['Vs30']))  # noqa: F405
                siteDataResults.append(
                    {
                        'Type': 'Vs30',
                        'Value': float(newParam.getValue()),
                        'Source': 'User Defined',
                    }
                )
            elif str(newParam.getName()) == 'Vs30 Type' and bool(
                cur_site.get('Vs30', None)
            ):
                newParam.setValue('Measured')
                siteDataResults.append(
                    {
                        'Type': 'Vs30 Type',
                        'Value': 'Measured',
                        'Source': 'User Defined',
                    }
                )
            elif siteDataFound:
                provider = 'Unknown'
                provider = get_DataSource(newParam.getName(), siteDataValues)
                if 'String' in str(type(newParam.getValue())):
                    tmp_value = str(newParam.getValue())
                elif 'Double' in str(type(newParam.getValue())):
                    tmp_value = float(newParam.getValue())
                    if str(newParam.getName()) == 'Vs30':
                        cur_site.update({'Vs30': tmp_value})
                else:
                    tmp_value = str(newParam.getValue())
                siteDataResults.append(
                    {
                        'Type': str(newParam.getName()),
                        'Value': tmp_value,
                        'Source': str(provider),
                    }
                )
            else:
                newParam.setValue(siteParam.getDefaultValue())
                siteDataResults.append(
                    {
                        'Type': str(siteParam.getName()),
                        'Value': siteParam.getDefaultValue(),
                        'Source': 'Default',
                    }
                )
            site.addParameter(newParam)
            # End for j
        # Updating site specifications
        siteSpec[i] = cur_site
        site_tmp.update({'Location': locResults, 'SiteData': siteDataResults})
        site_prop.append(site_tmp)

    # Return
    return siteSpec, sites, site_prop


def get_IM(  # noqa: C901, N802, D103
    gmpe_info,
    erf,
    sites,
    siteSpec,  # noqa: N803
    site_prop,
    source_info,
    station_info,
    im_info,
):
    # GMPE name
    gmpe_name = gmpe_info['Type']
    # Creating intensity measure relationship instance
    try:
        imr = CreateIMRInstance(gmpe_name)
    except:  # noqa: E722
        print('Please check GMPE name.')  # noqa: T201
        return 1, station_info
    # Getting supported intensity measure types
    ims = imr.getSupportedIntensityMeasures()
    saParam = ims.getParameter(SA_Param.NAME)  # noqa: N806, F405
    supportedPeriods = saParam.getPeriodParam().getPeriods()  # noqa: N806
    Arrays.sort(supportedPeriods)  # noqa: F405
    # Rupture
    eqRup = EqkRupture()  # noqa: N806, F405
    if source_info['Type'] == 'PointSource':
        eqRup.setMag(source_info['Magnitude'])
        eqRupLocation = Location(  # noqa: N806, F405
            source_info['Location']['Latitude'],
            source_info['Location']['Longitude'],
            source_info['Location']['Depth'],
        )
        eqRup.setPointSurface(eqRupLocation, source_info['AverageDip'])
        eqRup.setAveRake(source_info['AverageRake'])
        magnitude = source_info['Magnitude']
        meanAnnualRate = None  # noqa: N806
    elif source_info['Type'] == 'ERF':
        timeSpan = TimeSpan(TimeSpan.NONE, TimeSpan.YEARS)  # noqa: N806, F405
        erfParams = source_info.get('Parameters', None)  # noqa: N806
        # Additional parameters (if any)
        if erfParams is not None:
            for k in erfParams.keys:
                erf.setParameter(k, erfParams[k])
        # Time span
        timeSpan = erf.getTimeSpan()  # noqa: N806
        # Source
        eqSource = erf.getSource(source_info['SourceIndex'])  # noqa: N806
        eqSource.getName()
        # Rupture
        eqRup = eqSource.getRupture(source_info['RuptureIndex'])  # noqa: N806
        # Properties
        magnitude = eqRup.getMag()
        averageDip = eqRup.getRuptureSurface().getAveDip()  # noqa: N806, F841
        averageRake = eqRup.getAveRake()  # noqa: N806, F841
        # Probability
        probEqRup = eqRup  # noqa: N806
        probability = probEqRup.getProbability()  # noqa: F841
        # MAF
        meanAnnualRate = probEqRup.getMeanAnnualRate(timeSpan.getDuration())  # noqa: N806
        # Rupture surface
        surface = eqRup.getRuptureSurface()  # noqa: F841
    # Setting up imr
    imr.setEqkRupture(eqRup)
    imrParams = gmpe_info['Parameters']  # noqa: N806
    if bool(imrParams):
        for k in imrParams.keys():  # noqa: SIM118
            imr.getParameter(k).setValue(imrParams[k])
    # Station
    if station_info['Type'] == 'SiteList':
        siteSpec = station_info['SiteList']  # noqa: N806
    # Intensity measure
    periods = im_info.get('Periods', None)
    if periods is not None:
        periods = supportedPeriods
    tag_SA = False  # noqa: N806
    tag_PGA = False  # noqa: N806
    tag_PGV = False  # noqa: N806
    tag_Ds575 = False  # noqa: N806, F841
    tag_Ds595 = False  # noqa: N806, F841
    if 'SA' in im_info['Type']:
        tag_SA = True  # noqa: N806
    if 'PGA' in im_info['Type']:
        tag_PGA = True  # noqa: N806
    if 'PGV' in im_info['Type']:
        tag_PGV = True  # noqa: N806
    # Looping over sites
    gm_collector = []
    for i in range(len(siteSpec)):
        gmResults = site_prop[i]  # noqa: N806
        # Current site
        site = sites.get(i)
        # Location
        cur_site = siteSpec[i]  # noqa: F841
        # Set up the site in the imr
        imr.setSite(site)
        try:
            stdDevParam = imr.getParameter(StdDevTypeParam.NAME)  # noqa: N806, F405
            hasIEStats = stdDevParam.isAllowed(  # noqa: N806
                StdDevTypeParam.STD_DEV_TYPE_INTER  # noqa: F405
            ) and stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA)  # noqa: F405
        except:  # noqa: E722
            stdDevParaam = None  # noqa: N806, F841
            hasIEStats = False  # noqa: N806
        cur_T = im_info.get('Periods', None)  # noqa: N806
        if tag_SA:
            saResult = {'Mean': [], 'TotalStdDev': []}  # noqa: N806
            if hasIEStats:
                saResult.update({'InterEvStdDev': []})
                saResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure('SA')
            imtParam = imr.getIntensityMeasure()  # noqa: N806
            for Tj in cur_T:  # noqa: N806
                imtParam.getIndependentParameter(PeriodParam.NAME).setValue(  # noqa: F405
                    float(Tj)
                )
                mean = imr.getMean()
                saResult['Mean'].append(float(mean))
                if stdDevParam is not None:
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_TOTAL)  # noqa: F405
                stdDev = imr.getStdDev()  # noqa: N806
                saResult['TotalStdDev'].append(float(stdDev))
                if hasIEStats:
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)  # noqa: F405
                    interEvStdDev = imr.getStdDev()  # noqa: N806
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)  # noqa: F405
                    intraEvStdDev = imr.getStdDev()  # noqa: N806
                    saResult['InterEvStdDev'].append(float(interEvStdDev))
                    saResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnSA': saResult})
        if tag_PGA:
            # for PGV current T = 0
            cur_T = [0.00]  # noqa: N806
            pgaResult = {'Mean': [], 'TotalStdDev': []}  # noqa: N806
            if hasIEStats:
                pgaResult.update({'InterEvStdDev': []})
                pgaResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure('PGA')
            mean = imr.getMean()
            pgaResult['Mean'].append(float(mean))
            stdDev = imr.getStdDev()  # noqa: N806
            pgaResult['TotalStdDev'].append(float(stdDev))
            if hasIEStats:
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)  # noqa: F405
                interEvStdDev = imr.getStdDev()  # noqa: N806
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)  # noqa: F405
                intraEvStdDev = imr.getStdDev()  # noqa: N806
                pgaResult['InterEvStdDev'].append(float(interEvStdDev))
                pgaResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnPGA': pgaResult})
        if tag_PGV:
            # for PGV current T = 0
            cur_T = [0.00]  # noqa: N806
            pgvResult = {'Mean': [], 'TotalStdDev': []}  # noqa: N806
            if hasIEStats:
                pgvResult.update({'InterEvStdDev': []})
                pgvResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure('PGV')
            mean = imr.getMean()
            pgvResult['Mean'].append(float(mean))
            stdDev = imr.getStdDev()  # noqa: N806
            pgvResult['TotalStdDev'].append(float(stdDev))
            if hasIEStats:
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)  # noqa: F405
                interEvStdDev = imr.getStdDev()  # noqa: N806
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)  # noqa: F405
                intraEvStdDev = imr.getStdDev()  # noqa: N806
                pgvResult['InterEvStdDev'].append(float(interEvStdDev))
                pgvResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnPGV': pgvResult})

        gm_collector.append(gmResults)
    # Updating station information
    if station_info['Type'] == 'SiteList':
        station_info.update({'SiteList': siteSpec})
    # Final results
    res = {
        'Magnitude': magnitude,
        'MeanAnnualRate': meanAnnualRate,
        'SiteSourceDistance': source_info.get('SiteSourceDistance', None),
        'SiteRuptureDistance': source_info.get('SiteRuptureDistance', None),
        'Periods': cur_T,
        'GroundMotions': gm_collector,
    }
    # return
    return res, station_info


def get_site_vs30_from_opensha(lat, lon, vs30model='CGS/Wills VS30 Map (2015)'):  # noqa: D103
    # set up site java object
    sites = ArrayList()  # noqa: F405
    num_sites = len(lat)
    for i in range(num_sites):
        sites.add(Site(Location(lat[i], lon[i])))  # noqa: F405

    # prepare site data java object
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()  # noqa: N806, F405
    siteData = siteDataProviders.getAllAvailableData(sites)  # noqa: N806

    # search name
    vs30 = []
    for i in range(int(siteData.size())):
        cur_siteData = siteData.get(i)  # noqa: N806
        if str(cur_siteData.getSourceName()) == vs30model:
            vs30 = [
                float(cur_siteData.getValue(x).getValue()) for x in range(num_sites)
            ]
            break
        else:  # noqa: RET508
            continue

    # check if any nan (Wills Map return nan for offshore sites)
    # Using global vs30 as default patch - 'Global Vs30 from Topographic Slope (Wald & Allen 2008)'
    if any([np.isnan(x) for x in vs30]):  # noqa: C419
        non_list = np.where(np.isnan(vs30))[0].tolist()
        for i in non_list:
            vs30[i] = float(siteData.get(3).getValue(i).getValue())

    # return
    return vs30


def get_site_z1pt0_from_opensha(lat, lon):  # noqa: D103
    sites = ArrayList()  # noqa: F405
    sites.add(Site(Location(lat, lon)))  # noqa: F405
    # prepare site data java object
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()  # noqa: N806, F405
    siteData = siteDataProviders.getAllAvailableData(sites)  # noqa: N806
    for data in siteData:
        if data.getValue(0).getDataType() == 'Depth to Vs = 1.0 km/sec':
            z1pt0 = float(data.getValue(0).getValue())
            if not np.isnan(z1pt0):
                break
    return z1pt0 * 1000.0


def get_site_z2pt5_from_opensha(lat, lon):  # noqa: D103
    sites = ArrayList()  # noqa: F405
    sites.add(Site(Location(lat, lon)))  # noqa: F405
    # prepare site data java object
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()  # noqa: N806, F405
    siteData = siteDataProviders.getAllAvailableData(sites)  # noqa: N806
    for data in siteData:
        if data.getValue(0).getDataType() == 'Depth to Vs = 2.5 km/sec':
            z2pt5 = float(data.getValue(0).getValue())
            if not np.isnan(z2pt5):
                break
    return z2pt5 * 1000.0
