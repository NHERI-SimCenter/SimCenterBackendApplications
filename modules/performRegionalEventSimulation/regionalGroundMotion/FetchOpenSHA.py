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

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from java.io import *
from java.lang import *
from java.lang.reflect import *
from java.util import *

from org.opensha.commons.data import *
from org.opensha.commons.data.siteData import *
from org.opensha.commons.data.function import *
from org.opensha.commons.exceptions import ParameterException
from org.opensha.commons.geo import *
from org.opensha.commons.param import *
from org.opensha.commons.param.event import *
from org.opensha.commons.param.constraint import *
from org.opensha.commons.util import ServerPrefUtils

from org.opensha.sha.earthquake import *
from org.opensha.sha.earthquake.param import *
from org.opensha.sha.earthquake.rupForecastImpl.Frankel02 import Frankel02_AdjustableEqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF1 import WGCEP_UCERF1_EqkRupForecast
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final import UCERF2
from org.opensha.sha.earthquake.rupForecastImpl.WGCEP_UCERF_2_Final.MeanUCERF2 import MeanUCERF2
from org.opensha.sha.faultSurface import *
from org.opensha.sha.imr import *
from org.opensha.sha.imr.attenRelImpl import *
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
from org.opensha.sha.imr.param.IntensityMeasureParams import *
from org.opensha.sha.imr.param.OtherParams import *
from org.opensha.sha.imr.param.SiteParams import Vs30_Param
from org.opensha.sha.calc import *
from org.opensha.sha.util import *
try:
    from scratch.UCERF3.erf.mean import MeanUCERF3
except ModuleNotFoundError:
    MeanUCERF3 = jpype.JClass("scratch.UCERF3.erf.mean.MeanUCERF3")

from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.calc import *


def getERF(erf_name, update_flag):

    # Initialization
    erf = None
    # ERF model options
    if erf_name == 'WGCEP (2007) UCERF2 - Single Branch':
        erf = MeanUCERF2()
    elif erf_name == 'USGS/CGS 2002 Adj. Cal. ERF':
        erf = Frankel02_AdjustableEqkRupForecast()
    elif erf_name == 'WGCEP UCERF 1.0 (2005)':
        erf = WGCEP_UCERF1_EqkRupForecast()
    elif erf_name == 'Mean UCERF3':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'Mean UCERF3 FM3.1':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'Mean UCERF3 FM3.2':
        tmp = MeanUCERF3()
        tmp.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
        erf = tmp
        del tmp
    elif erf_name == 'WGCEP Eqk Rate Model 2 ERF':
        erf = UCERF2()
    else:
        print('Please check the ERF model name.')

    if erf_name and update_flag:
        erf.updateForecast()
    # return
    return erf


def get_source_distance(erf, source_index, lat, lon):

    rupSource = erf.getSource(source_index)
    sourceSurface = rupSource.getSourceSurface()
    print(lon)
    print(lat)
    distToSource = []
    for i in range(len(lat)):
        distToSource.append(float(sourceSurface.getDistanceRup(Location(lat[i], lon[i]))))

    return distToSource


def export_to_json(erf, site_loc, outfile = None, EqName = None, minMag = 0.0, maxMag = 10.0, maxDistance = 1000.0, maxSources = 500):

    # Initializing
    erf_data = {"type": "FeatureCollection"}
    site_loc = Location(site_loc[0], site_loc[1])
    # Total source number
    num_sources = erf.getNumSources()
    source_tag = []
    source_dist = []
    for i in range(num_sources):
        rupSource = erf.getSource(i)
        sourceSurface = rupSource.getSourceSurface()
        distanceToSource = sourceSurface.getDistanceRup(site_loc)
        source_tag.append(i)
        source_dist.append(distanceToSource)
    df = pd.DataFrame.from_dict({
            'sourceID': source_tag,
            'sourceDist': source_dist
        })
    # Sorting sources
    source_collection = df.sort_values(['sourceDist'], ascending = (True))
    # Collecting source features
    maxSources = min(maxSources, num_sources)
    feature_collection = []
    for i in tqdm(range(maxSources), desc='Sources'):
        source_index = source_collection.iloc[i, 0]
        distanceToSource = source_collection.iloc[i, 1]
        # Checking maximum distance
        if (distanceToSource > maxDistance):
            break
        # Getting rupture distances
        rupSource = erf.getSource(source_index)
        try:
            rupList = rupSource.getRuptureList()
        except:
            continue
        rup_tag = []
        rup_dist = []
        for j in range(rupList.size()):
            rupture = rupList.get(j)
            cur_dist = rupture.getRuptureSurface().getDistanceRup(site_loc)
            rup_tag.append(j)
            rup_dist.append(cur_dist)
        df = pd.DataFrame.from_dict({
            'rupID': rup_tag,
            'rupDist': rup_dist
        })
        # Sorting
        rup_collection = df.sort_values(['rupDist'], ascending = (True))
        # Preparing the dict of ruptures
        for j in range(rupList.size()):
            cur_dict = dict()
            cur_dict.update({'type': 'Feature'})
            rup_index = rup_collection.iloc[j, 0]
            cur_dist = rup_collection.iloc[j, 1]
            rupture = rupList.get(rup_index)
            maf = rupture.getMeanAnnualRate(erf.getTimeSpan().getDuration())
            if maf <= 0.:
                continue
            ruptureSurface = rupture.getRuptureSurface()
            # Properties
            cur_dict['properties'] = dict()
            name = str(rupSource.getName())
            if (EqName is not None):
                if (EqName not in name):
                    continue
            cur_dict['properties'].update({'Name': name})
            Mag = float(rupture.getMag())
            if (Mag < minMag) or (Mag > maxMag):
                continue
            cur_dict['properties'].update({'Magnitude': Mag})
            cur_dict['properties'].update({'Rupture': int(rup_index)})
            cur_dict['properties'].update({'Source': int(source_index)})
            if outfile is not None:
                # these calls are time-consuming, so only run them if one needs
                # detailed outputs of the sources
                cur_dict['properties'].update({'Distance': float(cur_dist)})
                distanceRup = rupture.getRuptureSurface().getDistanceRup(site_loc)
                cur_dict['properties'].update({'DistanceRup': float(distanceRup)})
                distanceSeis = rupture.getRuptureSurface().getDistanceSeis(site_loc)
                cur_dict['properties'].update({'DistanceSeis': float(distanceSeis)})
                distanceJB = rupture.getRuptureSurface().getDistanceJB(site_loc)
                cur_dict['properties'].update({'DistanceJB': float(distanceJB)})
                distanceX = rupture.getRuptureSurface().getDistanceX(site_loc)
                cur_dict['properties'].update({'DistanceX': float(distanceX)})
                Prob = rupture.getProbability()
                cur_dict['properties'].update({'Probability': float(Prob)})
                maf = rupture.getMeanAnnualRate(erf.getTimeSpan().getDuration())
                cur_dict['properties'].update({'MeanAnnualRate': abs(float(maf))})
                # Geometry
                cur_dict['geometry'] = dict()
                if (ruptureSurface.isPointSurface()):
                    # Point source
                    pointSurface = ruptureSurface
                    location = pointSurface.getLocation()
                    cur_dict['geometry'].update({'type': 'Point'})
                    cur_dict['geometry'].update({'coordinates': [float(location.getLongitude()), float(location.getLatitude())]})
                else:
                    # Line source
                    try:
                        trace = ruptureSurface.getUpperEdge()
                    except:
                        trace = ruptureSurface.getEvenlyDiscritizedUpperEdge()
                    coordinates = []
                    for k in trace:
                        coordinates.append([float(k.getLongitude()), float(k.getLatitude())])
                    cur_dict['geometry'].update({'type': 'LineString'})
                    cur_dict['geometry'].update({'coordinates': coordinates})
            # Appending
            feature_collection.append(cur_dict)
        # end for j
    # end for i
    erf_data.update({'features': feature_collection})
    # Output
    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(erf_data, f, indent=2)
    # return
    return erf_data


def CreateIMRInstance(gmpe_name):

    # GMPE name map
    gmpe_map = {str(ASK_2014.NAME): ASK_2014_Wrapper.class_.getName(),
                str(BSSA_2014.NAME): BSSA_2014_Wrapper.class_.getName(),
                str(CB_2014.NAME): CB_2014_Wrapper.class_.getName(),
                str(CY_2014.NAME): CY_2014_Wrapper.class_.getName(),
                str(KS_2006_AttenRel.NAME): KS_2006_AttenRel.class_.getName(),
                str(BommerEtAl_2009_AttenRel.NAME): BommerEtAl_2009_AttenRel.class_.getName(),
                str(AfshariStewart_2016_AttenRel.NAME): AfshariStewart_2016_AttenRel.class_.getName()}
    # Mapping GMPE name
    imrClassName = gmpe_map.get(gmpe_name, None)
    if imrClassName is None:
        return imrClassName
    # Getting the java class
    imrClass = Class.forName(imrClassName)
    ctor = imrClass.getConstructor()
    imr = ctor.newInstance()
    # Setting default parameters
    imr.setParamDefaults()
    # return
    return imr


def get_DataSource(paramName, siteData):
    typeMap = SiteTranslator.DATA_TYPE_PARAM_NAME_MAP
    for dataType in typeMap.getTypesForParameterName(paramName):
        if dataType == SiteData.TYPE_VS30:
            for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                vs30 = Double(dataValue.getValue())
                if (not vs30.isNaN()) and (vs30 > 0.0):
                    return dataValue.getSourceName()
        elif (dataType == SiteData.TYPE_DEPTH_TO_1_0) or (dataType == SiteData.TYPE_DEPTH_TO_2_5):
             for dataValue in siteData:
                if dataValue.getDataType() != dataType:
                    continue
                depth = Double(dataValue.getValue())
                if (not depth.isNaN()) and (depth > 0.0):
                    return dataValue.getSourceName()
    return 1


def get_site_prop(gmpe_name, siteSpec):

    # GMPE
    try:
        imr = CreateIMRInstance(gmpe_name)
    except:
        print('Please check GMPE name.')
        return 1
    # Site data
    sites = ArrayList()
    for cur_site in siteSpec:
        cur_loc = Location(cur_site['Location']['Latitude'], cur_site['Location']['Longitude'])
        sites.add(Site(cur_loc))
    siteDataProviders = OrderedSiteDataProviderList.createSiteDataProviderDefaults()
    try:
        availableSiteData = siteDataProviders.getAllAvailableData(sites)
    except:
        print('Error in getAllAvailableData')
        return 1
    siteTrans = SiteTranslator()
    # Looping over all sites
    site_prop = []
    for i in range(len(siteSpec)):
        site_tmp = dict()
        # Current site
        site = sites.get(i)
        # Location
        cur_site = siteSpec[i]
        locResults = {'Latitude': cur_site['Location']['Latitude'],
                      'Longitude': cur_site['Location']['Longitude']}
        cur_loc = Location(cur_site['Location']['Latitude'], cur_site['Location']['Longitude'])
        siteDataValues = ArrayList()
        for j in range(len(availableSiteData)):
            siteDataValues.add(availableSiteData.get(j).getValue(i))
        imrSiteParams = imr.getSiteParams()
        siteDataResults = []
        # Setting site parameters
        for j in range(imrSiteParams.size()):
            siteParam = imrSiteParams.getByIndex(j)
            newParam = Parameter.clone(siteParam)
            siteDataFound = siteTrans.setParameterValue(newParam, siteDataValues)
            if (str(newParam.getName())=='Vs30' and bool(cur_site.get('Vs30', None))):
                newParam.setValue(Double(cur_site['Vs30']))
                siteDataResults.append({'Type': 'Vs30',
                                        'Value': float(newParam.getValue()),
                                        'Source': 'User Defined'})
            elif (str(newParam.getName())=='Vs30 Type' and bool(cur_site.get('Vs30', None))):
                newParam.setValue("Measured")
                siteDataResults.append({'Type': 'Vs30 Type',
                                        'Value': 'Measured',
                                        'Source': 'User Defined'})
            elif siteDataFound:
                provider = "Unknown"
                provider = get_DataSource(newParam.getName(), siteDataValues)
                if 'String' in str(type(newParam.getValue())):
                    tmp_value = str(newParam.getValue())
                elif 'Double' in str(type(newParam.getValue())):
                    tmp_value = float(newParam.getValue())
                    if str(newParam.getName())=='Vs30':
                            cur_site.update({'Vs30': tmp_value})
                else:
                    tmp_value = str(newParam.getValue())
                siteDataResults.append({'Type': str(newParam.getName()),
                                        'Value': tmp_value,
                                        'Source': str(provider)})
            else:
                newParam.setValue(siteParam.getDefaultValue())
                siteDataResults.append({'Type': str(siteParam.getName()),
                                        'Value': float(siteParam.getDefaultValue()),
                                        'Source': 'Default'})
            site.addParameter(newParam)
            # End for j
        # Updating site specifications
        siteSpec[i] = cur_site
        site_tmp.update({'Location': locResults,
                         'SiteData': siteDataResults})
        site_prop.append(site_tmp)

    # Return
    return siteSpec, sites, site_prop


def get_IM(gmpe_info, erf, sites, siteSpec, site_prop, source_info, station_info, im_info):

    # GMPE name
    gmpe_name = gmpe_info['Type']
    # Creating intensity measure relationship instance
    try:
        imr = CreateIMRInstance(gmpe_name)
    except:
        print('Please check GMPE name.')
        return 1, station_info
    # Getting supported intensity measure types
    ims = imr.getSupportedIntensityMeasures()
    saParam = ims.getParameter(SA_Param.NAME)
    supportedPeriods = saParam.getPeriodParam().getPeriods()
    Arrays.sort(supportedPeriods)
    # Rupture
    eqRup = EqkRupture()
    if source_info['Type'] == 'PointSource':
        eqRup.setMag(source_info['Magnitude'])
        eqRupLocation = Location(source_info['Location']['Latitude'],
                                 source_info['Location']['Longitude'],
                                 source_info['Location']['Depth'])
        eqRup.setPointSurface(eqRupLocation, source_info['AverageDip'])
        eqRup.setAveRake(source_info['AverageRake'])
        magnitude = source_info['Magnitude']
        meanAnnualRate = None
    elif source_info['Type'] == 'ERF':
        timeSpan = TimeSpan(TimeSpan.NONE, TimeSpan.YEARS)
        erfParams = source_info.get('Parameters', None)
        # Additional parameters (if any)
        if erfParams is not None:
            for k in erfParams.keys:
                erf.setParameter(k, erfParams[k])
        # Time span
        timeSpan = erf.getTimeSpan()
        # Source
        eqSource = erf.getSource(source_info['SourceIndex'])
        eqSource.getName()
        # Rupture
        eqRup = eqSource.getRupture(source_info['RuptureIndex'])
        # Properties
        magnitude = eqRup.getMag()
        averageDip = eqRup.getRuptureSurface().getAveDip()
        averageRake = eqRup.getAveRake()
        # Probability
        probEqRup = eqRup
        probability = probEqRup.getProbability()
        # MAF
        meanAnnualRate = probEqRup.getMeanAnnualRate(timeSpan.getDuration())
        # Rupture surface
        surface = eqRup.getRuptureSurface()
    # Setting up imr
    imr.setEqkRupture(eqRup)
    imrParams = gmpe_info['Parameters']
    if bool(imrParams):
        for k in imrParams.keys():
            imr.getParameter(k).setValue(imrParams[k])
    # Station
    if station_info['Type'] == 'SiteList':
        siteSpec = station_info['SiteList']
    # Intensity measure
    periods = im_info.get('Periods', None)
    if periods is not None:
        periods = supportedPeriods
    tag_SA = False
    tag_PGA = False
    tag_PGV = False
    tag_Ds575 = False
    tag_Ds595 = False
    if 'SA' in im_info['Type']:
        tag_SA = True
    if 'PGA' in im_info['Type']:
        tag_PGA = True
    if 'PGV' in im_info['Type']:
        tag_PGV = True
    if 'Ds575' in im_info['Type']:
        tag_Ds575 = True
    if 'Ds595' in im_info['Type']:
        tag_Ds595 = True
    # Looping over sites
    gm_collector = []
    for i in range(len(siteSpec)):
        gmResults = site_prop[i]
        # Current site
        site = sites.get(i)
        # Location
        cur_site = siteSpec[i]
        # Set up the site in the imr
        imr.setSite(site)
        try:
            stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
            hasIEStats = stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER) and \
                stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA)
        except:
            stdDevParaam = None
            hasIEStats = False
        cur_T = im_info.get('Periods', None)
        if tag_SA:
            saResult = {'Mean': [],
                        'TotalStdDev': []}
            if hasIEStats:
                saResult.update({'InterEvStdDev': []})
                saResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure("SA")
            imtParam = imr.getIntensityMeasure()
            for Tj in cur_T:
                imtParam.getIndependentParameter(PeriodParam.NAME).setValue(float(Tj))
                mean = imr.getMean()
                saResult['Mean'].append(float(mean))
                if stdDevParam is not None:
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_TOTAL)
                stdDev = imr.getStdDev()
                saResult['TotalStdDev'].append(float(stdDev))
                if hasIEStats:
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                    interEvStdDev = imr.getStdDev()
                    stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                    intraEvStdDev = imr.getStdDev()
                    saResult['InterEvStdDev'].append(float(interEvStdDev))
                    saResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnSA': saResult})
        if tag_PGA:
            # for PGV current T = 0
            cur_T = [0.00]
            pgaResult = {'Mean': [],
                        'TotalStdDev': []}
            if hasIEStats:
                pgaResult.update({'InterEvStdDev': []})
                pgaResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure("PGA")
            mean = imr.getMean()
            pgaResult['Mean'].append(float(mean))
            stdDev = imr.getStdDev()
            pgaResult['TotalStdDev'].append(float(stdDev))
            if hasIEStats:
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                interEvStdDev = imr.getStdDev()
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                intraEvStdDev = imr.getStdDev()
                pgaResult['InterEvStdDev'].append(float(interEvStdDev))
                pgaResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnPGA': pgaResult})
        if tag_PGV:
            # for PGV current T = 0
            cur_T = [0.00]
            pgvResult = {'Mean': [],
                        'TotalStdDev': []}
            if hasIEStats:
                pgvResult.update({'InterEvStdDev': []})
                pgvResult.update({'IntraEvStdDev': []})
            imr.setIntensityMeasure("PGV")
            mean = imr.getMean()
            pgvResult['Mean'].append(float(mean))
            stdDev = imr.getStdDev()
            pgvResult['TotalStdDev'].append(float(stdDev))
            if hasIEStats:
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
                interEvStdDev = imr.getStdDev()
                stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
                intraEvStdDev = imr.getStdDev()
                pgvResult['InterEvStdDev'].append(float(interEvStdDev))
                pgvResult['IntraEvStdDev'].append(float(intraEvStdDev))
            gmResults.update({'lnPGV': pgvResult})
        gm_collector.append(gmResults)
    # Updating station information
    if station_info['Type'] == 'SiteList':
        station_info.update({'SiteList': siteSpec})
    # Final results
    res = {'Magnitude': magnitude,
           'MeanAnnualRate': meanAnnualRate,
           'SiteSourceDistance': source_info.get('SiteSourceDistance',None),
           'Periods': cur_T,
           'GroundMotions': gm_collector}
    # return
    return res, station_info
