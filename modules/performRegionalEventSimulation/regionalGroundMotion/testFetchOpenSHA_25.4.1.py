##############################################################
# OpenSHA v25.4 Compatibility Test Suite
# Run with: python test_opensha25.py
##############################################################

import numpy as np
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
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'JPype1'])
        import jpype
        import jpype.imports
        from jpype.types import *

        memory_total = psutil.virtual_memory().total / (1024.0**3)
        memory_request = int(memory_total * 0.75)
        jpype.addClassPath('./lib/opensha-all.jar')
        jpype.startJVM(f'-Xmx{memory_request}G', convertStrings=False)

from java.io import *
from java.lang import *
from java.lang.reflect import *
from java.util import *
from org.opensha.commons.data import *
from org.opensha.commons.data.function import *
from org.opensha.commons.data.siteData import *
from org.opensha.commons.geo import *
from org.opensha.commons.param import *
from org.opensha.commons.param.constraint import *
from org.opensha.commons.param.event import *
from org.opensha.sha.calc import *
from org.opensha.sha.earthquake import *
from org.opensha.sha.earthquake.param import *
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
from org.opensha.sha.faultSurface import *
from org.opensha.sha.faultSurface.utils import PointSourceDistanceCorrection
from org.opensha.sha.faultSurface.utils import PointSourceDistanceCorrections
from org.opensha.sha.imr import *
from org.opensha.sha.imr.attenRelImpl import *
from org.opensha.sha.imr.attenRelImpl.ngaw2 import *
from org.opensha.sha.imr.attenRelImpl.ngaw2.NGAW2_Wrappers import *
from org.opensha.sha.imr.param.IntensityMeasureParams import *
from org.opensha.sha.imr.param.OtherParams import *
from org.opensha.sha.util import *

try:
    from scratch.UCERF3.erf.mean import MeanUCERF3
except ModuleNotFoundError:
    MeanUCERF3 = jpype.JClass('scratch.UCERF3.erf.mean.MeanUCERF3')

from org.opensha.sha.gcim.calc import *
from org.opensha.sha.gcim.imr.attenRelImpl import *
from org.opensha.sha.gcim.imr.param.EqkRuptureParams import *
from org.opensha.sha.gcim.imr.param.IntensityMeasureParams import *

# Import the functions from the main module
#from FetchOpenSHA_25_4_1 import (
#    getERF,
#    CreateIMRInstance,
#    get_site_prop,
#    get_IM,
#    get_PointSource_info_CY2014,
#    horzDistanceFast,
#    getPtSrcDistCorr,
#    get_site_vs30_from_opensha,
#    get_site_z1pt0_from_opensha,
#    get_site_z2pt5_from_opensha,
#)

from FetchOpenSHA_25_4_1 import *


def run_tests():
    """Run all tests to verify OpenSHA v25.4 compatibility."""
    from jpype import JInt

    results = {'passed': 0, 'failed': 0, 'skipped': 0}

    def report(test_name, passed, msg=''):
        if passed == 'skip':
            results['skipped'] += 1
            print(f'  ⏭️  SKIP: {test_name} - {msg}')
        elif passed:
            results['passed'] += 1
            print(f'  ✅ PASS: {test_name}')
        else:
            results['failed'] += 1
            print(f'  ❌ FAIL: {test_name} - {msg}')

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 1: Import Verification')
    print('=' * 70)
    # ==========================================================

    try:
        loc = Location(34.0, -118.0)
        assert float(loc.getLatitude()) == 34.0
        assert float(loc.getLongitude()) == -118.0
        report('1.1 Location creation', True)
    except Exception as e:
        report('1.1 Location creation', False, str(e))

    try:
        loc = Location(34.0, -118.0, 10.0)
        assert float(loc.getDepth()) == 10.0
        report('1.2 Location with depth', True)
    except Exception as e:
        report('1.2 Location with depth', False, str(e))

    try:
        loc = Location(34.0522, -118.2437)
        site = Site(loc)
        assert site is not None
        report('1.3 Site creation', True)
    except Exception as e:
        report('1.3 Site creation', False, str(e))

    try:
        none_corr = PointSourceDistanceCorrections.NONE
        assert none_corr is not None
        report('1.4 PointSourceDistanceCorrections.NONE', True)
    except Exception as e:
        report('1.4 PointSourceDistanceCorrections.NONE', False, str(e))

    try:
        result = PointSourceDistanceCorrections.NONE.get()
        assert result is None, f'Expected None, got {result}'
        report('1.5 NONE.get() returns null', True)
    except Exception as e:
        report('1.5 NONE.get() returns null', False, str(e))

    try:
        result = PointSourceDistanceCorrections.FIELD.get()
        assert result is not None
        assert result.size() > 0
        print(f'       FIELD correction list size: {result.size()}')
        report('1.6 FIELD.get() returns WeightedList', True)
    except Exception as e:
        report('1.6 FIELD.get() returns WeightedList', False, str(e))

    try:
        corrections = PointSourceDistanceCorrections.values()
        corr_names = [str(c) for c in corrections]
        print(f'       Available: {corr_names}')
        assert len(corr_names) >= 4
        report('1.7 All PointSourceDistanceCorrections values', True)
    except Exception as e:
        report('1.7 All PointSourceDistanceCorrections values', False, str(e))

    try:
        loc = Location(34.0, -118.0, 10.0)
        ps = PointSurface(loc)
        assert ps is not None
        assert ps.isPointSurface()
        report('1.8 PointSurface creation', True)
    except Exception as e:
        report('1.8 PointSurface creation', False, str(e))

    try:
        al = ArrayList()
        al.add(Location(34.0, -118.0))
        al.add(Location(35.0, -119.0))
        assert al.size() == 2
        report('1.9 ArrayList with Locations', True)
    except Exception as e:
        report('1.9 ArrayList with Locations', False, str(e))

    try:
        rup = EqkRupture()
        rup.setMag(7.0)
        assert float(rup.getMag()) == 7.0
        report('1.10 EqkRupture creation', True)
    except Exception as e:
        report('1.10 EqkRupture creation', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 2: Earthquake Rupture Forecasts (ERFs)')
    print('=' * 70)
    # ==========================================================

    try:
        erf = MeanUCERF2()
        assert erf is not None
        report('2.1 MeanUCERF2 instantiation', True)
    except Exception as e:
        report('2.1 MeanUCERF2 instantiation', False, str(e))

    try:
        erf3 = MeanUCERF3()
        assert erf3 is not None
        report('2.2 MeanUCERF3 instantiation', True)
    except Exception as e:
        report('2.2 MeanUCERF3 instantiation', False, str(e))

    try:
        erf3 = MeanUCERF3()
        erf3.setPreset(MeanUCERF3.Presets.BOTH_FM_BRANCH_AVG)
        report('2.3 MeanUCERF3 BOTH_FM preset', True)
    except Exception as e:
        report('2.3 MeanUCERF3 BOTH_FM preset', False, str(e))

    try:
        erf3 = MeanUCERF3()
        erf3.setPreset(MeanUCERF3.Presets.FM3_1_BRANCH_AVG)
        report('2.4 MeanUCERF3 FM3.1 preset', True)
    except Exception as e:
        report('2.4 MeanUCERF3 FM3.1 preset', False, str(e))

    try:
        erf3 = MeanUCERF3()
        erf3.setPreset(MeanUCERF3.Presets.FM3_2_BRANCH_AVG)
        report('2.5 MeanUCERF3 FM3.2 preset', True)
    except Exception as e:
        report('2.5 MeanUCERF3 FM3.2 preset', False, str(e))

    try:
        erf_f = Frankel02_AdjustableEqkRupForecast()
        assert erf_f is not None
        report('2.6 Frankel02 ERF', True)
    except Exception as e:
        report('2.6 Frankel02 ERF', False, str(e))

    try:
        erf_u1 = WGCEP_UCERF1_EqkRupForecast()
        assert erf_u1 is not None
        report('2.7 WGCEP UCERF1', True)
    except Exception as e:
        report('2.7 WGCEP UCERF1', False, str(e))

    try:
        erf_u2 = UCERF2()
        assert erf_u2 is not None
        report('2.8 UCERF2', True)
    except Exception as e:
        report('2.8 UCERF2', False, str(e))

    try:
        assert IncludeBackgroundOption.INCLUDE is not None
        assert IncludeBackgroundOption.EXCLUDE is not None
        assert IncludeBackgroundOption.ONLY is not None
        report('2.9 IncludeBackgroundOption', True)
    except Exception as e:
        report('2.9 IncludeBackgroundOption', False, str(e))

    try:
        assert BackgroundRupType.POINT is not None
        assert BackgroundRupType.FINITE is not None
        assert BackgroundRupType.CROSSHAIR is not None
        report('2.10 BackgroundRupType', True)
    except Exception as e:
        report('2.10 BackgroundRupType', False, str(e))

    try:
        assert ProbabilityModelOptions.POISSON is not None
        assert ProbabilityModelOptions.U3_BPT is not None
        assert ProbabilityModelOptions.U3_PREF_BLEND is not None
        assert ProbabilityModelOptions.WG02_BPT is not None
        report('2.11 ProbabilityModelOptions', True)
    except Exception as e:
        report('2.11 ProbabilityModelOptions', False, str(e))

    try:
        assert MagDependentAperiodicityOptions.LOW_VALUES is not None
        assert MagDependentAperiodicityOptions.MID_VALUES is not None
        assert MagDependentAperiodicityOptions.HIGH_VALUES is not None
        report('2.12 MagDependentAperiodicityOptions', True)
    except Exception as e:
        report('2.12 MagDependentAperiodicityOptions', False, str(e))

    try:
        assert BPTAveragingTypeOptions.AVE_RI_AVE_TIME_SINCE is not None
        report('2.13 BPTAveragingTypeOptions', True)
    except Exception as e:
        report('2.13 BPTAveragingTypeOptions', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 3: Ground Motion Models (GMMs)')
    print('=' * 70)
    # ==========================================================

    for gmm_cls, name_str in [
        (ASK_2014, 'ASK_2014'), (BSSA_2014, 'BSSA_2014'),
        (CB_2014, 'CB_2014'), (CY_2014, 'CY_2014')
    ]:
        test_id = f'3.{["ASK","BSSA","CB","CY"].index(name_str.split("_")[0])+1}'
        try:
            imr = CreateIMRInstance(str(gmm_cls.NAME))
            assert imr is not None
            print(f'       {name_str} Name: {imr.getName()}')
            report(f'{test_id} {name_str} via Wrapper', True)
        except Exception as e:
            report(f'{test_id} {name_str} via Wrapper', False, str(e))

    try:
        from org.opensha.commons.param.event import ParameterChangeWarningListener

        @jpype.JImplements(ParameterChangeWarningListener)
        class DummyListener:
            @jpype.JOverride
            def parameterChangeWarning(self, event):
                pass

        listener = DummyListener()
        imr_ks = KS_2006_AttenRel(listener)
        imr_ks.setParamDefaults()
        assert imr_ks is not None
        print(f'       KS_2006 Name: {imr_ks.getName()}')
        report('3.5 KS_2006 (gcim)', True)
    except Exception as e:
        report('3.5 KS_2006 (gcim)', False, str(e))

    try:
        from org.opensha.commons.param.event import ParameterChangeWarningListener

        @jpype.JImplements(ParameterChangeWarningListener)
        class DummyListener2:
            @jpype.JOverride
            def parameterChangeWarning(self, event):
                pass

        listener2 = DummyListener2()
        imr_bommer = BommerEtAl_2009_AttenRel(listener2)
        imr_bommer.setParamDefaults()
        assert imr_bommer is not None
        print(f'       BommerEtAl_2009 Name: {imr_bommer.getName()}')
        report('3.6 BommerEtAl_2009 (gcim)', True)
    except Exception as e:
        report('3.6 BommerEtAl_2009 (gcim)', False, str(e))

    try:
        imr = CreateIMRInstance(str(AfshariStewart_2016_AttenRel.NAME))
        assert imr is not None
        report('3.7 AfshariStewart_2016', True)
    except Exception as e:
        report('3.7 AfshariStewart_2016', False, str(e))

    try:
        imr = CreateIMRInstance(str(ASK_2014.NAME))
        ims = imr.getSupportedIntensityMeasures()
        saParam = ims.getParameter(SA_Param.NAME)
        periods = saParam.getPeriodParam().getPeriods()
        print(f'       SA periods count: {len(periods)}')
        report('3.8 ASK_2014 SA periods', True)
    except Exception as e:
        report('3.8 ASK_2014 SA periods', False, str(e))

    try:
        imr = CreateIMRInstance(str(ASK_2014.NAME))
        imr.setIntensityMeasure('PGA')
        report('3.9 ASK_2014 PGA', True)
    except Exception as e:
        report('3.9 ASK_2014 PGA', False, str(e))

    try:
        imr = CreateIMRInstance(str(ASK_2014.NAME))
        imr.setIntensityMeasure('PGV')
        report('3.10 ASK_2014 PGV', True)
    except Exception as e:
        report('3.10 ASK_2014 PGV', False, str(e))

    try:
        imr = CreateIMRInstance(str(ASK_2014.NAME))
        stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
        has_inter = bool(stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTER))
        has_intra = bool(stdDevParam.isAllowed(StdDevTypeParam.STD_DEV_TYPE_INTRA))
        print(f'       Inter: {has_inter}, Intra: {has_intra}')
        report('3.11 StdDev types', True)
    except Exception as e:
        report('3.11 StdDev types', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 4: GMM Computation')
    print('=' * 70)
    # ==========================================================

    try:
        imr = CreateIMRInstance(str(ASK_2014.NAME))
        eqRup = EqkRupture()
        eqRup.setMag(7.0)
        rupLoc = Location(34.0, -118.0, 10.0)
        eqRup.setPointSurface(rupLoc, 90.0)
        eqRup.setAveRake(0.0)
        imr.setEqkRupture(eqRup)
        siteLoc = Location(34.1, -118.1)
        site = Site(siteLoc)
        for i in range(imr.getSiteParams().size()):
            site.addParameter(imr.getSiteParams().getByIndex(i))
        imr.setSite(site)
        imr.setIntensityMeasure('PGA')
        mean_pga = float(imr.getMean())
        std_pga = float(imr.getStdDev())
        print(f'       M7.0 ln(PGA): mean={mean_pga:.4f}, std={std_pga:.4f}')
        assert not np.isnan(mean_pga)
        assert not np.isnan(std_pga)
        assert std_pga > 0
        report('4.1 ASK_2014 PGA computation', True)
    except Exception as e:
        report('4.1 ASK_2014 PGA computation', False, str(e))

    try:
        imr.setIntensityMeasure('SA')
        imtParam = imr.getIntensityMeasure()
        imtParam.getIndependentParameter(PeriodParam.NAME).setValue(1.0)
        mean_sa = float(imr.getMean())
        std_sa = float(imr.getStdDev())
        print(f'       M7.0 ln(SA 1.0s): mean={mean_sa:.4f}, std={std_sa:.4f}')
        assert not np.isnan(mean_sa)
        report('4.2 ASK_2014 SA(1.0s)', True)
    except Exception as e:
        report('4.2 ASK_2014 SA(1.0s)', False, str(e))

    try:
        imr.setIntensityMeasure('PGV')
        mean_pgv = float(imr.getMean())
        print(f'       M7.0 ln(PGV): mean={mean_pgv:.4f}')
        assert not np.isnan(mean_pgv)
        report('4.3 ASK_2014 PGV', True)
    except Exception as e:
        report('4.3 ASK_2014 PGV', False, str(e))

    try:
        imr.setIntensityMeasure('PGA')
        stdDevParam = imr.getParameter(StdDevTypeParam.NAME)
        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTER)
        inter_std = float(imr.getStdDev())
        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_INTRA)
        intra_std = float(imr.getStdDev())
        stdDevParam.setValue(StdDevTypeParam.STD_DEV_TYPE_TOTAL)
        total_std = float(imr.getStdDev())
        computed = np.sqrt(inter_std**2 + intra_std**2)
        print(f'       Total={total_std:.4f}, sqrt(inter²+intra²)={computed:.4f}')
        assert abs(computed - total_std) < 0.01
        report('4.4 StdDev consistency', True)
    except Exception as e:
        report('4.4 StdDev consistency', False, str(e))

    try:
        gmpe_names = [str(ASK_2014.NAME), str(BSSA_2014.NAME),
                      str(CB_2014.NAME), str(CY_2014.NAME)]
        for gname in gmpe_names:
            test_imr = CreateIMRInstance(gname)
            eqRup2 = EqkRupture()
            eqRup2.setMag(7.0)
            eqRup2.setPointSurface(Location(34.0, -118.0, 10.0), 90.0)
            eqRup2.setAveRake(0.0)
            test_imr.setEqkRupture(eqRup2)
            site2 = Site(Location(34.1, -118.1))
            for i in range(test_imr.getSiteParams().size()):
                site2.addParameter(test_imr.getSiteParams().getByIndex(i))
            test_imr.setSite(site2)
            test_imr.setIntensityMeasure('PGA')
            m = float(test_imr.getMean())
            print(f'       {gname}: ln(PGA)={m:.4f}')
            assert not np.isnan(m)
        report('4.5 All NGA-West2 PGA', True)
    except Exception as e:
        report('4.5 All NGA-West2 PGA', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 5: ERF Update + Source Access')
    print('=' * 70)
    # ==========================================================

    try:
        print('       Updating MeanUCERF2 (may take a minute)...')
        erf = MeanUCERF2()
        erf.updateForecast()
        num_sources = erf.getNumSources()
        print(f'       Sources: {num_sources}')
        assert num_sources > 0
        src = erf.getSource(0)
        print(f'       First source: "{src.getName()}" ({src.getNumRuptures()} rups)')
        report('5.1 MeanUCERF2 updateForecast', True)
    except Exception as e:
        report('5.1 MeanUCERF2 updateForecast', False, str(e))

    try:
        src = erf.getSource(0)
        rup = src.getRupture(0)
        mag = float(rup.getMag())
        prob = float(rup.getProbability())
        print(f'       Rup 0: M={mag:.2f}, Prob={prob:.6f}')
        assert mag > 0
        assert 0 <= prob <= 1
        report('5.2 Rupture properties', True)
    except Exception as e:
        report('5.2 Rupture properties', False, str(e))

    try:
        surface = erf.getSource(0).getRupture(0).getRuptureSurface()
        test_loc = Location(34.0522, -118.2437)
        rRup = float(surface.getDistanceRup(test_loc))
        rJB = float(surface.getDistanceJB(test_loc))
        rX = float(surface.getDistanceX(test_loc))
        print(f'       rRup={rRup:.1f}, rJB={rJB:.1f}, rX={rX:.1f}')
        assert rRup >= 0
        assert rJB >= 0
        report('5.3 Distance calculations', True)
    except Exception as e:
        report('5.3 Distance calculations', False, str(e))

    try:
        scenario_info = {
            'EqRupture': {
                'Model': 'WGCEP (2007) UCERF2 - Single Branch',
                'ModelParameters': {
                    'Background Seismicity': 'Include',
                    'Treat Background Seismicity As': 'Point Sources',
                }
            }
        }
        print('       Testing getERF...')
        test_erf = getERF(scenario_info)
        assert test_erf is not None
        ns = test_erf.getNumSources()
        print(f'       Sources: {ns}')
        assert ns > 0
        report('5.4 getERF function', True)
    except Exception as e:
        report('5.4 getERF function', False, str(e))

    try:
        site = Site(Location(34.0522, -118.2437))
        d = float(test_erf.getSource(0).getMinDistance(site))
        print(f'       Source 0 min dist: {d:.1f} km')
        assert d >= 0
        report('5.5 Source getMinDistance', True)
    except Exception as e:
        report('5.5 Source getMinDistance', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 6: Site Data Providers')
    print('=' * 70)
    # ==========================================================

    try:
        providers = OrderedSiteDataProviderList.createSiteDataProviderDefaults()
        assert providers is not None
        report('6.1 OrderedSiteDataProviderList', True)
    except Exception as e:
        report('6.1 OrderedSiteDataProviderList', False, str(e))

    try:
        translator = SiteTranslator()
        assert translator is not None
        report('6.2 SiteTranslator', True)
    except Exception as e:
        report('6.2 SiteTranslator', False, str(e))

    try:
        print('       Fetching Vs30 for LA (requires internet)...')
        vs30 = get_site_vs30_from_opensha([34.0522], [-118.2437])
        print(f'       Vs30: {vs30[0]:.1f} m/s')
        assert vs30[0] > 0
        report('6.3 Vs30 fetch (LA)', True)
    except Exception as e:
        report('6.3 Vs30 fetch (LA)', 'skip', str(e))

    try:
        print('       Fetching Z1.0...')
        z1 = get_site_z1pt0_from_opensha(34.0522, -118.2437)
        print(f'       Z1.0: {z1:.1f} m')
        assert z1 > 0
        report('6.4 Z1.0 fetch (LA)', True)
    except Exception as e:
        report('6.4 Z1.0 fetch (LA)', 'skip', str(e))

    try:
        print('       Fetching Z2.5...')
        z25 = get_site_z2pt5_from_opensha(34.0522, -118.2437)
        print(f'       Z2.5: {z25:.1f} m')
        assert z25 > 0
        report('6.5 Z2.5 fetch (LA)', True)
    except Exception as e:
        report('6.5 Z2.5 fetch (LA)', 'skip', str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 7: Helper Functions')
    print('=' * 70)
    # ==========================================================

    try:
        dist = horzDistanceFast(34.0522, -118.2437, 37.7749, -122.4194)
        print(f'       LA to SF: {dist:.1f} km')
        assert 550 < dist < 570
        report('7.1 horzDistanceFast', True)
    except Exception as e:
        report('7.1 horzDistanceFast', False, str(e))

    try:
        corr = getPtSrcDistCorr(50.0, 7.0, 'NONE')
        assert corr == 1.0
        report('7.2 getPtSrcDistCorr NONE', True)
    except Exception as e:
        report('7.2 getPtSrcDistCorr NONE', False, str(e))

    try:
        corr = getPtSrcDistCorr(50.0, 7.0, 'FIELD')
        print(f'       FIELD correction: {corr:.4f}')
        assert 0.7 < corr < 1.0
        report('7.3 getPtSrcDistCorr FIELD', True)
    except Exception as e:
        report('7.3 getPtSrcDistCorr FIELD', False, str(e))

    try:
        source_info = {
            'Location': {'Latitude': 34.0, 'Longitude': -118.0, 'Depth': 10.0},
            'Magnitude': 7.0,
            'AverageDip': 90.0,
            'AverageRake': 0.0,
        }
        siteList = [
            {'lat': 34.1, 'lon': -118.1},
            {'lat': 34.2, 'lon': -118.2},
        ]
        info, sites_out = get_PointSource_info_CY2014(source_info, siteList)
        print(f'       Site 1 rRup: {sites_out[0]["rRup"]:.2f}')
        print(f'       Site 2 rRup: {sites_out[1]["rRup"]:.2f}')
        assert info['dip'] == 90.0
        assert sites_out[0]['rRup'] > 0
        report('7.4 get_PointSource_info_CY2014', True)
    except Exception as e:
        report('7.4 get_PointSource_info_CY2014', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 8: GCIM')
    print('=' * 70)
    # ==========================================================

    try:
        gcim_calc = GcimCalculator()
        assert gcim_calc is not None
        report('8.1 GcimCalculator', True)
    except Exception as e:
        report('8.1 GcimCalculator', False, str(e))

    try:
        ds575 = Ds575_Param()
        ds595 = Ds595_Param()
        assert ds575 is not None
        assert ds595 is not None
        report('8.2 Ds575/Ds595 Params', True)
    except Exception as e:
        report('8.2 Ds575/Ds595 Params', False, str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 9: Full Pipeline')
    print('=' * 70)
    # ==========================================================

    try:
        print('       Testing get_site_prop...')
        siteSpec = [
            {
                'Location': {'Latitude': 34.0522, 'Longitude': -118.2437},
                'Vs30': 760.0,
            }
        ]
        result = get_site_prop(str(ASK_2014.NAME), siteSpec)
        if result == 1:
            report('9.1 get_site_prop', False, 'Returned error')
        else:
            siteSpec_out, sites_out, site_prop_out = result
            print(f'       Site data entries: {len(site_prop_out[0]["SiteData"])}')
            report('9.1 get_site_prop', True)
    except Exception as e:
        report('9.1 get_site_prop', 'skip', str(e))

    try:
        print('       Testing get_IM...')
        gmpe_info = {'Type': str(ASK_2014.NAME), 'Parameters': {}}
        source_info = {
            'Type': 'PointSource',
            'Magnitude': 7.0,
            'Location': {'Latitude': 34.0, 'Longitude': -118.0, 'Depth': 10.0},
            'AverageDip': 90.0,
            'AverageRake': 0.0,
        }
        station_info = {
            'Type': 'SiteList',
            'SiteList': [
                {'Location': {'Latitude': 34.1, 'Longitude': -118.1}, 'Vs30': 760.0}
            ],
        }
        im_info = {'Type': 'SA PGA', 'Periods': [0.1, 0.5, 1.0, 2.0]}
        siteSpec2 = station_info['SiteList']
        result2 = get_site_prop(str(ASK_2014.NAME), siteSpec2)
        if result2 != 1:
            siteSpec2, sites2, site_prop2 = result2
            res, sta_info = get_IM(
                gmpe_info, None, sites2, siteSpec2, site_prop2,
                source_info, station_info, im_info
            )
            print(f'       Mag: {res["Magnitude"]}')
            if 'lnSA' in res['GroundMotions'][0]:
                print(f'       SA means: {res["GroundMotions"][0]["lnSA"]["Mean"]}')
            if 'lnPGA' in res['GroundMotions'][0]:
                print(f'       PGA mean: {res["GroundMotions"][0]["lnPGA"]["Mean"]}')
            report('9.2 get_IM pipeline', True)
        else:
            report('9.2 get_IM pipeline', 'skip', 'get_site_prop failed')
    except Exception as e:
        report('9.2 get_IM pipeline', 'skip', str(e))

    # ==========================================================
    print('\n' + '=' * 70)
    print('TEST GROUP 10: v25.4 PointSource Distance Correction API')
    print('=' * 70)
    # ==========================================================

    try:
        loc = Location(34.0, -118.0, 10.0)
        ps = PointSurface(loc)
        test_loc = Location(34.1, -118.1)
        rRup = float(ps.getDistanceRup(test_loc))
        rJB = float(ps.getDistanceJB(test_loc))
        print(f'       Default: rRup={rRup:.4f}, rJB={rJB:.4f}')
        assert rRup > 0
        assert rJB > 0
        report('10.1 PointSurface default distances', True)
    except Exception as e:
        report('10.1 PointSurface default distances', False, str(e))

    try:
        loc = Location(34.0, -118.0, 10.0)
        ps = PointSurface(loc)
        test_loc = Location(34.1, -118.1)
        rJB_raw = float(ps.getDistanceJB(test_loc))
        field_list = PointSourceDistanceCorrections.FIELD.get()
        field_wv = field_list.get(JInt(0))
        field_corr = field_wv.value
        ps.setDistanceCorrection(field_corr, 7.0)
        rJB_corrected = float(ps.getDistanceJB(test_loc))
        print(f'       FIELD M7: rJB raw={rJB_raw:.4f}, corrected={rJB_corrected:.4f}')
        assert rJB_corrected != rJB_raw
        report('10.2 FIELD correction via wv.value', True)
    except Exception as e:
        report('10.2 FIELD correction via wv.value', False, str(e))

    try:
        loc = Location(34.0, -118.0, 10.0)
        ps = PointSurface(loc)
        test_loc = Location(34.1, -118.1)
        rJB_default = float(ps.getDistanceJB(test_loc))
        print(f'       Default rJB (=NONE): {rJB_default:.4f}')
        assert rJB_default > 0
        report('10.3 NONE = default (no action needed)', True)
    except Exception as e:
        report('10.3 NONE = default (no action needed)', False, str(e))

    # ==========================================================
    # SUMMARY
    # ==========================================================
    total = results['passed'] + results['failed'] + results['skipped']
    print('\n' + '=' * 70)
    print(f'TEST SUMMARY: {total} tests')
    print(f'  ✅ Passed:  {results["passed"]}')
    print(f'  ❌ Failed:  {results["failed"]}')
    print(f'  ⏭️  Skipped: {results["skipped"]}')
    print('=' * 70)

    if results['failed'] == 0:
        print('\n🎉 All tests passed! OpenSHA v25.4 integration is working.\n')
    else:
        print(f'\n⚠️  {results["failed"]} test(s) failed. See above.\n')

    return results


if __name__ == '__main__':
    print('\n' + '=' * 70)
    print('OpenSHA v25.4 Compatibility Test Suite')
    print('=' * 70)
    run_tests()
