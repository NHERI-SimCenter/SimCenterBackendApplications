"""The wntrfr.epanet.io module contains methods for reading/writing EPANET input and output files.

.. rubric:: Contents

.. autosummary::

    InpFile
    BinFile

----


"""  # noqa: CPY001

import logging
import re

import numpy as np
import pandas as pd

# from .time_utils import run_lineprofile
import wntrfr.network
import wntrfr.sim
from wntrfr.network.base import Link
from wntrfr.network.controls import (
    Comparison,
    Control,
    SimTimeCondition,
    TimeOfDayCondition,
    ValueCondition,
    _ControlType,  # noqa: PLC2701
)
from wntrfr.network.elements import Junction, Pipe, Pump, Tank, Valve
from wntrfr.network.model import LinkStatus

from .util import (
    EN,
    FlowUnits,
    HydParam,
    MassUnits,
    MixType,
    PressureUnits,
    QualParam,
    QualType,
    ResultType,
    StatisticsType,
    from_si,
    to_si,
)

logger = logging.getLogger(__name__)

_INP_SECTIONS = [
    '[OPTIONS]',
    '[TITLE]',
    '[JUNCTIONS]',
    '[RESERVOIRS]',
    '[TANKS]',
    '[PIPES]',
    '[PUMPS]',
    '[VALVES]',
    '[EMITTERS]',
    '[CURVES]',
    '[PATTERNS]',
    '[ENERGY]',
    '[STATUS]',
    '[CONTROLS]',
    '[RULES]',
    '[DEMANDS]',
    '[QUALITY]',
    '[REACTIONS]',
    '[SOURCES]',
    '[MIXING]',
    '[TIMES]',
    '[REPORT]',
    '[COORDINATES]',
    '[VERTICES]',
    '[LABELS]',
    '[BACKDROP]',
    '[TAGS]',
]

_JUNC_ENTRY = ' {name:20} {elev:15.11g} {dem:15.11g} {pat:24} {com:>3s}\n'
_JUNC_LABEL = '{:21} {:>12s} {:>12s} {:24}\n'

_RES_ENTRY = ' {name:20s} {head:15.11g} {pat:>24s} {com:>3s}\n'
_RES_LABEL = '{:21s} {:>20s} {:>24s}\n'

_TANK_ENTRY = ' {name:20s} {elev:15.11g} {initlev:15.11g} {minlev:15.11g} {maxlev:15.11g} {diam:15.11g} {minvol:15.11g} {curve:20s} {com:>3s}\n'
_TANK_LABEL = '{:21s} {:>20s} {:>20s} {:>20s} {:>20s} {:>20s} {:>20s} {:20s}\n'

_PIPE_ENTRY = ' {name:20s} {node1:20s} {node2:20s} {len:15.11g} {diam:15.11g} {rough:15.11g} {mloss:15.11g} {status:>20s} {com:>3s}\n'
_PIPE_LABEL = '{:21s} {:20s} {:20s} {:>20s} {:>20s} {:>20s} {:>20s} {:>20s}\n'

_PUMP_ENTRY = (
    ' {name:20s} {node1:20s} {node2:20s} {ptype:8s} {params:20s} {com:>3s}\n'
)
_PUMP_LABEL = '{:21s} {:20s} {:20s} {:20s}\n'

_VALVE_ENTRY = ' {name:20s} {node1:20s} {node2:20s} {diam:15.11g} {vtype:4s} {set:15.11g} {mloss:15.11g} {com:>3s}\n'
_GPV_ENTRY = ' {name:20s} {node1:20s} {node2:20s} {diam:15.11g} {vtype:4s} {set:20s} {mloss:15.11g} {com:>3s}\n'
_VALVE_LABEL = '{:21s} {:20s} {:20s} {:>20s} {:4s} {:>20s} {:>20s}\n'

_CURVE_ENTRY = ' {name:10s} {x:12f} {y:12f} {com:>3s}\n'
_CURVE_LABEL = '{:11s} {:12s} {:12s}\n'


def _split_line(line):  # noqa: ANN001, ANN202
    _vc = line.split(';', 1)
    _cmnt = None
    _vals = None
    if len(_vc) == 0:
        pass
    elif len(_vc) == 1:
        _vals = _vc[0].split()
    elif _vc[0] == '':  # noqa: PLC1901
        _cmnt = _vc[1]
    else:
        _vals = _vc[0].split()
        _cmnt = _vc[1]
    return _vals, _cmnt


def _is_number(s):  # noqa: ANN001, ANN202
    """Checks if input is a number

    Parameters
    ----------
    s : anything

    """  # noqa: D400, D401
    try:
        float(s)
        return True  # noqa: TRY300
    except ValueError:
        return False


def _str_time_to_sec(s):  # noqa: ANN001, ANN202
    """Converts EPANET time format to seconds.

    Parameters
    ----------
    s : string
        EPANET time string. Options are 'HH:MM:SS', 'HH:MM', 'HH'


    Returns
    -------
     Integer value of time in seconds.

    """  # noqa: D401
    pattern1 = re.compile(r'^(\d+):(\d+):(\d+)$')
    time_tuple = pattern1.search(s)
    if bool(time_tuple):
        return (
            int(time_tuple.groups()[0]) * 60 * 60
            + int(time_tuple.groups()[1]) * 60
            + int(round(float(time_tuple.groups()[2])))
        )
    else:  # noqa: RET505
        pattern2 = re.compile(r'^(\d+):(\d+)$')
        time_tuple = pattern2.search(s)
        if bool(time_tuple):
            return (
                int(time_tuple.groups()[0]) * 60 * 60
                + int(time_tuple.groups()[1]) * 60
            )
        else:  # noqa: RET505
            pattern3 = re.compile(r'^(\d+)$')
            time_tuple = pattern3.search(s)
            if bool(time_tuple):
                return int(time_tuple.groups()[0]) * 60 * 60
            else:  # noqa: RET505
                raise RuntimeError('Time format in ' 'INP file not recognized. ')  # noqa: DOC501, EM101, TRY003


def _clock_time_to_sec(s, am_pm):  # noqa: ANN001, ANN202, C901
    """Converts EPANET clocktime format to seconds.

    Parameters
    ----------
    s : string
        EPANET time string. Options are 'HH:MM:SS', 'HH:MM', HH'

    am : string
        options are AM or PM


    Returns
    -------
    Integer value of time in seconds

    """  # noqa: D401
    if am_pm.upper() == 'AM':
        am = True
    elif am_pm.upper() == 'PM':
        am = False
    else:
        raise RuntimeError('am_pm option not recognized; options are AM or PM')  # noqa: DOC501, EM101, TRY003

    pattern1 = re.compile(r'^(\d+):(\d+):(\d+)$')
    time_tuple = pattern1.search(s)
    if bool(time_tuple):
        time_sec = (
            int(time_tuple.groups()[0]) * 60 * 60
            + int(time_tuple.groups()[1]) * 60
            + int(round(float(time_tuple.groups()[2])))
        )
        if s.startswith('12'):
            time_sec -= 3600 * 12
        if not am:
            if time_sec >= 3600 * 12:
                raise RuntimeError(  # noqa: DOC501, TRY003
                    'Cannot specify am/pm for times greater than 12:00:00'  # noqa: EM101
                )
            time_sec += 3600 * 12
        return time_sec
    else:  # noqa: RET505
        pattern2 = re.compile(r'^(\d+):(\d+)$')
        time_tuple = pattern2.search(s)
        if bool(time_tuple):
            time_sec = (
                int(time_tuple.groups()[0]) * 60 * 60
                + int(time_tuple.groups()[1]) * 60
            )
            if s.startswith('12'):
                time_sec -= 3600 * 12
            if not am:
                if time_sec >= 3600 * 12:
                    raise RuntimeError(  # noqa: DOC501, TRY003
                        'Cannot specify am/pm for times greater than 12:00:00'  # noqa: EM101
                    )
                time_sec += 3600 * 12
            return time_sec
        else:  # noqa: RET505
            pattern3 = re.compile(r'^(\d+)$')
            time_tuple = pattern3.search(s)
            if bool(time_tuple):
                time_sec = int(time_tuple.groups()[0]) * 60 * 60
                if s.startswith('12'):
                    time_sec -= 3600 * 12
                if not am:
                    if time_sec >= 3600 * 12:
                        raise RuntimeError(  # noqa: DOC501, TRY003
                            'Cannot specify am/pm for times greater than 12:00:00'  # noqa: EM101
                        )
                    time_sec += 3600 * 12
                return time_sec
            else:  # noqa: RET505
                raise RuntimeError('Time format in ' 'INP file not recognized. ')  # noqa: DOC501, EM101, TRY003


def _sec_to_string(sec):  # noqa: ANN001, ANN202
    hours = int(sec / 3600.0)
    sec -= hours * 3600
    mm = int(sec / 60.0)
    sec -= mm * 60
    return (hours, mm, int(sec))


class InpFile(wntrfr.epanet.InpFile):
    """EPANET INP file reader and writer class.

    This class provides read and write functionality for EPANET INP files.
    The EPANET Users Manual provides full documentation for the INP file format.
    """

    def __init__(self):  # noqa: ANN204
        super().__init__()

    def _write_junctions(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[JUNCTIONS]\n'.encode('ascii'))
        f.write(
            _JUNC_LABEL.format(';ID', 'Elevation', 'Demand', 'Pattern').encode(
                'ascii'
            )
        )
        nnames = list(wn.junction_name_list)
        # nnames.sort()
        for junction_name in nnames:
            junction = wn.nodes[junction_name]
            # sina added this
            if junction._is_isolated == True:  # noqa: SLF001, E712
                continue
            if junction.demand_timeseries_list:
                base_demands = junction.demand_timeseries_list.base_demand_list()
                demand_patterns = junction.demand_timeseries_list.pattern_list()
                if base_demands:
                    base_demand = base_demands[0]
                else:
                    base_demand = 0.0
                if demand_patterns:
                    if demand_patterns[0] == wn.options.hydraulic.pattern:
                        demand_pattern = None
                    else:
                        demand_pattern = demand_patterns[0]
                else:
                    demand_pattern = None
            else:
                base_demand = 0.0
                demand_pattern = None
            E = {  # noqa: N806
                'name': junction_name,
                'elev': from_si(
                    self.flow_units, junction.elevation, HydParam.Elevation
                ),
                'dem': from_si(self.flow_units, base_demand, HydParam.Demand),
                'pat': '',
                'com': ';',
            }
            if demand_pattern is not None:
                E['pat'] = str(demand_pattern)
            f.write(_JUNC_ENTRY.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_reservoirs(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[RESERVOIRS]\n'.encode('ascii'))
        f.write(_RES_LABEL.format(';ID', 'Head', 'Pattern').encode('ascii'))
        nnames = list(wn.reservoir_name_list)
        # nnames.sort()
        for reservoir_name in nnames:
            reservoir = wn.nodes[reservoir_name]
            # sina added this
            if reservoir._is_isolated == True:  # noqa: SLF001, E712
                continue
            E = {  # noqa: N806
                'name': reservoir_name,
                'head': from_si(
                    self.flow_units,
                    reservoir.head_timeseries.base_value,
                    HydParam.HydraulicHead,
                ),
                'com': ';',
            }
            if reservoir.head_timeseries.pattern is None:
                E['pat'] = ''
            else:
                E['pat'] = reservoir.head_timeseries.pattern.name
            f.write(_RES_ENTRY.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_tanks(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[TANKS]\n'.encode('ascii'))
        f.write(
            _TANK_LABEL.format(
                ';ID',
                'Elevation',
                'Init Level',
                'Min Level',
                'Max Level',
                'Diameter',
                'Min Volume',
                'Volume Curve',
            ).encode('ascii')
        )
        nnames = list(wn.tank_name_list)
        # nnames.sort()
        for tank_name in nnames:
            tank = wn.nodes[tank_name]
            if tank._is_isolated == True:  # sina added this  # noqa: SLF001, E712
                continue
            E = {  # noqa: N806
                'name': tank_name,
                'elev': from_si(self.flow_units, tank.elevation, HydParam.Elevation),
                'initlev': from_si(
                    self.flow_units, tank.init_level, HydParam.HydraulicHead
                ),
                'minlev': from_si(
                    self.flow_units, tank.min_level, HydParam.HydraulicHead
                ),
                'maxlev': from_si(
                    self.flow_units, tank.max_level, HydParam.HydraulicHead
                ),
                'diam': from_si(
                    self.flow_units, tank.diameter, HydParam.TankDiameter
                ),
                'minvol': from_si(self.flow_units, tank.min_vol, HydParam.Volume),
                'curve': '',
                'com': ';',
            }
            if tank.vol_curve is not None:
                E['curve'] = tank.vol_curve.name
            f.write(_TANK_ENTRY.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_pipes(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[PIPES]\n'.encode('ascii'))
        f.write(
            _PIPE_LABEL.format(
                ';ID',
                'Node1',
                'Node2',
                'Length',
                'Diameter',
                'Roughness',
                'Minor Loss',
                'Status',
            ).encode('ascii')
        )
        lnames = list(wn.pipe_name_list)
        # lnames.sort()
        for pipe_name in lnames:
            pipe = wn.links[pipe_name]
            if pipe._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            E = {  # noqa: N806
                'name': pipe_name,
                'node1': pipe.start_node_name,
                'node2': pipe.end_node_name,
                'len': from_si(self.flow_units, pipe.length, HydParam.Length),
                'diam': from_si(
                    self.flow_units, pipe.diameter, HydParam.PipeDiameter
                ),
                'rough': pipe.roughness,
                'mloss': pipe.minor_loss,
                'status': str(pipe.initial_status),
                'com': ';',
            }
            if pipe.cv:
                E['status'] = 'CV'
            f.write(_PIPE_ENTRY.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_pumps(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[PUMPS]\n'.encode('ascii'))
        f.write(
            _PUMP_LABEL.format(';ID', 'Node1', 'Node2', 'Properties').encode('ascii')
        )
        lnames = list(wn.pump_name_list)
        # lnames.sort()
        for pump_name in lnames:
            pump = wn.links[pump_name]
            if pump._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            E = {  # noqa: N806
                'name': pump_name,
                'node1': pump.start_node_name,
                'node2': pump.end_node_name,
                'ptype': pump.pump_type,
                'params': '',
                #                 'speed_keyword': 'SPEED',
                #                 'speed': pump.speed_timeseries.base_value,
                'com': ';',
            }
            if pump.pump_type == 'HEAD':
                E['params'] = pump.pump_curve_name
            elif pump.pump_type == 'POWER':
                E['params'] = str(
                    from_si(self.flow_units, pump.power, HydParam.Power)
                )
            else:
                raise RuntimeError('Only head or power info is supported of pumps.')  # noqa: EM101, TRY003
            tmp_entry = _PUMP_ENTRY
            if pump.speed_timeseries.base_value != 1:
                E['speed_keyword'] = 'SPEED'
                E['speed'] = pump.speed_timeseries.base_value
                tmp_entry = (
                    tmp_entry.rstrip('\n').rstrip('}').rstrip('com:>3s').rstrip(' {')
                    + ' {speed_keyword:8s} {speed:15.11g} {com:>3s}\n'
                )
            if pump.speed_timeseries.pattern is not None:
                tmp_entry = (
                    tmp_entry.rstrip('\n').rstrip('}').rstrip('com:>3s').rstrip(' {')
                    + ' {pattern_keyword:10s} {pattern:20s} {com:>3s}\n'
                )
                E['pattern_keyword'] = 'PATTERN'
                E['pattern'] = pump.speed_timeseries.pattern.name
            f.write(tmp_entry.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_valves(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[VALVES]\n'.encode('ascii'))
        f.write(
            _VALVE_LABEL.format(
                ';ID', 'Node1', 'Node2', 'Diameter', 'Type', 'Setting', 'Minor Loss'
            ).encode('ascii')
        )
        lnames = list(wn.valve_name_list)
        # lnames.sort()
        for valve_name in lnames:
            valve = wn.links[valve_name]
            if valve._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            E = {  # noqa: N806
                'name': valve_name,
                'node1': valve.start_node_name,
                'node2': valve.end_node_name,
                'diam': from_si(
                    self.flow_units, valve.diameter, HydParam.PipeDiameter
                ),
                'vtype': valve.valve_type,
                'set': valve._initial_setting,  # noqa: SLF001
                'mloss': valve.minor_loss,
                'com': ';',
            }
            valve_type = valve.valve_type
            formatter = _VALVE_ENTRY
            if valve_type in ['PRV', 'PSV', 'PBV']:  # noqa: PLR6201
                valve_set = from_si(
                    self.flow_units,
                    valve._initial_setting,  # noqa: SLF001
                    HydParam.Pressure,
                )
            elif valve_type == 'FCV':
                valve_set = from_si(
                    self.flow_units,
                    valve._initial_setting,  # noqa: SLF001
                    HydParam.Flow,
                )
            elif valve_type == 'TCV':
                valve_set = valve._initial_setting  # noqa: SLF001
            elif valve_type == 'GPV':
                valve_set = valve.headloss_curve_name
                formatter = _GPV_ENTRY
            E['set'] = valve_set
            f.write(formatter.format(**E).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_emitters(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[EMITTERS]\n'.encode('ascii'))
        entry = '{:10s} {:10s}\n'
        label = '{:10s} {:10s}\n'
        f.write(label.format(';ID', 'Flow coefficient').encode('ascii'))
        njunctions = list(wn.junction_name_list)
        # njunctions.sort()
        for junction_name in njunctions:
            junction = wn.nodes[junction_name]
            # Sina added this
            if junction._is_isolated == True:  # noqa: SLF001, E712
                continue
            if junction._emitter_coefficient:  # noqa: SLF001
                val = from_si(
                    self.flow_units,
                    junction._emitter_coefficient,  # noqa: SLF001
                    HydParam.Flow,
                )
                f.write(entry.format(junction_name, str(val)).encode('ascii'))
        f.write('\n'.encode('ascii'))

    # System Operation

    def _write_status(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[STATUS]\n'.encode('ascii'))
        f.write('{:10s} {:10s}\n'.format(';ID', 'Setting').encode('ascii'))
        for link_name, link in wn.links():
            if link._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if isinstance(link, Pipe):
                continue
            if isinstance(link, Pump):
                setting = link.initial_setting
                if type(setting) is float and setting != 1.0:
                    f.write(f'{link_name:10s} {setting:10.10g}\n'.encode('ascii'))
            if link.initial_status == LinkStatus.Closed:
                f.write(
                    f'{link_name:10s} {LinkStatus(link.initial_status).name:10s}\n'.encode(
                        'ascii'
                    )
                )
            if isinstance(link, wntrfr.network.Valve) and link.initial_status in (  # noqa: PLR6201
                LinkStatus.Open,
                LinkStatus.Opened,
            ):
                #           if link.initial_status in (LinkStatus.Closed,):
                f.write(
                    f'{link_name:10s} {LinkStatus(link.initial_status).name:10s}\n'.encode(
                        'ascii'
                    )
                )

    #                if link.initial_status is LinkStatus.Active:
    #                    valve_type = link.valve_type
    #                    if valve_type in ['PRV', 'PSV', 'PBV']:
    #                        setting = from_si(self.flow_units, link.initial_setting, HydParam.Pressure)
    #                    elif valve_type == 'FCV':
    #                        setting = from_si(self.flow_units, link.initial_setting, HydParam.Flow)
    #                    elif valve_type == 'TCV':
    #                        setting = link.initial_setting
    #                    else:
    #                        continue
    #                    continue
    #                elif isinstance(link, wntrfr.network.Pump):
    #                    setting = link.initial_setting
    #                else: continue
    #                f.write('{:10s} {:10.10g}\n'.format(link_name,
    #                        setting).encode('ascii'))
    #        f.write('\n'.encode('ascii'))

    def _write_controls(self, f, wn):  # noqa: ANN001, ANN202, C901
        def get_setting(control_action, control_name):  # noqa: ANN001, ANN202
            value = control_action._value  # noqa: SLF001
            attribute = control_action._attribute.lower()  # noqa: SLF001
            if attribute == 'status':
                setting = LinkStatus(value).name
            elif attribute == 'base_speed':
                setting = str(value)
            elif attribute == 'setting' and isinstance(
                control_action._target_obj,  # noqa: SLF001
                Valve,
            ):
                valve = control_action._target_obj  # noqa: SLF001
                valve_type = valve.valve_type
                if valve_type == 'PRV' or valve_type == 'PSV' or valve_type == 'PBV':  # noqa: PLR1714
                    setting = str(from_si(self.flow_units, value, HydParam.Pressure))
                elif valve_type == 'FCV':
                    setting = str(from_si(self.flow_units, value, HydParam.Flow))
                elif valve_type == 'TCV':
                    setting = str(value)
                elif valve_type == 'GPV':
                    setting = value
                else:
                    raise ValueError('Valve type not recognized' + str(valve_type))
            elif attribute == 'setting':
                setting = value
            else:
                setting = None
                logger.warning(
                    'Could not write control ' + str(control_name) + ' - skipping'  # noqa: G003
                )

            return setting

        f.write('[CONTROLS]\n'.encode('ascii'))
        # Time controls and conditional controls only
        for text, all_control in wn.controls():
            control_action = all_control._then_actions[0]  # noqa: SLF001
            # Sina added this
            if control_action._target_obj._is_isolated == True:  # noqa: SLF001, E712
                continue
            if all_control.epanet_control_type is not _ControlType.rule:
                if (
                    len(all_control._then_actions) != 1  # noqa: SLF001
                    or len(all_control._else_actions) != 0  # noqa: SLF001
                ):
                    logger.error('Too many actions on CONTROL "%s"' % text)  # noqa: G002, UP031
                    raise RuntimeError('Too many actions on CONTROL "%s"' % text)  # noqa: UP031
                if not isinstance(control_action.target()[0], Link):
                    continue
                if isinstance(
                    all_control._condition,  # noqa: SLF001
                    (SimTimeCondition, TimeOfDayCondition),
                ):
                    entry = '{ltype} {link} {setting} AT {compare} {time:g}\n'
                    vals = {
                        'ltype': control_action._target_obj.link_type,  # noqa: SLF001
                        'link': control_action._target_obj.name,  # noqa: SLF001
                        'setting': get_setting(control_action, text),
                        'compare': 'TIME',
                        'time': all_control._condition._threshold / 3600.0,  # noqa: SLF001
                    }
                    if vals['setting'] is None:
                        continue
                    if isinstance(all_control._condition, TimeOfDayCondition):  # noqa: SLF001
                        vals['compare'] = 'CLOCKTIME'
                    f.write(entry.format(**vals).encode('ascii'))
                elif (
                    all_control._condition._source_obj._is_isolated == True  # noqa: SLF001, E712
                ):  # Sina added this
                    continue
                elif isinstance(all_control._condition, (ValueCondition)):  # noqa: SLF001
                    entry = '{ltype} {link} {setting} IF {ntype} {node} {compare} {thresh}\n'
                    vals = {
                        'ltype': control_action._target_obj.link_type,  # noqa: SLF001
                        'link': control_action._target_obj.name,  # noqa: SLF001
                        'setting': get_setting(control_action, text),
                        'ntype': all_control._condition._source_obj.node_type,  # noqa: SLF001
                        'node': all_control._condition._source_obj.name,  # noqa: SLF001
                        'compare': 'above',
                        'thresh': 0.0,
                    }
                    if vals['setting'] is None:
                        continue
                    if all_control._condition._relation in [  # noqa: PLR6201, SLF001
                        np.less,
                        np.less_equal,
                        Comparison.le,
                        Comparison.lt,
                    ]:
                        vals['compare'] = 'below'
                    threshold = all_control._condition._threshold  # noqa: SLF001
                    if isinstance(all_control._condition._source_obj, Tank):  # noqa: SLF001
                        vals['thresh'] = from_si(
                            self.flow_units, threshold, HydParam.HydraulicHead
                        )
                    elif isinstance(all_control._condition._source_obj, Junction):  # noqa: SLF001
                        vals['thresh'] = from_si(
                            self.flow_units, threshold, HydParam.Pressure
                        )
                    else:
                        raise RuntimeError(  # noqa: TRY004
                            'Unknown control for EPANET INP files: %s'  # noqa: UP031
                            % type(all_control)
                        )
                    f.write(entry.format(**vals).encode('ascii'))
                elif not isinstance(all_control, Control):
                    raise RuntimeError(
                        'Unknown control for EPANET INP files: %s'  # noqa: UP031
                        % type(all_control)
                    )
        f.write('\n'.encode('ascii'))

    def _write_rules(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[RULES]\n'.encode('ascii'))
        for text, all_control in wn.controls():  # noqa: B007
            entry = '{}\n'
            if all_control.epanet_control_type == _ControlType.rule:
                # Sina added this begin
                try:
                    if all_control._then_actions[0]._target_obj._is_isolated == True:  # noqa: SLF001, E712
                        continue
                except:  # noqa: S110, E722
                    pass

                try:
                    if all_control.condition._source_obj._is_isolated == True:  # noqa: SLF001, E712
                        continue
                except:  # noqa: S110, E722
                    pass

                # Sina added this end
                rule = _EpanetRule('blah', self.flow_units, self.mass_units)  # noqa: F821
                rule.from_if_then_else(all_control)
                f.write(entry.format(str(rule)).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_demands(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[DEMANDS]\n'.encode('ascii'))
        entry = '{:10s} {:10s} {:10s}{:s}\n'
        label = '{:10s} {:10s} {:10s}\n'
        f.write(label.format(';ID', 'Demand', 'Pattern').encode('ascii'))
        nodes = list(wn.junction_name_list)
        # nodes.sort()
        for node in nodes:
            # Sina added this
            if wn.get_node(node)._is_isolated == True:  # noqa: SLF001, E712
                continue
            demands = wn.get_node(node).demand_timeseries_list
            # leak =
            if len(demands) > 1:
                for ct, demand in enumerate(demands):  # noqa: B007, FURB148
                    cat = str(demand.category)
                    # if cat == 'EN2 base':
                    #    cat = ''
                    if cat.lower() == 'none':
                        cat = ''
                    else:
                        cat = ' ;' + demand.category
                    E = {  # noqa: N806
                        'node': node,
                        'base': from_si(
                            self.flow_units, demand.base_value, HydParam.Demand
                        ),
                        'pat': '',
                        'cat': cat,
                    }
                    if demand.pattern_name in wn.pattern_name_list:
                        E['pat'] = demand.pattern_name
                    f.write(
                        entry.format(
                            E['node'], str(E['base']), E['pat'], E['cat']
                        ).encode('ascii')
                    )
        f.write('\n'.encode('ascii'))

    # Water Quality

    def _write_quality(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[QUALITY]\n'.encode('ascii'))
        entry = '{:10s} {:10s}\n'
        label = '{:10s} {:10s}\n'  # noqa: F841
        nnodes = list(wn.nodes.keys())
        # nnodes.sort()
        for node_name in nnodes:
            node = wn.nodes[node_name]
            if node._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if node.initial_quality:
                if wn.options.quality.mode == 'CHEMICAL':
                    quality = from_si(
                        self.flow_units,
                        node.initial_quality,
                        QualParam.Concentration,
                        mass_units=self.mass_units,
                    )
                elif wn.options.quality.mode == 'AGE':
                    quality = from_si(
                        self.flow_units, node.initial_quality, QualParam.WaterAge
                    )
                else:
                    quality = node.initial_quality
                f.write(entry.format(node_name, str(quality)).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_reactions(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[REACTIONS]\n'.encode('ascii'))
        f.write(
            ';Type           Pipe/Tank               Coefficient\n'.encode('ascii')
        )
        entry_int = ' {:s} {:s} {:d}\n'
        entry_float = ' {:s} {:s} {:<10.4f}\n'
        for tank_name, tank in wn.nodes(Tank):
            if tank._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if tank.bulk_rxn_coeff is not None:
                f.write(
                    entry_float.format(
                        'TANK',
                        tank_name,
                        from_si(
                            self.flow_units,
                            tank.bulk_rxn_coeff,
                            QualParam.BulkReactionCoeff,
                            mass_units=self.mass_units,
                            reaction_order=wn.options.quality.bulk_rxn_order,
                        ),
                    ).encode('ascii')
                )
        for pipe_name, pipe in wn.links(Pipe):
            if pipe._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if pipe.bulk_rxn_coeff is not None:
                f.write(
                    entry_float.format(
                        'BULK',
                        pipe_name,
                        from_si(
                            self.flow_units,
                            pipe.bulk_rxn_coeff,
                            QualParam.BulkReactionCoeff,
                            mass_units=self.mass_units,
                            reaction_order=wn.options.quality.bulk_rxn_order,
                        ),
                    ).encode('ascii')
                )
            if pipe.wall_rxn_coeff is not None:
                f.write(
                    entry_float.format(
                        'WALL',
                        pipe_name,
                        from_si(
                            self.flow_units,
                            pipe.wall_rxn_coeff,
                            QualParam.WallReactionCoeff,
                            mass_units=self.mass_units,
                            reaction_order=wn.options.quality.wall_rxn_order,
                        ),
                    ).encode('ascii')
                )
        f.write('\n'.encode('ascii'))
        #        f.write('[REACTIONS]\n'.encode('ascii'))  # EPANET GUI puts this line in here
        f.write(
            entry_int.format(
                'ORDER', 'BULK', int(wn.options.quality.bulk_rxn_order)
            ).encode('ascii')
        )
        f.write(
            entry_int.format(
                'ORDER', 'TANK', int(wn.options.quality.tank_rxn_order)
            ).encode('ascii')
        )
        f.write(
            entry_int.format(
                'ORDER', 'WALL', int(wn.options.quality.wall_rxn_order)
            ).encode('ascii')
        )
        f.write(
            entry_float.format(
                'GLOBAL',
                'BULK',
                from_si(
                    self.flow_units,
                    wn.options.quality.bulk_rxn_coeff,
                    QualParam.BulkReactionCoeff,
                    mass_units=self.mass_units,
                    reaction_order=wn.options.quality.bulk_rxn_order,
                ),
            ).encode('ascii')
        )
        f.write(
            entry_float.format(
                'GLOBAL',
                'WALL',
                from_si(
                    self.flow_units,
                    wn.options.quality.wall_rxn_coeff,
                    QualParam.WallReactionCoeff,
                    mass_units=self.mass_units,
                    reaction_order=wn.options.quality.wall_rxn_order,
                ),
            ).encode('ascii')
        )
        if wn.options.quality.limiting_potential is not None:
            f.write(
                entry_float.format(
                    'LIMITING', 'POTENTIAL', wn.options.quality.limiting_potential
                ).encode('ascii')
            )
        if wn.options.quality.roughness_correl is not None:
            f.write(
                entry_float.format(
                    'ROUGHNESS', 'CORRELATION', wn.options.quality.roughness_correl
                ).encode('ascii')
            )
        f.write('\n'.encode('ascii'))

    def _write_sources(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[SOURCES]\n'.encode('ascii'))
        entry = '{:10s} {:10s} {:10s} {:10s}\n'
        label = '{:10s} {:10s} {:10s} {:10s}\n'
        f.write(label.format(';Node', 'Type', 'Quality', 'Pattern').encode('ascii'))
        nsources = list(wn._sources.keys())  # noqa: SLF001
        # nsources.sort()
        for source_name in nsources:
            source = wn._sources[source_name]  # noqa: SLF001
            if source._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue

            if source.source_type.upper() == 'MASS':
                strength = from_si(
                    self.flow_units,
                    source.strength_timeseries.base_value,
                    QualParam.SourceMassInject,
                    self.mass_units,
                )
            else:  # CONC, SETPOINT, FLOWPACED
                strength = from_si(
                    self.flow_units,
                    source.strength_timeseries.base_value,
                    QualParam.Concentration,
                    self.mass_units,
                )

            E = {  # noqa: N806
                'node': source.node_name,
                'type': source.source_type,
                'quality': str(strength),
                'pat': '',
            }
            if source.strength_timeseries.pattern_name is not None:
                E['pat'] = source.strength_timeseries.pattern_name
            f.write(
                entry.format(
                    E['node'], E['type'], str(E['quality']), E['pat']
                ).encode('ascii')
            )
        f.write('\n'.encode('ascii'))

    def _write_mixing(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[MIXING]\n'.encode('ascii'))
        f.write(
            '{:20s} {:5s} {}\n'.format(';Tank ID', 'Model', 'Fraction').encode(
                'ascii'
            )
        )
        lnames = list(wn.tank_name_list)
        # lnames.sort()
        for tank_name in lnames:
            tank = wn.nodes[tank_name]
            if tank._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if tank._mix_model is not None:  # noqa: SLF001
                if tank._mix_model in [MixType.Mixed, MixType.Mix1, 0]:  # noqa: PLR6201, SLF001
                    f.write(f' {tank_name:19s} MIXED\n'.encode('ascii'))
                elif tank._mix_model in [  # noqa: PLR6201, SLF001
                    MixType.TwoComp,
                    MixType.Mix2,
                    '2comp',
                    '2COMP',
                    1,
                ]:
                    f.write(
                        f' {tank_name:19s} 2COMP  {tank._mix_frac}\n'.encode('ascii')  # noqa: SLF001
                    )
                elif tank._mix_model in [MixType.FIFO, 2]:  # noqa: PLR6201, SLF001
                    f.write(f' {tank_name:19s} FIFO\n'.encode('ascii'))
                elif tank._mix_model in [MixType.LIFO, 3]:  # noqa: PLR6201, SLF001
                    f.write(f' {tank_name:19s} LIFO\n'.encode('ascii'))
                elif isinstance(tank._mix_model, str) and tank._mix_frac is not None:  # noqa: SLF001
                    f.write(
                        f' {tank_name:19s} {tank._mix_model} {tank._mix_frac}\n'.encode(  # noqa: SLF001
                            'ascii'
                        )
                    )
                elif isinstance(tank._mix_model, str):  # noqa: SLF001
                    f.write(f' {tank_name:19s} {tank._mix_model}\n'.encode('ascii'))  # noqa: SLF001
                else:
                    logger.warning('Unknown mixing model: %s', tank._mix_model)  # noqa: SLF001
        f.write('\n'.encode('ascii'))

    # Options and Reporting

    def _write_options(self, f, wn):  # noqa: ANN001, ANN202
        f.write('[OPTIONS]\n'.encode('ascii'))
        entry_string = '{:20s} {:20s}\n'
        entry_float = '{:20s} {:.11g}\n'
        f.write(entry_string.format('UNITS', self.flow_units.name).encode('ascii'))

        f.write(
            entry_string.format('HEADLOSS', wn.options.hydraulic.headloss).encode(
                'ascii'
            )
        )

        f.write(
            entry_float.format(
                'SPECIFIC GRAVITY', wn.options.hydraulic.specific_gravity
            ).encode('ascii')
        )

        f.write(
            entry_float.format('VISCOSITY', wn.options.hydraulic.viscosity).encode(
                'ascii'
            )
        )

        f.write(
            entry_float.format('TRIALS', wn.options.solver.trials).encode('ascii')
        )

        f.write(
            entry_float.format('ACCURACY', wn.options.solver.accuracy).encode(
                'ascii'
            )
        )

        f.write(
            entry_float.format('CHECKFREQ', wn.options.solver.checkfreq).encode(
                'ascii'
            )
        )

        f.write(
            entry_float.format('MAXCHECK', wn.options.solver.maxcheck).encode(
                'ascii'
            )
        )

        if wn.options.solver.damplimit != 0:
            f.write(
                entry_float.format('DAMPLIMIT', wn.options.solver.damplimit).encode(
                    'ascii'
                )
            )

        if wn.options.solver.unbalanced_value is None:
            f.write(
                entry_string.format(
                    'UNBALANCED', wn.options.solver.unbalanced
                ).encode('ascii')
            )
        else:
            f.write(
                '{:20s} {:s} {:d}\n'.format(
                    'UNBALANCED',
                    wn.options.solver.unbalanced,
                    wn.options.solver.unbalanced_value,
                ).encode('ascii')
            )

        # Sina Added here
        if wn.options.hydraulic.pattern is not None:
            f.write(
                entry_string.format('PATTERN', wn.options.hydraulic.pattern).encode(
                    'ascii'
                )
            )
        else:
            f.write(entry_string.format('PATTERN', '1').encode('ascii'))

        f.write(
            entry_float.format(
                'DEMAND MULTIPLIER', wn.options.hydraulic.demand_multiplier
            ).encode('ascii')
        )

        f.write(
            entry_string.format(
                'DEMAND MODEL', wn.options.hydraulic.demand_model
            ).encode('ascii')
        )

        f.write(
            entry_float.format(
                'MINIMUM PRESSURE', wn.options.hydraulic.minimum_pressure
            ).encode('ascii')
        )

        f.write(
            entry_float.format(
                'REQUIRED PRESSURE', wn.options.hydraulic.required_pressure
            ).encode('ascii')
        )

        f.write(
            entry_float.format(
                'PRESSURE EXPONENT', wn.options.hydraulic.pressure_exponent
            ).encode('ascii')
        )

        f.write(
            entry_float.format(
                'EMITTER EXPONENT', wn.options.hydraulic.emitter_exponent
            ).encode('ascii')
        )

        if wn.options.quality.mode.upper() in ['NONE', 'AGE']:  # noqa: PLR6201
            f.write(
                entry_string.format('QUALITY', wn.options.quality.mode).encode(
                    'ascii'
                )
            )
        elif wn.options.quality.mode.upper() == 'TRACE':
            f.write(
                '{:20s} {} {}\n'.format(
                    'QUALITY', wn.options.quality.mode, wn.options.quality.trace_node
                ).encode('ascii')
            )
        else:
            f.write(
                '{:20s} {} {}\n'.format(
                    'QUALITY',
                    wn.options.quality.chemical_name,
                    wn.options.quality.wq_units,
                ).encode('ascii')
            )

        f.write(
            entry_float.format('DIFFUSIVITY', wn.options.quality.diffusivity).encode(
                'ascii'
            )
        )

        f.write(
            entry_float.format('TOLERANCE', wn.options.solver.tolerance).encode(
                'ascii'
            )
        )

        if wn.options.hydraulic.hydraulics is not None:
            f.write(
                '{:20s} {:s} {:<30s}\n'.format(
                    'HYDRAULICS',
                    wn.options.hydraulic.hydraulics,
                    wn.options.hydraulic.hydraulics_filename,
                ).encode('ascii')
            )

        if wn.options.graphics.map_filename is not None:
            f.write(
                entry_string.format('MAP', wn.options.graphics.map_filename).encode(
                    'ascii'
                )
            )
        f.write('\n'.encode('ascii'))

    def _write_times(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[TIMES]\n'.encode('ascii'))
        entry = '{:20s} {:10s}\n'
        time_entry = '{:20s} {:02d}:{:02d}:{:02d}\n'

        hrs, mm, sec = _sec_to_string(wn.options.time.duration)
        f.write(time_entry.format('DURATION', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.hydraulic_timestep)
        f.write(
            time_entry.format('HYDRAULIC TIMESTEP', hrs, mm, sec).encode('ascii')
        )

        hrs, mm, sec = _sec_to_string(wn.options.time.quality_timestep)
        f.write(time_entry.format('QUALITY TIMESTEP', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.pattern_timestep)
        f.write(time_entry.format('PATTERN TIMESTEP', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.pattern_start)
        f.write(time_entry.format('PATTERN START', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.report_timestep)
        f.write(time_entry.format('REPORT TIMESTEP', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.report_start)
        f.write(time_entry.format('REPORT START', hrs, mm, sec).encode('ascii'))

        hrs, mm, sec = _sec_to_string(wn.options.time.start_clocktime)

        # Sina
        day = int(hrs / 24)
        hrs -= day * 24

        if hrs < 12:  # noqa: PLR2004
            time_format = ' AM'
        else:
            hrs -= 12
            time_format = ' PM'
        f.write(
            '{:20s} {:02d}:{:02d}:{:02d}{:s}\n'.format(
                'START CLOCKTIME', hrs, mm, sec, time_format
            ).encode('ascii')
        )

        hrs, mm, sec = _sec_to_string(wn.options.time.rule_timestep)

        # TODO: RULE TIMESTEP is not written?!  # noqa: TD002
        # f.write(time_entry.format('RULE TIMESTEP', hrs, mm, int(sec)).encode('ascii'))
        f.write(
            entry.format('STATISTIC', wn.options.results.statistic).encode('ascii')
        )
        f.write('\n'.encode('ascii'))

    def _write_coordinates(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[COORDINATES]\n'.encode('ascii'))
        entry = '{:10s} {:20.9f} {:20.9f}\n'
        label = '{:10s} {:10s} {:10s}\n'
        f.write(label.format(';Node', 'X-Coord', 'Y-Coord').encode('ascii'))
        for name, node in wn.nodes():
            if node._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            val = node.coordinates
            f.write(entry.format(name, val[0], val[1]).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_vertices(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[VERTICES]\n'.encode('ascii'))
        entry = '{:10s} {:20.9f} {:20.9f}\n'
        label = '{:10s} {:10s} {:10s}\n'
        f.write(label.format(';Link', 'X-Coord', 'Y-Coord').encode('ascii'))
        lnames = list(wn.pipe_name_list)
        # lnames.sort()
        for pipe_name in lnames:
            pipe = wn.links[pipe_name]
            if pipe._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            for vert in pipe._vertices:  # noqa: SLF001
                f.write(entry.format(pipe_name, vert[0], vert[1]).encode('ascii'))
        f.write('\n'.encode('ascii'))

    def _write_tags(self, f, wn):  # noqa: ANN001, ANN202, PLR6301
        f.write('[TAGS]\n'.encode('ascii'))
        entry = '{:10s} {:10s} {:10s}\n'
        label = '{:10s} {:10s} {:10s}\n'
        f.write(label.format(';type', 'name', 'tag').encode('ascii'))
        nnodes = list(wn.node_name_list)
        # nnodes.sort()
        for node_name in nnodes:
            node = wn.nodes[node_name]
            if node._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if node.tag:
                f.write(entry.format('NODE', node_name, node.tag).encode('ascii'))
        nlinks = list(wn.link_name_list)
        nlinks.sort()
        for link_name in nlinks:
            link = wn.links[link_name]
            if link._is_isolated == True:  # Sina added this  # noqa: SLF001, E712
                continue
            if link.tag:
                f.write(entry.format('LINK', link_name, link.tag).encode('ascii'))
        f.write('\n'.encode('ascii'))

    # End of File


class BinFile(wntrfr.epanet.io.BinFile):
    """EPANET binary output file reader class.

    This class provides read functionality for EPANET binary output files.

    Parameters
    ----------
    results_type : list of :class:`~wntrfr.epanet.util.ResultType`, default=None
        This parameter is *only* active when using a subclass of the BinFile that implements
        a custom reader or writer.
        If ``None``, then all results will be saved (node quality, demand, link flow, etc.).
        Otherwise, a list of result types can be passed to limit the memory used.
    network : bool, default=False
        Save a new WaterNetworkModel from the description in the output binary file. Certain
        elements may be missing, such as patterns and curves, if this is done.
    energy : bool, default=False
        Save the pump energy results.
    statistics : bool, default=False
        Save the statistics lines (different from the stats flag in the inp file) that are
        automatically calculated regarding hydraulic conditions.
    convert_status : bool, default=True
        Convert the EPANET link status (8 values) to simpler WNTR status (3 values). By
        default, this is done, and the encoded-cause status values are converted simple state
        values, instead.

    Returns
    -------
    :class:`~wntrfr.sim.results.SimulationResults`
        A WNTR results object will be created and added to the instance after read.

    """

    def __init__(  # noqa: ANN204
        self,
        result_types=None,  # noqa: ANN001, ARG002
        network=False,  # noqa: ANN001, FBT002, ARG002
        energy=False,  # noqa: ANN001, FBT002, ARG002
        statistics=False,  # noqa: ANN001, FBT002, ARG002
        convert_status=True,  # noqa: ANN001, FBT002, ARG002
    ):
        super().__init__()

    def read(self, filename, custom_handlers=False, start_time=None):  # noqa: ANN001, ANN201, C901, FBT002, PLR0914, PLR0915
        """Read a binary file and create a results object.

        Parameters
        ----------
        filename : str
            An EPANET BIN output file
        custom_handlers : bool, optional
            If true, then the the custom, by-line handlers will be used. (:func:`~save_ep_line`,
            :func:`~setup_ep_results`, :func:`~finalize_save`, etc.) Otherwise read will use
            a faster, all-at-once reader that reads all results.
        start_time : int
            If the simulation is interval based, then start_time can identify the time passed after
            the last simulation. Start_time will be added to all th timings in the result.

        Returns
        -------
        object
            returns a WaterNetworkResults object

        .. note:: Overloading
            This function should **not** be overloaded. Instead, overload the other functions
            to change how it saves the results. Specifically, overload :func:`~setup_ep_results`,
            :func:`~save_ep_line` and :func:`~finalize_save` to change how extended period
            simulation results in a different format (such as directly to a file or database).

        """
        self.results = wntrfr.sim.SimulationResults()
        logger.debug(start_time)
        if start_time == None:  # noqa: E711
            start_time = 0
        logger.debug('Read binary EPANET data from %s', filename)
        dt_str = f'|S{self.idlen}'
        with open(filename, 'rb') as fin:  # noqa: PTH123
            ftype = self.ftype
            idlen = self.idlen  # noqa: F841
            logger.debug('... read prolog information ...')
            prolog = np.fromfile(fin, dtype=np.int32, count=15)
            magic1 = prolog[0]
            version = prolog[1]
            nnodes = prolog[2]
            ntanks = prolog[3]
            nlinks = prolog[4]
            npumps = prolog[5]
            nvalve = prolog[6]
            wqopt = QualType(prolog[7])
            srctrace = prolog[8]
            flowunits = FlowUnits(prolog[9])
            presunits = PressureUnits(prolog[10])
            statsflag = StatisticsType(prolog[11])
            reportstart = prolog[12]
            reportstep = prolog[13]
            duration = prolog[14]
            logger.debug('EPANET/Toolkit version %d', version)
            logger.debug(
                'Nodes: %d; Tanks/Resrv: %d Links: %d; Pumps: %d; Valves: %d',
                nnodes,
                ntanks,
                nlinks,
                npumps,
                nvalve,
            )
            logger.debug(
                'WQ opt: %s; Trace Node: %s; Flow Units %s; Pressure Units %s',
                wqopt,
                srctrace,
                flowunits,
                presunits,
            )
            logger.debug(
                'Statistics: %s; Report Start %d, step %d; Duration=%d sec',
                statsflag,
                reportstart,
                reportstep,
                duration,
            )

            # Ignore the title lines
            np.fromfile(fin, dtype=np.uint8, count=240)
            inpfile = np.fromfile(fin, dtype=np.uint8, count=260)
            rptfile = np.fromfile(fin, dtype=np.uint8, count=260)
            chemical = str(np.fromfile(fin, dtype=dt_str, count=1)[0])
            #            wqunits = ''.join([chr(f) for f in np.fromfile(fin, dtype=np.uint8, count=idlen) if f!=0 ])
            wqunits = str(np.fromfile(fin, dtype=dt_str, count=1)[0])
            mass = wqunits.split('/', 1)[0]
            if mass in ['mg', 'ug', 'mg', 'ug']:  # noqa: PLR6201
                massunits = MassUnits[mass]
            else:
                massunits = MassUnits.mg
            self.flow_units = flowunits
            self.pres_units = presunits
            self.quality_type = wqopt
            self.mass_units = massunits
            self.num_nodes = nnodes
            self.num_tanks = ntanks
            self.num_links = nlinks
            self.num_pumps = npumps
            self.num_valves = nvalve
            self.report_start = reportstart
            self.report_step = reportstep
            self.duration = duration
            self.chemical = chemical
            self.chem_units = wqunits
            self.inp_file = inpfile
            self.rpt_file = rptfile
            nodenames = []
            linknames = []
            nodenames = np.array(
                np.fromfile(fin, dtype=dt_str, count=nnodes), dtype=str
            ).tolist()
            linknames = np.array(
                np.fromfile(fin, dtype=dt_str, count=nlinks), dtype=str
            ).tolist()
            self.node_names = nodenames
            self.link_names = linknames
            linkstart = np.array(  # noqa: F841
                np.fromfile(fin, dtype=np.int32, count=nlinks), dtype=int
            )
            linkend = np.array(  # noqa: F841
                np.fromfile(fin, dtype=np.int32, count=nlinks), dtype=int
            )
            linktype = np.fromfile(fin, dtype=np.int32, count=nlinks)
            tankidxs = np.fromfile(fin, dtype=np.int32, count=ntanks)  # noqa: F841
            tankarea = np.fromfile(fin, dtype=np.dtype(ftype), count=ntanks)  # noqa: F841
            elevation = np.fromfile(fin, dtype=np.dtype(ftype), count=nnodes)  # noqa: F841
            linklen = np.fromfile(fin, dtype=np.dtype(ftype), count=nlinks)  # noqa: F841
            diameter = np.fromfile(fin, dtype=np.dtype(ftype), count=nlinks)  # noqa: F841
            """
            self.save_network_desc_line('link_start', linkstart)
            self.save_network_desc_line('link_end', linkend)
            self.save_network_desc_line('link_type', linktype)
            self.save_network_desc_line('tank_node_index', tankidxs)
            self.save_network_desc_line('tank_area', tankarea)
            self.save_network_desc_line('node_elevation', elevation)
            self.save_network_desc_line('link_length', linklen)
            self.save_network_desc_line('link_diameter', diameter)
            """
            logger.debug('... read energy data ...')
            for i in range(npumps):  # noqa: B007
                pidx = int(np.fromfile(fin, dtype=np.int32, count=1))
                energy = np.fromfile(fin, dtype=np.dtype(ftype), count=6)
                self.save_energy_line(pidx, linknames[pidx - 1], energy)
            peakenergy = np.fromfile(fin, dtype=np.dtype(ftype), count=1)
            self.peak_energy = peakenergy

            logger.debug('... read EP simulation data ...')
            reporttimes = (
                np.arange(reportstart, duration + reportstep, reportstep)
                + start_time
            )
            nrptsteps = len(reporttimes)
            statsN = nrptsteps  # noqa: N806, F841
            if statsflag in [  # noqa: PLR6201
                StatisticsType.Maximum,
                StatisticsType.Minimum,
                StatisticsType.Range,
            ]:
                nrptsteps = 1
                reporttimes = [reportstart + reportstep]
            self.num_periods = nrptsteps
            self.report_times = reporttimes

            # set up results metadata dictionary
            """
            if wqopt == QualType.Age:
                self.results.meta['quality_mode'] = 'AGE'
                self.results.meta['quality_units'] = 's'
            elif wqopt == QualType.Trace:
                self.results.meta['quality_mode'] = 'TRACE'
                self.results.meta['quality_units'] = '%'
                self.results.meta['quality_trace'] = srctrace
            elif wqopt == QualType.Chem:
                self.results.meta['quality_mode'] = 'CHEMICAL'
                self.results.meta['quality_units'] = wqunits
                self.results.meta['quality_chem'] = chemical
            self.results.time = reporttimes
            self.save_network_desc_line('report_times', reporttimes)
            self.save_network_desc_line('node_elevation', pd.Series(data=elevation, index=nodenames))
            self.save_network_desc_line('link_length', pd.Series(data=linklen, index=linknames))
            self.save_network_desc_line('link_diameter', pd.Series(data=diameter, index=linknames))
            self.save_network_desc_line('stats_mode', statsflag)
            self.save_network_desc_line('stats_N', statsN)
            nodetypes = np.array(['Junction']*self.num_nodes, dtype='|S10')
            nodetypes[tankidxs-1] = 'Tank'
            nodetypes[tankidxs[tankarea==0]-1] = 'Reservoir'
            linktypes = np.array(['Pipe']*self.num_links)
            linktypes[ linktype == EN.PUMP ] = 'Pump'
            linktypes[ linktype > EN.PUMP ] = 'Valve'
            self.save_network_desc_line('link_type', pd.Series(data=linktypes, index=linknames, copy=True))
            linktypes[ linktype == EN.CVPIPE ] = 'CV'
            linktypes[ linktype == EN.FCV ] = 'FCV'
            linktypes[ linktype == EN.PRV ] = 'PRV'
            linktypes[ linktype == EN.PSV ] = 'PSV'
            linktypes[ linktype == EN.PBV ] = 'PBV'
            linktypes[ linktype == EN.TCV ] = 'TCV'
            linktypes[ linktype == EN.GPV ] = 'GPV'
            self.save_network_desc_line('link_subtype', pd.Series(data=linktypes, index=linknames, copy=True))
            self.save_network_desc_line('node_type', pd.Series(data=nodetypes, index=nodenames, copy=True))
            self.save_network_desc_line('node_names', np.array(nodenames, dtype=str))
            self.save_network_desc_line('link_names', np.array(linknames, dtype=str))
            names = np.array(nodenames, dtype=str)
            self.save_network_desc_line('link_start', pd.Series(data=names[linkstart-1], index=linknames, copy=True))
            self.save_network_desc_line('link_end', pd.Series(data=names[linkend-1], index=linknames, copy=True))
            """
            if custom_handlers is True:
                logger.debug('... set up results object ...')
                # self.setup_ep_results(reporttimes, nodenames, linknames)
                # print(nodenames[5712]+'  '+nodenames[5717]+'  '+nodenames[5718]+'  ')
                for ts in range(nrptsteps):
                    try:
                        demand = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nnodes
                        )
                        # print(repr(demand[5712])+'  '+repr(demand[5717])+'  '+repr(demand[5718]))
                        head = np.fromfile(fin, dtype=np.dtype(ftype), count=nnodes)
                        pressure = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nnodes
                        )
                        quality = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nnodes
                        )
                        flow = np.fromfile(fin, dtype=np.dtype(ftype), count=nlinks)
                        velocity = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        headloss = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        linkquality = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        linkstatus = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        linksetting = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        reactionrate = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        frictionfactor = np.fromfile(
                            fin, dtype=np.dtype(ftype), count=nlinks
                        )
                        self.save_ep_line(ts, ResultType.demand, demand)
                        self.save_ep_line(ts, ResultType.head, head)
                        self.save_ep_line(ts, ResultType.pressure, pressure)
                        self.save_ep_line(ts, ResultType.quality, quality)
                        self.save_ep_line(ts, ResultType.flowrate, flow)
                        self.save_ep_line(ts, ResultType.velocity, velocity)
                        self.save_ep_line(ts, ResultType.headloss, headloss)
                        self.save_ep_line(ts, ResultType.linkquality, linkquality)
                        self.save_ep_line(ts, ResultType.status, linkstatus)
                        self.save_ep_line(ts, ResultType.setting, linksetting)
                        self.save_ep_line(ts, ResultType.rxnrate, reactionrate)
                        self.save_ep_line(
                            ts, ResultType.frictionfact, frictionfactor
                        )
                    except Exception as e:  # noqa: PERF203
                        logger.exception('Error reading or writing EP line: %s', e)  # noqa: TRY401
                        logger.warning('Missing results from report period %d', ts)
            else:
                #                type_list = 4*nnodes*['node'] + 8*nlinks*['link']
                name_list = nodenames * 4 + linknames * 8
                valuetype = (
                    nnodes * ['demand']
                    + nnodes * ['head']
                    + nnodes * ['pressure']
                    + nnodes * ['quality']
                    + nlinks * ['flow']
                    + nlinks * ['velocity']
                    + nlinks * ['headloss']
                    + nlinks * ['linkquality']
                    + nlinks * ['linkstatus']
                    + nlinks * ['linksetting']
                    + nlinks * ['reactionrate']
                    + nlinks * ['frictionfactor']
                )

                #                tuples = zip(type_list, valuetype, name_list)
                tuples = list(zip(valuetype, name_list))
                #                tuples = [(valuetype[i], v) for i, v in enumerate(name_list)]
                index = pd.MultiIndex.from_tuples(tuples, names=['value', 'name'])
                try:
                    data = np.fromfile(
                        fin,
                        dtype=np.dtype(ftype),
                        count=(4 * nnodes + 8 * nlinks) * nrptsteps,
                    )
                    data = np.reshape(data, (nrptsteps, (4 * nnodes + 8 * nlinks)))
                except Exception as e:
                    logger.exception('Failed to process file: %s', e)  # noqa: TRY401

                df = pd.DataFrame(data.transpose(), index=index, columns=reporttimes)  # noqa: PD901
                df = df.transpose()  # noqa: PD901

                self.results.node = {}
                self.results.link = {}
                self.results.network_name = self.inp_file

                # Node Results
                self.results.node['demand'] = HydParam.Demand._to_si(  # noqa: SLF001
                    self.flow_units, df['demand']
                )
                self.results.node['head'] = HydParam.HydraulicHead._to_si(  # noqa: SLF001
                    self.flow_units, df['head']
                )
                self.results.node['pressure'] = HydParam.Pressure._to_si(  # noqa: SLF001
                    self.flow_units, df['pressure']
                )

                # Water Quality Results (node and link)
                if self.quality_type is QualType.Chem:
                    self.results.node['quality'] = QualParam.Concentration._to_si(  # noqa: SLF001
                        self.flow_units, df['quality'], mass_units=self.mass_units
                    )
                    self.results.link['linkquality'] = (
                        QualParam.Concentration._to_si(  # noqa: SLF001
                            self.flow_units,
                            df['linkquality'],
                            mass_units=self.mass_units,
                        )
                    )
                elif self.quality_type is QualType.Age:
                    self.results.node['quality'] = QualParam.WaterAge._to_si(  # noqa: SLF001
                        self.flow_units, df['quality'], mass_units=self.mass_units
                    )
                    self.results.link['linkquality'] = QualParam.WaterAge._to_si(  # noqa: SLF001
                        self.flow_units,
                        df['linkquality'],
                        mass_units=self.mass_units,
                    )
                else:
                    self.results.node['quality'] = df['quality']
                    self.results.link['linkquality'] = df['linkquality']

                # Link Results
                self.results.link['flowrate'] = HydParam.Flow._to_si(  # noqa: SLF001
                    self.flow_units, df['flow']
                )
                self.results.link['headloss'] = df['headloss']  # Unit is per 1000
                self.results.link['velocity'] = HydParam.Velocity._to_si(  # noqa: SLF001
                    self.flow_units, df['velocity']
                )

                #                self.results.link['status'] = df['linkstatus']
                status = np.array(df['linkstatus'])
                if self.convert_status:
                    status[status <= 2] = 0  # noqa: PLR2004
                    status[status == 3] = 1  # noqa: PLR2004
                    status[status >= 5] = 1  # noqa: PLR2004
                    status[status == 4] = 2  # noqa: PLR2004
                self.results.link['status'] = pd.DataFrame(
                    data=status, columns=linknames, index=reporttimes
                )

                settings = np.array(df['linksetting'])
                settings[:, linktype == EN.PRV] = to_si(
                    self.flow_units,
                    settings[:, linktype == EN.PRV],
                    HydParam.Pressure,
                )
                settings[:, linktype == EN.PSV] = to_si(
                    self.flow_units,
                    settings[:, linktype == EN.PSV],
                    HydParam.Pressure,
                )
                settings[:, linktype == EN.PBV] = to_si(
                    self.flow_units,
                    settings[:, linktype == EN.PBV],
                    HydParam.Pressure,
                )
                settings[:, linktype == EN.FCV] = to_si(
                    self.flow_units, settings[:, linktype == EN.FCV], HydParam.Flow
                )
                self.results.link['setting'] = pd.DataFrame(
                    data=settings, columns=linknames, index=reporttimes
                )
                self.results.link['frictionfact'] = df['frictionfactor']
                self.results.link['rxnrate'] = df['reactionrate']

            logger.debug('... read epilog ...')
            # Read the averages and then the number of periods for checks
            averages = np.fromfile(fin, dtype=np.dtype(ftype), count=4)
            self.averages = averages
            np.fromfile(fin, dtype=np.int32, count=1)
            warnflag = np.fromfile(fin, dtype=np.int32, count=1)
            magic2 = np.fromfile(fin, dtype=np.int32, count=1)
            if magic1 != magic2:
                logger.critical(
                    'The magic number did not match -- binary incomplete or incorrectly read. If you believe this file IS complete, please try a different float type. Current type is "%s"',
                    ftype,
                )
            # print numperiods, warnflag, magic
            if warnflag != 0:
                logger.warning('Warnings were issued during simulation')
        self.finalize_save(magic1 == magic2, warnflag)

        return self.results
