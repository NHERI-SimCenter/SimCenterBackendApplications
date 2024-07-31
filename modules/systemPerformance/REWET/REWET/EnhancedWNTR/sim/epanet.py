"""Created on Tue Jun  1 17:09:25 2021

@author: snaeimi
"""  # noqa: D400, D415

import itertools
import logging
from collections import OrderedDict

import numpy as np
import scipy.sparse.csr
import wntrfr.epanet.io
from Report_Reading import Report_Reading
from wntrfr.network.io import write_inpfile
from wntrfr.network.model import LinkStatus
from wntrfr.sim.core import _get_csr_data_index
from wntrfr.sim.epanet import EpanetSimulator
from wntrfr.sim.network_isolation import check_for_isolated_junctions, get_long_size
from wntrfr.utils.ordered_set import OrderedSet

from ..epanet import toolkit  # noqa: TID252

logger = logging.getLogger(__name__)


class EpanetSimulator(EpanetSimulator):
    """Fast EPANET simulator class.

    Use the EPANET DLL to run an INP file as-is, and read the results from the
    binary output file. Multiple water quality simulations are still possible
    using the WQ keyword in the run_sim function. Hydraulics will be stored and
    saved to a file. This file will not be deleted by default, nor will any
    binary files be deleted.

    The reason this is considered a "fast" simulator is due to the fact that there
    is no looping within Python. The "ENsolveH" and "ENsolveQ" toolkit
    functions are used instead.

    Parameters
    ----------
    wn : WaterNetworkModel
        Water network model
    mode: DD or PDD with default None value(read mode from InpFile, if there is no mode
        provided in inpdile either, it will be DD) If there is a conflict between mode in
        the class argument and inpfile, the augment will supersede the InpFile
    reader : wntrfr.epanet.io.BinFile derived object
        Defaults to None, which will create a new wntrfr.epanet.io.BinFile object with
        the results_types specified as an init option. Otherwise, a fully
    result_types : dict
        Defaults to None, or all results. Otherwise, is a keyword dictionary to pass to
        the reader to specify what results should be saved.


    .. seealso::

        wntrfr.epanet.io.BinFile

    """

    def __init__(self, wn):  # noqa: ANN001, ANN204, D107
        super(EpanetSimulator, self).__init__(wn)  # noqa: UP008

        # Sina added this for time manipulate function

        self._initial_hydraulic_timestep = wn.options.time.hydraulic_timestep
        self._initial_report_timestep = wn.options.time.hydraulic_timestep

        # Sina added this for isolation init
        long_size = get_long_size()
        if long_size == 4:  # noqa: PLR2004
            self._int_dtype = np.int32
        else:
            assert long_size == 8  # noqa: S101, PLR2004
            self._int_dtype = np.int64
        self._link_name_to_id = OrderedDict()
        self._link_id_to_name = OrderedDict()
        self._node_name_to_id = OrderedDict()
        self._node_id_to_name = OrderedDict()
        self._initialize_name_id_maps()
        # sina end

    def manipulateTimeOrder(  # noqa: ANN201, N802, D102
        self,
        begin_time,  # noqa: ANN001
        end_time,  # noqa: ANN001
        change_time_step=False,  # noqa: ANN001, FBT002
        min_correction_time_step=None,  # noqa: ANN001
    ):
        time_dif = end_time - begin_time
        min_step_time = min_correction_time_step
        self._wn.options.time.duration = time_dif
        if time_dif <= 1:
            self._wn.options.time.report_timestep = time_dif
        self._wn.options.time.pattern_start = begin_time
        self._wn.options.time.start_clocktime = begin_time
        if change_time_step:
            if min_correction_time_step == None:  # noqa: E711
                raise ValueError(  # noqa: TRY003
                    'if change_time_step is True, then min_correction_time_step must be provided'  # noqa: EM101
                )

            self._wn.options.time.hydraulic_timestep = (
                self._initial_hydraulic_timestep
            )
            self._wn.options.time.report_timestep = self._initial_report_timestep
            time_step = min(
                self._wn.options.time.hydraulic_timestep,
                self._wn.options.time.report_timestep,
            )
            if min_step_time > time_step:
                min_step_time = time_step
            iFinished = False  # noqa: N806
            i = 1
            logger.debug('time_dif= ' + repr(time_dif))  # noqa: G003

            time_step_list = list(range(min_step_time, time_step, min_step_time))
            time_step_list.append(time_step)
            while i <= len(time_step_list):
                if time_dif % time_step_list[-i] == 0:
                    new_time_step = time_step_list[-i]
                    iFinished = True  # noqa: N806
                    break
                elif i == len(time_step_list):  # noqa: RET508
                    raise ('There was no time check when creating time event?')  # noqa: B016
                i += 1
            if iFinished == False:  # noqa: E712
                raise RuntimeError('no timestep is found')  # noqa: EM101, TRY003
            self._wn.options.time.report_timestep = new_time_step

    def run_sim(  # noqa: ANN201, C901, D417, PLR0912, PLR0913, PLR0915
        self,
        file_prefix='temp',  # noqa: ANN001
        save_hyd=False,  # noqa: ANN001, FBT002
        use_hyd=False,  # noqa: ANN001, FBT002
        hydfile=None,  # noqa: ANN001
        version=2.2,  # noqa: ANN001
        convergence_error=False,  # noqa: ANN001, FBT002, ARG002
        start_time=None,  # noqa: ANN001
        iModified=True,  # noqa: ANN001, FBT002, N803
    ):
        """Run the EPANET simulator.

        Runs the EPANET simulator through the compiled toolkit DLL. Can use/save hydraulics
        to allow for separate WQ runs.

        Parameters
        ----------
        file_prefix : str
            Default prefix is "temp". All files (.inp, .bin/.out, .hyd, .rpt) use this prefix
        use_hyd : bool
            Will load hydraulics from ``file_prefix + '.hyd'`` or from file specified in `hydfile_name`
        save_hyd : bool
            Will save hydraulics to ``file_prefix + '.hyd'`` or to file specified in `hydfile_name`
        hydfile : str
            Optionally specify a filename for the hydraulics file other than the `file_prefix`

        """
        solver_parameters_list = [(1, 10, 0), (10, 100, 0), (10, 100, 0.01)]
        # solver_parameters_list = [(10,100, 0.01), (10, 100, 0), (1,10, 0)]
        # balanced_system = False
        run_successful = False
        i = 0
        for solver_parameter in solver_parameters_list:
            i += 1
            print(solver_parameter)  # noqa: T201
            self._wn.options.hydraulic.checkfreq = solver_parameter[0]
            self._wn.options.hydraulic.maxcheck = solver_parameter[1]
            self._wn.options.hydraulic.damplimit = solver_parameter[2]
            self._wn.options.hydraulic.unbalanced_value = 100

            inpfile = file_prefix + '.inp'
            write_inpfile(
                self._wn,
                inpfile,
                units=self._wn.options.hydraulic.inpfile_units,
                version=version,
            )

            enData = toolkit.ENepanet(changed_epanet=iModified, version=version)  # noqa: N806
            rptfile = file_prefix + '.rpt'
            outfile = file_prefix + '.bin'
            if hydfile is None:
                hydfile = file_prefix + '.hyd'
            enData.ENopen(inpfile, rptfile, outfile)
            if use_hyd:
                enData.ENusehydfile(hydfile)
                logger.debug('Loaded hydraulics')
            else:
                try:
                    enData.ENsolveH()
                except Exception as err:
                    enData.ENclose()
                    if err.args[0] == 'EPANET Error 110':
                        print(enData.errcode)  # noqa: T201
                        run_successful = False
                        if i < len(solver_parameters_list):
                            continue
                        else:  # noqa: RET507
                            raise err  # noqa: TRY201
                    else:
                        raise err  # noqa: TRY201
                else:
                    run_successful = True
                logger.debug('Solved hydraulics')
            if save_hyd:
                enData.ENsavehydfile(hydfile)
                logger.debug('Saved hydraulics')

            try:
                enData.ENsolveQ()
                logger.debug('Solved quality')
                enData.ENreport()
                logger.debug('Ran quality')
            except Exception as err:
                enData.ENclose()
                raise err  # noqa: TRY201
            enData.ENclose()
            logger.debug('Completed run')
            result_data = self.reader.read(outfile)

            self._updateResultStartTime(result_data, start_time)

            report_data = Report_Reading(rptfile)

            result_data.maximum_trial_time = []

            for time in report_data.maximum_trial_time:
                result_data.maximum_trial_time.append(time + start_time)
            if run_successful:
                break

        return result_data, run_successful

    def _updateResultStartTime(self, result_data, start_time):  # noqa: ANN001, ANN202, N802
        for res_type, res in result_data.link.items():  # noqa: B007, PERF102
            # result_data.link[res_type].index = res
            res.index = res.index + start_time

        for res_type, res in result_data.node.items():  # noqa: B007, PERF102
            # result_data.link[res_type].index = res
            res.index = res.index + start_time

    def _get_isolated_junctions_and_links(  # noqa: ANN202
        self,
        prev_isolated_junctions,  # noqa: ANN001
        prev_isolated_links,  # noqa: ANN001
    ):
        self._prev_isolated_junctions = prev_isolated_junctions
        self._prev_isolated_links = prev_isolated_links

        self._initialize_internal_graph()
        logger_level = logger.getEffectiveLevel()

        if logger_level <= logging.DEBUG:
            logger.debug('checking for isolated junctions and links')
        for j in self._prev_isolated_junctions:
            try:
                junction = self._wn.get_node(j)
                junction._is_isolated = False  # noqa: SLF001
            except:  # noqa: S110, PERF203, E722
                pass
        for l in self._prev_isolated_links:  # noqa: E741
            try:
                link = self._wn.get_link(l)
                link._is_isolated = False  # noqa: SLF001
            except:  # noqa: S110, PERF203, E722
                pass
        node_indicator = np.ones(self._wn.num_nodes, dtype=self._int_dtype)
        check_for_isolated_junctions(
            self._source_ids,
            node_indicator,
            self._internal_graph.indptr,
            self._internal_graph.indices,
            self._internal_graph.data,
            self._number_of_connections,
        )

        isolated_junction_ids = [
            i for i in range(len(node_indicator)) if node_indicator[i] == 1
        ]
        isolated_junctions = OrderedSet()
        isolated_links = OrderedSet()
        for j_id in isolated_junction_ids:
            j = self._node_id_to_name[j_id]
            junction = self._wn.get_node(j)
            junction._is_isolated = True  # noqa: SLF001
            isolated_junctions.add(j)
            connected_links = self._wn.get_links_for_node(j)
            for l in connected_links:  # noqa: E741
                link = self._wn.get_link(l)
                link._is_isolated = True  # noqa: SLF001
                isolated_links.add(l)

        if logger_level <= logging.DEBUG:  # noqa: SIM102
            if len(isolated_junctions) > 0 or len(isolated_links) > 0:
                raise ValueError(f'isolated junctions: {isolated_junctions}')  # noqa: EM102, TRY003
                logger.debug(f'isolated links: {isolated_links}')  # noqa: G004

        self._prev_isolated_junctions = isolated_junctions
        self._prev_isolated_links = isolated_links
        return isolated_junctions, isolated_links

    def _initialize_internal_graph(self):  # noqa: ANN202, C901, PLR0912, PLR0915
        n_links = OrderedDict()
        rows = []
        cols = []
        vals = []
        for link_name, link in itertools.chain(  # noqa: B007
            self._wn.pipes(), self._wn.pumps(), self._wn.valves()
        ):
            from_node_name = link.start_node_name
            to_node_name = link.end_node_name
            from_node_id = self._node_name_to_id[from_node_name]
            to_node_id = self._node_name_to_id[to_node_name]
            if (from_node_id, to_node_id) not in n_links:
                n_links[(from_node_id, to_node_id)] = 0
                n_links[(to_node_id, from_node_id)] = 0
            n_links[(from_node_id, to_node_id)] += 1
            n_links[(to_node_id, from_node_id)] += 1
            rows.append(from_node_id)
            cols.append(to_node_id)
            rows.append(to_node_id)
            cols.append(from_node_id)
            if link.initial_status == wntrfr.network.LinkStatus.closed:
                vals.append(0)
                vals.append(0)
                # sina remove comment amrks
            elif link.link_type == 'Pipe':
                if link.cv:
                    vals.append(1)
                    vals.append(0)
                else:
                    vals.append(1)
                    vals.append(1)
            elif link.link_type == 'Valve':
                if (
                    link.valve_type == 'PRV'  # noqa: PLR1714
                    or link.valve_type == 'PSV'
                    or link.valve_type == 'FCV'
                ):
                    vals.append(1)
                    vals.append(0)
                else:
                    vals.append(1)
                    vals.append(1)
            else:
                vals.append(1)
                vals.append(1)

        rows = np.array(rows, dtype=self._int_dtype)
        cols = np.array(cols, dtype=self._int_dtype)
        vals = np.array(vals, dtype=self._int_dtype)
        self._internal_graph = scipy.sparse.csr_matrix((vals, (rows, cols)))

        ndx_map = OrderedDict()
        for link_name, link in self._wn.links():  # noqa: B007
            from_node_name = link.start_node_name
            to_node_name = link.end_node_name
            from_node_id = self._node_name_to_id[from_node_name]
            to_node_id = self._node_name_to_id[to_node_name]
            ndx1 = _get_csr_data_index(
                self._internal_graph, from_node_id, to_node_id
            )
            ndx2 = _get_csr_data_index(
                self._internal_graph, to_node_id, from_node_id
            )
            ndx_map[link] = (ndx1, ndx2)
        self._map_link_to_internal_graph_data_ndx = ndx_map

        self._number_of_connections = [0 for i in range(self._wn.num_nodes)]
        for node_id in self._node_id_to_name.keys():  # noqa: SIM118
            self._number_of_connections[node_id] = (
                self._internal_graph.indptr[node_id + 1]
                - self._internal_graph.indptr[node_id]
            )
        self._number_of_connections = np.array(
            self._number_of_connections, dtype=self._int_dtype
        )

        self._node_pairs_with_multiple_links = OrderedDict()
        for from_node_id, to_node_id in n_links.keys():  # noqa: SIM118
            if n_links[(from_node_id, to_node_id)] > 1:
                if (
                    to_node_id,
                    from_node_id,
                ) in self._node_pairs_with_multiple_links:
                    continue
                self._internal_graph[from_node_id, to_node_id] = 0
                self._internal_graph[to_node_id, from_node_id] = 0
                from_node_name = self._node_id_to_name[from_node_id]
                to_node_name = self._node_id_to_name[to_node_id]
                tmp_list = self._node_pairs_with_multiple_links[
                    (from_node_id, to_node_id)
                ] = []
                for link_name in self._wn.get_links_for_node(from_node_name):
                    link = self._wn.get_link(link_name)
                    if (
                        link.start_node_name == to_node_name  # noqa: PLR1714
                        or link.end_node_name == to_node_name
                    ):
                        tmp_list.append(link)
                        if link.initial_status != wntrfr.network.LinkStatus.closed:
                            ndx1, ndx2 = ndx_map[link]
                            self._internal_graph.data[ndx1] = 1
                            self._internal_graph.data[ndx2] = 1

        self._source_ids = []
        for node_name, node in self._wn.tanks():
            if node.init_level - node.min_level < 0.01:  # noqa: PLR2004
                continue
            node_id = self._node_name_to_id[node_name]
            self._source_ids.append(node_id)

        for node_name, node in self._wn.reservoirs():  # noqa: B007
            connected_link_name_list = self._wn.get_links_for_node(
                node_name
            )  # this is to exclude the reservoirs that are for leak only
            out_going_link_list_name = [
                link_name
                for link_name in connected_link_name_list
                if self._wn.get_link(link_name).link_type != 'Pipe'
            ]
            out_going_pipe_list_name = [
                self._wn.get_link(pipe_name)
                for pipe_name in connected_link_name_list
                if self._wn.get_link(pipe_name).link_type == 'Pipe'
            ]
            out_going_pipe_list_name = [
                link.name
                for link in out_going_pipe_list_name
                if (
                    (link.cv == False and link.initial_status != LinkStatus.Closed)  # noqa: E712
                    or (link.cv == True and link.end_node_name != node_name)  # noqa: E712
                )
            ]
            out_going_link_list_name.extend(out_going_pipe_list_name)
            if len(out_going_link_list_name) < 1:
                continue
            node_id = self._node_name_to_id[node_name]
            self._source_ids.append(node_id)
        self._source_ids = np.array(self._source_ids, dtype=self._int_dtype)

    def _update_internal_graph(self):  # noqa: ANN202
        data = self._internal_graph.data
        ndx_map = self._map_link_to_internal_graph_data_ndx
        for mgr in [self._presolve_controls, self._rules, self._postsolve_controls]:
            for obj, attr in mgr.get_changes():
                if attr == 'status':
                    if obj.status == wntrfr.network.LinkStatus.closed:
                        ndx1, ndx2 = ndx_map[obj]
                        data[ndx1] = 0
                        data[ndx2] = 0
                    else:
                        ndx1, ndx2 = ndx_map[obj]
                        data[ndx1] = 1
                        data[ndx2] = 1

        for key, link_list in self._node_pairs_with_multiple_links.items():  # noqa: B007, PERF102
            first_link = link_list[0]
            ndx1, ndx2 = ndx_map[first_link]
            data[ndx1] = 0
            data[ndx2] = 0
            for link in link_list:
                if link.status != wntrfr.network.LinkStatus.closed:
                    ndx1, ndx2 = ndx_map[link]
                    data[ndx1] = 1
                    data[ndx2] = 1

    def _initialize_name_id_maps(self):  # noqa: ANN202
        n = 0
        for link_name, link in self._wn.links():  # noqa: B007
            self._link_name_to_id[link_name] = n
            self._link_id_to_name[n] = link_name
            n += 1  # noqa: SIM113
        n = 0
        for node_name, node in self._wn.nodes():  # noqa: B007
            self._node_name_to_id[node_name] = n
            self._node_id_to_name[n] = node_name
            n += 1

    def now_temp(  # noqa: ANN201, D102
        self,
        rr,  # noqa: ANN001
        isolated_link_list,  # noqa: ANN001
        alread_closed_pipes,  # noqa: ANN001
        _prev_isolated_junctions,  # noqa: ANN001
        already_done_nodes,  # noqa: ANN001
    ):
        check_nodes = [
            node_name
            for node_name in self._wn.junction_name_list
            if node_name not in _prev_isolated_junctions
            and node_name not in already_done_nodes
        ]
        junctions_pressure = (rr.node['pressure'][check_nodes]).iloc[-1]
        negative_junctions_pressure = junctions_pressure[(junctions_pressure < -10)]  # noqa: PLR2004
        negative_junctions_pressure = negative_junctions_pressure.sort_values(
            ascending=False
        )
        negative_junctions_name_list = negative_junctions_pressure.index.to_list()
        print('size= ' + repr(len(negative_junctions_name_list)))  # noqa: T201

        pipes_to_be_closed = []
        closed_pipes = []
        # closed_nodes = []
        ifinish = False

        if len(negative_junctions_name_list) > 0:
            i = 0
            c = 0
            while i < np.ceil(
                len(negative_junctions_name_list) / len(negative_junctions_name_list)
            ):
                # for i in np.arange(0, ,1 ):
                if i + c >= len(negative_junctions_name_list):
                    break
                node_name = negative_junctions_name_list[i + c]
                already_done_nodes.append(node_name)
                # for node_name in negative_junctions_name_list:
                pipe_linked_to_node = self._wn.get_links_for_node(node_name)
                # get_checked_pipe_bool = self.check_pipes_sin(self, pipe_linked_to_node)
                checked_pipe_list = [
                    checked_pipe
                    for checked_pipe in pipe_linked_to_node
                    if self._wn.get_link(checked_pipe).link_type == 'Pipe'
                    and checked_pipe not in isolated_link_list
                    and self._wn.get_link(checked_pipe).cv == False  # noqa: E712
                    and self._wn.get_link(checked_pipe).initial_status == 1
                    and self._wn.get_link(checked_pipe).start_node.node_type
                    == 'Junction'
                    and self._wn.get_link(checked_pipe).end_node.node_type
                    == 'Junction'
                    and checked_pipe not in alread_closed_pipes
                ]
                pipes_to_be_closed.extend(checked_pipe_list)

                flag = False
                for pipe_name in pipes_to_be_closed:
                    # pipe = self.wn.get_link(pipe_name)
                    flow = rr.link['flowrate'][pipe_name].iloc[-1]

                    if abs(flow) > 0.01:  # noqa: PLR2004
                        flag = True
                        # pipe.initial_status = LinkStatus(0)
                        closed_pipes.append(pipe_name)
                if not flag:
                    i = i - 1
                    c = c + 1
                i = i + 1
        else:
            ifinish = True
        return closed_pipes, already_done_nodes, ifinish

    def alterPipeKmNNN(  # noqa: ANN201, N802, D102
        self,
        rr,  # noqa: ANN001
        isolated_link_list,  # noqa: ANN001
        _prev_isolated_junctions,  # noqa: ANN001
        flow_criteria,  # noqa: ANN001
        negative_pressure_limit,  # noqa: ANN001
    ):
        # t1 = time.time()

        closed_pipes = {}

        check_nodes = [
            node_name
            for node_name in self._wn.junction_name_list
            if node_name not in _prev_isolated_junctions
        ]  # not isolated junctions
        junctions_pressure = (rr.node['pressure'][check_nodes]).iloc[
            -1
        ]  # latest pressure result for not-isolated junctions
        negative_junctions_pressure = junctions_pressure[
            (junctions_pressure < negative_pressure_limit)
        ]  # not-isolated junctions that have pressure less than specified amount

        negative_junctions_pressure = negative_junctions_pressure.sort_values(
            ascending=False
        )
        negative_junctions_name_list = negative_junctions_pressure.index.to_list()

        last_flow_row = rr.link['flowrate'].iloc[-1]

        pipe_found = False
        while pipe_found == False:  # noqa: E712
            if len(negative_junctions_name_list) == 0:
                ifinish = True
                return closed_pipes, ifinish

            pipe_name_list = []
            pipe_name_list_temp = self._wn.get_links_for_node(
                negative_junctions_name_list[-1]
            )  # new: the most negative
            pipe_name_list.extend(pipe_name_list_temp)

            pipe_name_list = set(pipe_name_list) - set(isolated_link_list)
            pipe_name_list = [
                pipe_name
                for pipe_name in pipe_name_list
                if pipe_name in self._wn.pipe_name_list
            ]
            most_recent_flow_for_pipes = last_flow_row[pipe_name_list]
            abs_most_recent_flow_for_pipes = most_recent_flow_for_pipes.abs()
            abs_most_recent_flow_for_pipes = abs_most_recent_flow_for_pipes[
                abs_most_recent_flow_for_pipes >= flow_criteria
            ]

            if len(abs_most_recent_flow_for_pipes) == 0:
                negative_junctions_pressure.drop(
                    negative_junctions_name_list[-1],
                    inplace=True,  # noqa: PD002
                )
                negative_junctions_name_list = (
                    negative_junctions_pressure.index.to_list()
                )
            else:
                pipe_found = True
        ifinish = False
        abs_most_recent_flow_for_pipes = abs_most_recent_flow_for_pipes.sort_values(
            ascending=False
        )
        biggest_flow_pipe_name = abs_most_recent_flow_for_pipes.index[0]
        biggest_flow_pipe_abs_flow = abs_most_recent_flow_for_pipes.iloc[0]
        pipe = self._wn.get_link(biggest_flow_pipe_name)
        # n1 = pipe.start_node_name
        # n2 = pipe.end_node_name
        # n1_pressure = rr.node['pressure'][n1].iloc[-1]
        # n2_pressure = rr.node['pressure'][n2].iloc[-1]
        already_C = pipe.minor_loss  # noqa: N806
        # if already_C < 0.001:
        # already_C = 1
        new_C = (1000 * 2 * 9.81 * (pipe.diameter**2 * 3.14 / 4) ** 2) / (  # noqa: N806
            (biggest_flow_pipe_abs_flow) ** 2
        ) + already_C  # the last of 100 is to magnify the c choosing
        pipe.minor_loss = new_C
        closed_pipes[biggest_flow_pipe_name] = already_C

        # t2 = time.time()
        # print(t2-t1)
        # print('size closed: '+repr(len(closed_pipes)) )
        return closed_pipes, ifinish

        # if pipe.cv == True:
        # continue
        # if pipe._is_isolated == True:
        # continue
        # node_A = pipe.start_node
        # node_B = pipe.end_node

        # if node_A.node_type != "Junction" or node_B.node_type != "Junction":
        # continue

        # if node_A.name in already_nodes or node_B.name in already_nodes:
        # continue

        # if pipe.initial_status != 1:
        # continue

        # for
        # flow = rr.link['flowrate']

        # i_possitive_rate = True

        # if flow > 0.01:
        # i_possitive_rate = True
        # chosen_node = node_A
        # elif flow < 0.01:
        # i_possitive_rate = False
        # chosen_node = node_B
        # else:
        # continue

    # def check_pipes_sin(self, pipe_list):
    # for pipe_name in pipe_list:
    def closePipeNNN(  # noqa: ANN201, N802, D102
        self,
        rr,  # noqa: ANN001
        isolated_link_list,  # noqa: ANN001
        _prev_isolated_junctions,  # noqa: ANN001
        flow_criteria,  # noqa: ANN001
        negative_pressure_limit,  # noqa: ANN001
    ):
        closed_pipes = {}

        check_nodes = [
            node_name
            for node_name in self._wn.junction_name_list
            if node_name not in _prev_isolated_junctions
        ]  # not isolated junctions
        junctions_pressure = (rr.node['pressure'][check_nodes]).iloc[
            -1
        ]  # latest pressure result for not-isolated junctions
        negative_junctions_pressure = junctions_pressure[
            (junctions_pressure < negative_pressure_limit)
        ]  # not-isolated junctions that have pressure less than specified amount

        negative_junctions_pressure = negative_junctions_pressure.sort_values(
            ascending=False
        )
        negative_junctions_name_list = negative_junctions_pressure.index.to_list()

        last_flow_row = rr.link['flowrate'].iloc[-1]

        pipe_found = False
        while pipe_found == False:  # noqa: E712
            if len(negative_junctions_name_list) == 0:
                ifinish = True
                return closed_pipes, ifinish

            pipe_name_list = []
            pipe_name_list_temp = self._wn.get_links_for_node(
                negative_junctions_name_list[-1]
            )  # new: the most negative
            pipe_name_list.extend(pipe_name_list_temp)

            pipe_name_list = set(pipe_name_list) - set(isolated_link_list)
            pipe_name_list = [
                pipe_name
                for pipe_name in pipe_name_list
                if pipe_name in self._wn.pipe_name_list
            ]
            most_recent_flow_for_pipes = last_flow_row[pipe_name_list]
            abs_most_recent_flow_for_pipes = most_recent_flow_for_pipes.abs()
            abs_most_recent_flow_for_pipes = abs_most_recent_flow_for_pipes[
                abs_most_recent_flow_for_pipes >= flow_criteria
            ]

            if len(abs_most_recent_flow_for_pipes) == 0:
                negative_junctions_pressure.drop(
                    negative_junctions_name_list[-1],
                    inplace=True,  # noqa: PD002
                )
                negative_junctions_name_list = (
                    negative_junctions_pressure.index.to_list()
                )
            else:
                pipe_found = True
        ifinish = False
        abs_most_recent_flow_for_pipes = abs_most_recent_flow_for_pipes.sort_values(
            ascending=False
        )
        biggest_flow_pipe_name = abs_most_recent_flow_for_pipes.index[0]
        biggest_flow_pipe_abs_flow = abs_most_recent_flow_for_pipes.iloc[0]  # noqa: F841
        pipe = self._wn.get_link(biggest_flow_pipe_name)

        already_C = pipe.minor_loss  # noqa: N806, F841
        initial_status = pipe.initial_status
        closed_pipes[biggest_flow_pipe_name] = initial_status
        pipe.initial_status = LinkStatus.Closed

        return closed_pipes, ifinish

        # if pipe.cv == True:
        # continue
        # if pipe._is_isolated == True:
        # continue
        # node_A = pipe.start_node
        # node_B = pipe.end_node

        # if node_A.node_type != "Junction" or node_B.node_type != "Junction":
        # continue

        # if node_A.name in already_nodes or node_B.name in already_nodes:
        # continue

        # if pipe.initial_status != 1:
        # continue

        # for
        # flow = rr.link['flowrate']

        # i_possitive_rate = True

        # if flow > 0.01:
        # i_possitive_rate = True
        # chosen_node = node_A
        # elif flow < 0.01:
        # i_possitive_rate = False
        # chosen_node = node_B
        # else:
        # continue

    # def check_pipes_sin(self, pipe_list):
    # for pipe_name in pipe_list:
