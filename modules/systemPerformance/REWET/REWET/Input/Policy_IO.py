"""Created on Wed Dec 19 19:10:35 2020

This is the Restoration Policy Reader/Writtter Module.

@author: snaeimi
"""  # noqa: INP001, D400

import logging
from collections import OrderedDict

import pandas as pd

ELEMENTS = ['PIPE', 'DISTNODE', 'GNODE', 'TANK', 'PUMP', 'RESERVOIR']

logger = logging.getLogger(__name__)


# the following function is borrowed from WNTR
def _split_line(line):
    _vc = line.split(';', 1)
    _cmnt = None
    _vals = None
    if len(_vc) == 0:
        pass
    elif len(_vc) == 1:
        _vals = _vc[0].split()
    elif _vc[0] == '':
        _cmnt = _vc[1]
    else:
        _vals = _vc[0].split()
        _cmnt = _vc[1]
    return _vals, _cmnt


class restoration_data:  # noqa: D101
    def __init__(self):
        self.files = {}
        self.shift = {}
        self.entity = {}
        self.entity_rule = {}
        self.sequence = {}
        self.agents = []
        self.group = {}
        self.priority = []
        self.jobs = []
        self.jobs_default = []
        self.time_overwrite = {}
        self.final_method = {}
        self.once = {}

        for el in ELEMENTS:
            self.group[el] = OrderedDict()


class RestorationIO:  # noqa: D101
    def __init__(self, definition_file_name):
        """Needs a file that contains:

        Parameters
        ----------
        restoratio_model : Restoration object
            restoration model to be defined by files
        definition_file : str
            path for tegh definition file.

        Returns
        -------
        None.

        """  # noqa: D400, DOC202
        # some of the following lines have been adopted from WNTR
        self.rm = restoration_data()

        self.crew_data = {}

        expected_sections = [
            '[FILES]',
            '[ENTITIES]',
            '[JOBS]',
            '[AGENTS]',
            '[GROUPS]',
            '[PRIORITIES]',
            '[SHIFTS]',
            '[SEQUENCES]',
            '[DEFINE]',
            '[DAMAGE GROUPS]',
            '[EFFECTS]',
            '[CREWS]',
        ]

        self.config_file_comment = []
        self.edata = []

        self.sections = OrderedDict()
        for sec in expected_sections:
            self.sections[sec] = []

        section = None
        lnum = 0
        edata = {'fname': definition_file_name}
        with open(definition_file_name, encoding='utf-8') as f:  # noqa: PTH123
            for line in f:
                lnum += 1
                edata['lnum'] = lnum
                line = line.strip()  # noqa: PLW2901
                nwords = len(line.split())
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.startswith('['):  # noqa: RET507
                    vals = line.split()
                    sec = vals[0].upper()
                    edata['sec'] = sec
                    if sec in expected_sections:
                        section = sec
                        continue
                    else:  # noqa: RET507
                        raise RuntimeError(
                            '%(fname)s:%(lnum)d: Invalid section "%(sec)s"' % edata
                        )
                elif section is None and line.startswith(';'):
                    self.config_file_comment.append(line[1:])
                    continue
                elif section is None:
                    raise RuntimeError(
                        '%(fname)s:%(lnum)d: Non-comment outside of valid section!'
                        % edata
                    )
                # We have text, and we are in a section
                self.sections[section].append((lnum, line))

        # Parse each of the sections
        self._read_files()
        self._read_shifts()
        self._read_entities()
        self._read_agents()
        self._read_groups()
        self._read_sequences()
        self._read_priorities()
        self._read_jobs()
        self._read_define()
        # self._read_config()

    def _read_files(self):
        edata = OrderedDict()
        self.file_name = []
        self._file_data = {}
        self._file_handle_address = {}
        for lnum, line in self.sections['[FILES]']:
            edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) != 2:  # noqa: PLR2004
                    edata['key'] = words[0]
                    raise RuntimeError(
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'
                        % edata
                    )
                file_handle = words[0]
                file_address = words[1]

                self._file_handle_address[file_handle] = file_address

        for file_handle, file_address in self._file_handle_address.items():
            self._file_data[file_handle] = self._read_each_file(file_address)
        self.rm.files = self._file_data

    def _read_each_file(self, file_address, method=0):
        lnum = 0
        iTitle = True  # noqa: N806
        data_temp = None
        if method == 0:
            try:
                raise  # noqa: PLE0704
                with open(file_address, encoding='utf-8') as f:  # noqa: PTH123
                    for line in f:
                        line = line.strip()  # noqa: PLW2901
                        nwords = len(line.split())
                        if len(line) == 0 or nwords == 0:
                            # Blank line
                            continue
                        elif line.startswith(';'):  # noqa: RET507
                            # comment
                            continue
                        else:
                            lnum += 1
                            vals = line.split()
                            if iTitle == True:  # noqa: E712
                                iTitle = False  # noqa: N806
                                data_temp = pd.DataFrame(columns=vals)
                            else:
                                data_temp.loc[lnum - 2] = vals
            except:  # noqa: E722
                data_temp = self._read_each_file(file_address, method=1)
        elif method == 1:
            data_temp = pd.read_csv(file_address)
        else:
            raise ValueError('Uknown method: ' + str(method))
        return data_temp

    def _read_shifts(self):
        for lnum, line in self.sections['[SHIFTS]']:  # noqa: B007
            # edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) != 3:  # noqa: PLR2004
                    raise RuntimeError(  # noqa: TRY003
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'  # noqa: EM101
                    )
                shift_name = words[0]
                shift_begining = int(words[1]) * 3600
                shift_ending = int(words[2]) * 3600

                self.rm.shift[shift_name] = (shift_begining, shift_ending)

    def _read_entities(self):  # noqa: C901
        """Reads damage group definitions and updates the Restoration Model
        object data.

        Raises
        ------
        RuntimeError
            If the number of damages are not right.
        ValueError
            If the input data is not correctly provided.

            If the input data is not correctly provided.

        Returns
        -------
        None.

        """  # noqa: D205, D401, DOC202
        # Entities is kept for legacy compatibility with the first version
        damage_group_data = self.sections.get(
            '[ENTITIES]', self.sections.get('[Damage Group]')
        )

        for lnum, line in damage_group_data:
            arg1 = None
            arg2 = None
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) != 2 and len(words) != 4:  # noqa: PLR2004
                    raise RuntimeError(  # noqa: TRY003
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'  # noqa: EM101
                    )
                entity_name = words[0]
                element = words[1].upper()

                if element not in ELEMENTS:
                    raise ValueError('Unknown element line number ' + str(lnum))

                # if entity_name in self.rm.entity:
                # raise ValueError('Entity already defined')

                if len(words) == 4:  # noqa: PLR2004
                    arg1 = words[2]
                    arg2 = words[3]

                    # if (element=='PIPE' and arg1 not in self.rm._registry._pipe_damage_table.columns and arg1!='FILE' and arg1!='NOT_IN_FILE') and (element=='DISTNODE' and arg1 not in self.rm._registry._node_damage_table.columns):
                    # raise ValueError('Argument 1('+arg1+') is not recognized in line number: ' + str(lnum))

                if arg1 == None:  # noqa: E711
                    self.rm.entity[entity_name] = element
                    ent_rule = [('ALL', None, None)]

                    if entity_name not in self.rm.entity_rule:
                        self.rm.entity_rule[entity_name] = ent_rule
                    else:
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

                    # sina: take care of this in registry opening
                    # self.rm._registry.addAttrToElementDamageTable(element ,entity_name , True)

                elif arg1 == 'FILE' or arg1 == 'NOT_IN_FILE':  # noqa: PLR1714
                    name_list = self.rm.files[arg2]['ElementID'].unique().tolist()
                    ent_rule = [(arg1, None, name_list)]
                    self.rm.entity[entity_name] = element

                    if entity_name not in self.rm.entity_rule:
                        self.rm.entity_rule[entity_name] = ent_rule
                        # self.rm._registry.addAttrToElementDamageTable(element ,entity_name , True)
                    else:
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

                else:
                    if ':' in arg2:
                        split_arg = arg2.split(':')

                        if len(split_arg) != 2:  # noqa: PLR2004
                            raise ValueError(
                                'There must be two parts: PART1:PART2. Now there are '
                                + repr(
                                    len(split_arg)
                                    + ' parts. Line number is '
                                    + repr(lnum)
                                )
                            )
                        if split_arg[0] == '':
                            raise ValueError(
                                'The first part is Empty in line ' + repr(lnum)
                            )
                        if split_arg[1] == '':
                            raise ValueError(
                                'The second part is Empty in line ' + repr(lnum)
                            )
                    else:
                        raise ValueError(
                            'There must be two parts as a condition, separated with ":". Example: PART1:PART2 \nPart1 can be one of the following: EQ, BG, LT, BG-EQ, and LT-EQ. Line number: '
                            + repr(lnum)
                        )

                    rest_of_args = arg2.split(':')
                    arg2 = rest_of_args[0]
                    arg3 = rest_of_args[1]

                    try:
                        temp_arg3 = float(arg3)
                    except:  # noqa: E722
                        temp_arg3 = str(arg3)

                    arg3 = temp_arg3
                    ent_rule = [(arg1, arg2, arg3)]
                    if entity_name not in self.rm.entity:
                        self.rm.entity[entity_name] = element
                        self.rm.entity_rule[entity_name] = ent_rule
                        # self.rm._registry.addAttrToElementDamageTable(element ,entity_name , True)
                    else:
                        if self.rm.entity[entity_name] != element:
                            raise ValueError(
                                'Element must not change in an added condition. Line '
                                + str(lnum)
                            )
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

    def _read_sequences(self):
        # sina: there is a part that you need to add in restroation init
        for lnum, line in self.sections['[SEQUENCES]']:  # noqa: B007
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                # if len(words) != 2 or len(words)!=4:
                # raise RuntimeError('%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s' % edata)
                element = words[0].upper()
                seq = []
                for arg in words[1:]:
                    seq.append(arg)  # noqa: PERF402
                if element in self.rm.sequence:
                    raise ValueError('Element already in sequences')  # noqa: EM101, TRY003
                if element not in ELEMENTS:
                    raise ValueError(
                        'The Element '
                        + repr(element)
                        + ' is not a recognized element'
                    )
                self.rm.sequence[element] = seq

    def _read_agents(self):
        agent_file_handle = {}
        group_names = {}
        group_column = {}

        crews_data = self.sections.get('[AGENTS]', self.sections.get('CREWS'))
        for lnum, line in crews_data:  # noqa: B007
            # edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                _group_name = None
                _group_column = None

                if len(words) < 3:  # noqa: PLR2004
                    raise RuntimeError(  # noqa: TRY003
                        'less than three argument is not valid for crew definition'  # noqa: EM101
                    )
                agent_type = words[0]
                if words[1].upper() == 'FILE':
                    agent_file_handle[words[0]] = words[2]
                else:
                    raise ValueError('Unknown key')  # noqa: EM101, TRY003
                if len(words) >= 4:  # noqa: PLR2004
                    group_data = words[3]
                    _group_name = group_data.split(':')[0]
                    _group_column = group_data.split(':')[1]

                group_names[agent_type] = _group_name
                group_column[agent_type] = _group_column

        for agent_type, file_handle in agent_file_handle.items():
            data = self._file_data[file_handle]

            # print(file_handle)
            # print(self._file_data[file_handle])

            agent_number = data['Number']
            j = 0
            for lnum, line in data.iterrows():  # noqa: B007
                # try:
                num = int(agent_number[j])
                # except :
                # print('exception')
                # pass
                _r = range(num)

                for i in _r:
                    agent_name = agent_type + str(j) + str(i)
                    predefinitions = line.to_dict()
                    definitions = {}
                    definitions['cur_x'] = predefinitions['Curr.-X-Coord']
                    definitions['cur_y'] = predefinitions['Curr.-Y-Coord']
                    definitions['base_x'] = predefinitions['Home-X-Coord']
                    definitions['base_y'] = predefinitions['Home-Y-Coord']
                    definitions['shift_name'] = predefinitions['Shift']

                    group_name_temp = None
                    if group_names[agent_type] != None:  # noqa: E711
                        definitions['group'] = predefinitions[
                            group_column[agent_type]
                        ]
                        group_name_temp = group_names[agent_type]
                    else:
                        group_name_temp = 'default'
                        definitions['group'] = 'Default'

                    definitions['group_name'] = group_name_temp
                    self.rm.agents.append((agent_name, agent_type, definitions))
                j += 1  # noqa: SIM113

    def _read_groups(self):
        for lnum, line in self.sections['[GROUPS]']:
            words, comments = _split_line(line)

            if words is not None and len(words) > 0:
                if not len(words) >= 6:  # noqa: PLR2004
                    raise ValueError('error in line: ' + str(lnum))
                group_name = words[0]
                element_type = words[1]
                argument = words[2]
                file_handler = words[3]
                element_col_ID = words[4]  # noqa: N806
                pipe_col_ID = words[5]  # noqa: N806

                if element_type not in ELEMENTS:
                    raise ValueError(
                        'Unknown element type: '
                        + repr(element_type)
                        + ', in line: '
                        + repr(lnum)
                    )
                if argument != 'FILE':
                    raise ValueError(
                        'the Only acceptable argument is FILE. Not: '
                        + repr(argument)
                        + '. Line: '
                        + repr(lnum)
                    )

                data = self.rm.files[file_handler]

                if pipe_col_ID not in data:
                    raise ValueError(
                        repr(pipe_col_ID)
                        + 'not in file handle='
                        + repr(file_handler)
                    )

                group_list = data[pipe_col_ID]
                group_list.index = data[element_col_ID]

                if element_type not in self.rm.group:
                    raise ValueError(
                        'This error must never happen: ' + repr(element_type)
                    )

                if group_name in self.rm.group[element_type]:
                    raise ValueError(
                        'The Group is already identified: '
                        + repr(group_name)
                        + ' in line: '
                        + repr(lnum)
                    )

                self.rm.group[element_type][group_name] = group_list

    def _read_priorities(self):  # noqa: C901
        for lnum, line in self.sections['[PRIORITIES]']:
            words, comments = _split_line(line)

            if words is not None and len(words) > 0:
                if not len(words) >= 3:  # noqa: PLR2004
                    raise ValueError('error in line: ' + str(lnum))
                agent_type = words[0]

                priority = None
                try:
                    priority = int(words[1])
                except:  # noqa: E722
                    print('exeption handled in _read_priorities')  # noqa: T201
                if type(priority) != int:  # noqa: E721
                    raise ValueError(
                        'Priority casting failed:'
                        + str(priority)
                        + 'in line: '
                        + repr(lnum)
                    )
                arg = []
                for word in words[2:]:
                    temp = None  # noqa: F841
                    if word.find(':') != -1:
                        split_temp = word.split(':')
                        arg.append((split_temp[0], split_temp[1]))
                        if split_temp[1] not in self.rm.entity:
                            raise ValueError(
                                'Entity value is used which is not defined before: '
                                + split_temp[1]
                                + ', Line: '
                                + str(lnum)
                            )
                        if (
                            split_temp[0]
                            not in self.rm.sequence[self.rm.entity[split_temp[1]]]
                        ):
                            raise ValueError(
                                'There is no action: '
                                + repr(split_temp[0])
                                + ' in element: '
                                + repr(self.rm.entity[split_temp[1]])
                            )
                    else:
                        arg.append(word)
                        if word not in ['EPICENTERDIST', 'WaterSource']:
                            raise ValueError('Unnown value in line: ' + str(lnum))

                self.rm.priority.append((agent_type, priority, arg))

    def _read_jobs(self):
        for lnum, line in self.sections['[JOBS]']:
            words, comments = _split_line(line)

            if words is not None and len(words) > 0:
                if not len(words) >= 3:  # noqa: PLR2004
                    raise ValueError(
                        'Not enough arguments. error in line: ' + str(lnum)
                    )
                agent_type = words[0]

                action_entity = words[1]
                if not action_entity.find(':') != -1:
                    raise ValueError(
                        'There must be an action and entity separated by : in line '
                        + str(lnum)
                    )
                split_temp = action_entity.split(':')
                action = split_temp[0]
                entity = split_temp[1]

                definer_arg = words[2]
                if not definer_arg.find(':') != -1:
                    raise ValueError(
                        'There must be an Time Definer and Argument separated by : in line '
                        + str(lnum)
                    )
                split_temp = definer_arg.split(':')
                definer = split_temp[0]
                argument = split_temp[1]

                if definer.upper() == 'FIXED':
                    try:
                        argument = int(argument)
                    except:  # noqa: E722
                        print('exeption handled in _read_jobs')  # noqa: T201
                else:
                    raise ValueError('Definer is not recognized: ' + definer)

                effect = None
                if len(words) >= 4:  # noqa: PLR2004
                    effect = words[3]

                self.rm.jobs.append((agent_type, entity, action, argument, effect))

    def _read_define(self):  # noqa: C901, PLR0912
        job = {}  # noqa: F841

        effect_data = self.sections.get('[DEFINE]', self.sections.get('[EFFECTS]'))
        for lnum, line in effect_data:
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                # if not len(words) >= 3:
                # raise ValueError('Not enough arguments. error in line: ' + str(lnum))
                job_name = words[0]

                try:
                    method_name = float(words[1])
                except:  # noqa: E722
                    method_name = words[1]

                res_list = []
                flag = False

                if method_name == 'FILE':
                    file_data = self._read_file_effect(words[2:], job_name)
                    self.rm.jobs.append((job_name, 'DATA', file_data))
                    continue

                method_data_list = words[2:]
                for method_data in method_data_list:
                    res = {}
                    definition = method_data.split(':')

                    i = 0
                    if len(definition) % 2 != 1:
                        raise ValueError('Error in line ' + str(lnum))

                    main_arg = None

                    while i < len(definition):
                        arg = definition[i].upper()
                        if i == 0:
                            main_arg = arg
                            i += 1
                            res['EFFECT'] = main_arg
                            continue
                        val = definition[i + 1].upper()

                        if main_arg == 'RECONNECT':
                            if arg == 'PIPESIZE':
                                if 'PIPESIZEFACTOR' in res:
                                    raise ValueError(  # noqa: TRY003
                                        'Either pipe size or pipe size factor can be defined'  # noqa: EM101
                                    )
                                res['PIPESIZE'] = float(val)

                            elif arg == 'PIPESIZEFACTOR':
                                if 'PIPESIZE' in res:
                                    raise ValueError(  # noqa: TRY003
                                        'Either pipe size or pipe size factor can be defined'  # noqa: EM101
                                    )
                                val = float(val)
                                if val > 1 or val < 0:
                                    raise ValueError(
                                        'Pipe Size Factor must be bigger than 0 and less than or eqal to 1: '
                                        + str(val)
                                    )
                                res['PIPESIZEFACTOR'] = float(val)
                            elif arg == 'CV':
                                if val == 'TRUE' or val == '1':  # noqa: PLR1714
                                    val = True
                                elif val == 'FALSE' or val == '0':  # noqa: PLR1714
                                    val = False
                                else:
                                    raise ValueError(
                                        'Unrecognized value for CV in line '
                                        + str(lnum)
                                        + ': '
                                        + val
                                        + (
                                            'Value for CV must be either True or False'
                                        )
                                    )
                                res['CV'] = val
                            elif arg == 'PIPELENGTH':
                                try:
                                    val == float(val)  # noqa: B015
                                except Exception as e:
                                    print(  # noqa: T201
                                        'The value for PIPELENGTH must be a number'
                                    )
                                    raise e  # noqa: TRY201
                                res['PIPELENGTH'] = val
                            elif arg == 'PIPEFRICTION':
                                try:
                                    val == float(val)  # noqa: B015
                                except Exception as e:
                                    print(  # noqa: T201
                                        'The value for PIPEFRICTION must be a number'
                                    )
                                    raise e  # noqa: TRY201
                                res['PIPEFRICTION'] = val
                            else:
                                raise ValueError(
                                    'Unrecognized argument: '
                                    + arg
                                    + ', in effect: '
                                    + main_arg
                                )
                        elif main_arg == 'ADD_RESERVOIR':
                            if arg == 'PUMP':
                                res['PUMP'] = float(val)

                            elif arg == 'CV':
                                if val == 'TRUE' or val == '1':  # noqa: PLR1714
                                    val = True
                                elif val == 'FALSE' or val == '0':  # noqa: PLR1714
                                    val = False
                                else:
                                    raise ValueError(
                                        'Unrecognized value for CV in line '
                                        + str(lnum)
                                        + ': '
                                        + val
                                        + (
                                            'Value for CV must be either True or False'
                                        )
                                    )
                                res['CV'] = val
                            elif arg == 'ADDEDELEVATION':
                                val = float(val)
                                res['ADDEDELEVATION'] = float(val)
                            else:
                                raise ValueError(
                                    'Unrecognized argument: '
                                    + arg
                                    + ', in effect: '
                                    + main_arg
                                )
                        elif main_arg == 'REMOVE_LEAK':
                            if arg == 'LEAKFACTOR':
                                val = float(val)
                                if val > 1 or val <= 0:
                                    raise ValueError(
                                        'Leak factor must be bigger than 0 and less than or eqal to 1: '
                                        + str(val)
                                    )
                                res['LEAKFACTOR'] = val
                            else:
                                raise ValueError(
                                    'Unrecognized argument: '
                                    + arg
                                    + ', in effect: '
                                    + main_arg
                                )

                        elif main_arg == 'COL_CLOSE_PIPE':
                            raise ValueError(  # noqa: TRY003
                                'REPAIR at this stage does not accept any argument'  # noqa: EM101
                            )

                        elif main_arg == 'ISOLATE_DN':
                            if arg == 'PIDR':  # Post Incident Demand Ratio
                                if (
                                    val[0] != '('
                                    or val[-1] != ')'
                                    or val.find(',') == -1
                                ):
                                    ValueError(  # noqa: PLW0133
                                        'After PIDR the format must be like (CONDIION,VALUE)'
                                    )

                                val = val.strip('(').strip(')')
                                val_split = val.split(',')
                                _con = val_split[0].upper()
                                _con_val = float(val_split[1])

                                if not (
                                    _con == 'BG'  # noqa: PLR1714
                                    or _con == 'EQ'
                                    or _con == 'LT'
                                    or _con == 'BG-EQ'
                                    or _con == 'LT-EQ'
                                ):
                                    raise ValueError(
                                        'Condition is not recognized:' + str(_con)
                                    )

                                if _con_val < 0:
                                    raise ValueError(
                                        'PIDR condition value cannot be less than zero-->'
                                        + repr(_con_val)
                                    )

                                res['PIDR'] = (_con, _con_val)

                        elif main_arg == 'REPAIR':
                            raise ValueError(  # noqa: TRY003
                                'REPAIR at this stage does not accept any argument'  # noqa: EM101
                            )

                        elif method_name.upper() == 'DEFAULT':
                            try:  # noqa: SIM105
                                arg = int(arg)
                            except:  # noqa: S110, E722
                                pass

                            if main_arg == 'METHOD_PROBABILITY':
                                val = float(val)

                                if val < 0:
                                    raise ValueError(
                                        'Probability cannot be less than zero. '  # noqa: ISC003
                                        + ' In line  '
                                        + lnum
                                        + ' probability: '
                                        + val
                                    )
                                elif val > 1:  # noqa: RET506
                                    raise ValueError(
                                        'Probability cannot be bigger than 1. '  # noqa: ISC003
                                        + ' In line  '
                                        + lnum
                                        + ' probability: '
                                        + val
                                    )
                                temp = {
                                    'effect_definition_name': job_name,
                                    'method_name': arg,
                                    'argument': main_arg,
                                    'value': val,
                                }
                                self.rm.jobs_default.append(temp)
                                # temp={'effect_definition_name':effect_name, 'method_name':arg,'argument':'METHOD_PROBABILITY','value':val}
                            elif main_arg == 'FINALLY':
                                if val.upper() == 'NULL':
                                    val = None
                                else:
                                    val = None
                                    print(  # noqa: T201
                                        'WARNING: At default line in FINAL section, the third argument is not NULL: '
                                        + str(val)
                                        + 'The value is ignored antywhere'
                                    )
                                self.rm.final_method[job_name] = arg
                            elif main_arg == 'ONLYONCE':
                                try:  # noqa: SIM105
                                    val = float(val)
                                except:  # noqa: S110, E722
                                    pass

                                if job_name in self.rm.once:
                                    self.rm.once[job_name].append(val)
                                else:
                                    self.rm.once[job_name] = [val]
                            else:
                                raise ValueError(
                                    'Unrecognized argument in line '
                                    + str(lnum)
                                    + ': '
                                    + arg
                                )

                            flag = True
                        else:
                            raise ValueError(
                                'Unrecognized argument in line '
                                + str(lnum)
                                + ': '
                                + arg
                            )

                        i += 2
                    res_list.append(res)
                if flag == False:  # noqa: E712
                    self.rm.jobs.append((job_name, method_name, res_list))

        # for self.rm.effects.pruneData()

    def _read_file_effect(self, file_info, effect_name):
        res = {}

        file_handle = file_info[0]
        file_data = file_info[1:]

        data = self.rm.files[file_handle]

        # columns_to_remove = data.columns.tolist()
        aliases = {}

        for pair in file_data:
            if not pair.find(':'):
                raise ValueError('Error in file info. Not Pair: ' + pair)
            _arg, val = pair.split(':')
            arg = _arg.upper()

            if arg in res:
                raise ValueError('Argument already added: ' + _arg)

            if val not in data.columns:
                raise ValueError('Value not in file: ' + val)
            if (
                arg == 'ELEMENT_NAME'  # noqa: PLR1714
                or arg == 'METHOD_NAME'
                or arg == 'METHOD_PROBABILITY'
            ):
                aliases[arg] = val
                res[arg] = data[val].to_dict()

            elif arg == 'FIXED_TIME_OVERWRITE':
                time_overwrite_data = data[val].to_list()
                # self.rm.jobs._job_list[self.rm.jobs._job_list['effect']==effect_name]
                temp_list_for_effect_name = [effect_name] * data[val].size
                _key = list(
                    zip(
                        temp_list_for_effect_name,
                        data[aliases['METHOD_NAME']],
                        data[aliases['ELEMENT_NAME']],
                    )
                )

                time_overwrite_data = [
                    {'FIXED_TIME_OVERWRITE': int(time_overwrite_data[i] * 3600)}
                    for i in range(len(time_overwrite_data))
                ]
                self.rm.time_overwrite.update(
                    pd.Series(index=_key, data=time_overwrite_data).to_dict()
                )

            else:
                raise ValueError('Unrecognized argument in pair: ' + _arg)
        res = pd.DataFrame(res)
        # print(res)
        return res  # noqa: RET504

    def _read_demand_nodes(self):
        titles = []  # noqa: F841
        ntitle = 0
        lnum = 0
        dtemp = []
        with open(self._demand_Node_file_name, encoding='utf-8') as f:  # noqa: PTH123
            for line in f:
                lnum += 1
                line = line.strip()  # noqa: PLW2901
                nwords = len(line.split())
                words = line.split()
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.upper().startswith('NODEID'):  # noqa: RET507
                    title = words.copy()
                    ntitle = len(
                        words
                    )  # we need this to confirm that every line has data for every title(column)
                    continue
                elif nwords != ntitle:
                    raise ValueError(  # noqa: TRY003
                        '%{fname}s:%(lnum)d: Number of data does not match number of titles'  # noqa: EM101
                    )
                elif nwords == ntitle:
                    dtemp.append(words)
                else:
                    raise ValueError(  # noqa: TRY003
                        '%{fname}s:%(lnum)d:This error must nnever happen'  # noqa: EM101
                    )
            self.demand_node = pd.DataFrame(dtemp, columns=title)

    def _read_crew(self):
        titles = []  # noqa: F841
        ntitle = 0
        lnum = 0
        dtemp = []
        with open(self._crew_file_name[-1], encoding='utf-8') as f:  # noqa: PTH123
            for line in f:
                lnum += 1
                line = line.strip()  # noqa: PLW2901
                nwords = len(line.split())
                words = line.split()
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.upper().startswith('DISTYARDID'):  # noqa: RET507
                    title = words.copy()
                    ntitle = len(
                        words
                    )  # we need this to confirm that every line has data for every title(column)
                    continue
                elif nwords != ntitle:
                    raise ValueError(  # noqa: TRY003
                        '%{fname}s:%(lnum)d: Number of data does not match number of titles'  # noqa: EM101
                    )
                elif nwords == ntitle:
                    dtemp.append(words)
                else:
                    raise ValueError(  # noqa: TRY003
                        '%{fname}s:%(lnum)d:This error must nnever happen'  # noqa: EM101
                    )
            self.crew_data[self._crew_file_type[-1]] = pd.DataFrame(
                dtemp, columns=title
            )
