# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 19:10:35 2020

@author: snaeimi
"""

import io
import logging
import pandas as pd
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)


# the follwing function is borrowed from WNTR
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


class RestorationIO:
    def __init__(self, restoration_model, definition_file_name):
        """
        Needs a file that contains:

        Parameters
        ----------
        restoratio_model : Restoration object
            restoration model to be defined by files
        definition_file : str
            path for tegh definition file.

        Returns
        -------
        None.

        """

        # some of the following lines have been addopted from WNTR
        self.rm = restoration_model
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
            '[POINTS]',
        ]

        self.config_file_comment = []
        self.edata = []

        self.sections = OrderedDict()
        for sec in expected_sections:
            self.sections[sec] = []

        section = None
        lnum = 0
        edata = {'fname': definition_file_name}
        # Sprint(definition_file_name)
        config_file_path = Path(definition_file_name)

        if config_file_path.is_absolute():
            pass
        else:
            config_file_path = config_file_path.resolve()

        self.config_file_dir = config_file_path.parent

        with io.open(definition_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                lnum += 1
                edata['lnum'] = lnum
                line = line.strip()
                nwords = len(line.split())
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.startswith('['):
                    vals = line.split()
                    sec = vals[0].upper()
                    edata['sec'] = sec
                    if sec in expected_sections:
                        section = sec
                        continue
                    else:
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
        self._read_points()
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
                if len(words) != 2:
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
        self.rm._files = self._file_data

    def _read_each_file(self, file_address, method=0):
        lnum = 0
        iTitle = True
        data_temp = None
        if method == 0:
            try:
                raise
                with io.open(file_address, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        nwords = len(line.split())
                        if len(line) == 0 or nwords == 0:
                            # Blank line
                            continue
                        elif line.startswith(';'):
                            # comment
                            continue
                        else:
                            lnum += 1
                            vals = line.split()
                            if iTitle == True:
                                iTitle = False
                                data_temp = pd.DataFrame(columns=vals)
                            else:
                                data_temp.loc[lnum - 2] = vals
            except:
                data_temp = self._read_each_file(file_address, method=1)
        elif method == 1:
            file_address = self.config_file_dir / file_address
            data_temp = pd.read_csv(file_address)
        else:
            raise ValueError('Uknown method: ' + str(method))
        return data_temp

    def _read_shifts(self):
        # self._shift_data=pd.DataFrame()
        # self._file_handle_address = {}
        for lnum, line in self.sections['[SHIFTS]']:
            # edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) != 3:
                    raise RuntimeError(
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'
                    )
                shift_name = words[0]
                shift_begining = int(words[1]) * 3600
                shift_ending = int(words[2]) * 3600

                self.rm.shifting.addShift(shift_name, shift_begining, shift_ending)

    def _read_entities(self):
        for lnum, line in self.sections['[ENTITIES]']:
            arg1 = None
            arg2 = None
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) != 2 and len(words) != 4:
                    raise RuntimeError(
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'
                    )
                entity_name = words[0]
                element = words[1].upper()

                if element not in self.rm.ELEMENTS:
                    raise ValueError('Unknown element line number ' + str(lnum))

                # if entity_name in self.rm.entity:
                # raise ValueError('Entity already defined')

                if len(words) == 4:
                    arg1 = words[2]
                    arg2 = words[3]

                    if (
                        element == 'PIPE'
                        and arg1 not in self.rm._registry._pipe_damage_table.columns
                        and arg1 != 'FILE'
                        and arg1 != 'NOT_IN_FILE'
                    ) and (
                        element == 'DISTNODE'
                        and arg1 not in self.rm._registry._node_damage_table.columns
                    ):
                        raise ValueError(
                            'Argument 1('
                            + arg1
                            + ') is not recognized in line number: '
                            + str(lnum)
                        )

                if arg1 == None:
                    self.rm.entity[entity_name] = element
                    ent_rule = [('ALL', None, None)]

                    if entity_name not in self.rm.entity_rule:
                        self.rm.entity_rule[entity_name] = ent_rule
                    else:
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

                    self.rm._registry.addAttrToElementDamageTable(
                        element, entity_name, True
                    )

                elif arg1 == 'FILE' or arg1 == 'NOT_IN_FILE':
                    name_list = self.rm._files[arg2]['ElementID'].unique().tolist()
                    ent_rule = [(arg1, None, name_list)]
                    self.rm.entity[entity_name] = element

                    if entity_name not in self.rm.entity_rule:
                        self.rm.entity_rule[entity_name] = ent_rule
                        self.rm._registry.addAttrToElementDamageTable(
                            element, entity_name, True
                        )
                    else:
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

                else:
                    if ':' in arg2:
                        split_arg = arg2.split(':')

                        if len(split_arg) != 2:
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
                            'There must be two parts as a conditio, separted with ":". Example: PART1:PART2 \nPart1 can be one of teh following: EQ, BG, LT, BG-EQ, and LT-EQ. Line number: '
                            + repr(lnum)
                        )

                    rest_of_args = arg2.split(':')
                    arg2 = rest_of_args[0]
                    arg3 = rest_of_args[1]

                    try:
                        temp_arg3 = float(arg3)
                    except:
                        temp_arg3 = str(arg3)

                    arg3 = temp_arg3
                    ent_rule = [(arg1, arg2, arg3)]
                    if entity_name not in self.rm.entity:
                        self.rm.entity[entity_name] = element
                        self.rm.entity_rule[entity_name] = ent_rule
                        self.rm._registry.addAttrToElementDamageTable(
                            element, entity_name, True
                        )
                    else:
                        if self.rm.entity[entity_name] != element:
                            raise ValueError(
                                'Element must not chanage in an added condition. Line '
                                + str(lnum)
                            )
                        self.rm.entity_rule[entity_name].append(ent_rule[0])

    # =============================================================================
    #                     if element == 'PIPE':
    #                         #candidate = self.getDamageData(element)
    #                         #candidate.index.tolist()
    #                         ent_rule = [element,'ALL']
    #                         #self.rm.entity[entity_name]      = element
    #                         self.rm.entity_rule[entity_name] = ent_rule
    #                         self.rm._registry.addAttrToPipeDamageTable(entity_name, True)
    #                     elif element == 'DISTNODE':
    #                         ent_rule = [element,'ALL']
    #                         #self.rm.entity[entity_name]
    #                         self.rm.entity_rule[entity_name] = ent_rule
    #                         self.rm._registry.AttrToDistNodeDamageTable(entity_name, True)
    #                     else:
    #                         raise ValueError('Element type is not recognized')
    # =============================================================================

    def _read_sequences(self):
        for lnum, line in self.sections['[SEQUENCES]']:
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                # if len(words) != 2 or len(words)!=4:
                # raise RuntimeError('%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s' % edata)
                element = words[0].upper()
                seq = []
                for arg in words[1:]:
                    seq.append(arg)
                if element in self.rm.sequence:
                    raise ValueError('Element already in sequences')
                self.rm.sequence[element] = seq
        for el in self.rm.sequence:
            if el in self.rm.ELEMENTS:
                for action in self.rm.sequence[el]:
                    self.rm._registry.addAttrToElementDamageTable(el, action, None)

    def _read_agents(self):
        agent_file_handle = {}
        group_names = {}
        group_column = {}

        for lnum, line in self.sections['[AGENTS]']:
            # edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                _group_name = None
                _group_column = None

                if len(words) < 3:
                    raise RuntimeError(
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'
                    )
                agent_type = words[0]
                if words[1].upper() == 'FILE':
                    agent_file_handle[words[0]] = words[2]
                else:
                    raise ValueError('Unknown key')
                if len(words) >= 4:
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
            for lnum, line in data.iterrows():
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
                    if group_names[agent_type] != None:
                        definitions['group'] = predefinitions[
                            group_column[agent_type]
                        ]
                        group_name_temp = group_names[agent_type]
                    else:
                        group_name_temp = 'default'
                        definitions['group'] = 'Default'

                    definitions['group_name'] = group_name_temp
                    self.rm.agents.addAgent(agent_name, agent_type, definitions)
                j += 1

    def _read_groups(self):
        for lnum, line in self.sections['[GROUPS]']:
            words, comments = _split_line(line)

            if words is not None and len(words) > 0:
                if len(words) != 6:
                    raise ValueError(
                        'error in line: ' + str(lnum) + ': ' + repr(len(words))
                    )
                group_name = words[0]
                element_type = words[1]
                arguement = words[2]
                file_handler = words[3]
                element_col_ID = words[4]
                pipe_col_ID = words[5]

                if element_type not in self.rm.ELEMENTS:
                    raise ValueError(
                        'Unknown element type: '
                        + repr(element_type)
                        + ', in line: '
                        + repr(lnum)
                    )
                if arguement != 'FILE':
                    raise ValueError(
                        'the Only acceptable argument is  FILE. Line: ' + repr(lnum)
                    )

                data = self.rm._files[file_handler]

                if pipe_col_ID not in data:
                    raise ValueError(
                        repr(pipe_col_ID)
                        + 'not in file handle='
                        + repr(file_handler)
                    )

                if element_col_ID not in data:
                    raise ValueError(
                        repr(element_col_ID)
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

    def _read_points(self):
        for lnum, line in self.sections['[POINTS]']:
            words, comments = _split_line(line)

            if words is None or len(words) < 1:  # Empty Line
                continue

            if not len(words) >= 2:  # Syntax Error
                raise ValueError(
                    'Syntax error in line: '
                    + str(lnum)
                    + '\n'
                    + "Each Point Group must have at least one name and one point coordinate sperated by a ':'"
                    + '\n'
                    + "Example= 'PointGroupName X1:Y1 [X2:Y2 ...]'"
                )

            group_name = words[0]
            current_group_point_list = []

            if group_name.upper() in self.rm.reserved_priority_names:
                raise ValueError(
                    'Syntax error in line: '
                    + str(lnum)
                    + '\n'
                    + 'Group name '
                    + "'"
                    + group_name
                    + "'"
                    + ' is ambiguous. '
                    + "'"
                    + group_name
                    + ' is a reserved priority'
                )

            for word in words[1:]:
                if ':' not in word:
                    raise ValueError(
                        'Syntax error in line: '
                        + str(lnum)
                        + '\n'
                        + "'"
                        + word
                        + "'"
                        + " is not an accpetable point coordinate. It must be point coordinate sperated by a ':'"
                        + '\n'
                        + "Example= 'X1:Y1'"
                    )

                x_y_coord = word.split(':')
                if len(x_y_coord) > 2:
                    raise ValueError(
                        'Syntax error in line: '
                        + str(lnum)
                        + '\n'
                        + "Multiple devider (':') in "
                        + "'"
                        + word
                        + "'"
                        + "It must be point coordinate sperated by a ':'"
                        + '\n'
                        + "Example= 'X1:Y1'"
                    )

                x_coord = x_y_coord[0]
                y_coord = x_y_coord[1]

                try:
                    x_coord = float(x_coord)
                except:
                    raise ValueError(
                        'Syntax error in line: '
                        + str(lnum)
                        + '\n'
                        + "'"
                        + x_coord
                        + "'"
                        + ' in '
                        "'" + word + "'" + ' is not a number'
                    )

                try:
                    y_coord = float(y_coord)
                except:
                    raise ValueError(
                        'Syntax error in line: '
                        + str(lnum)
                        + '\n'
                        + "'"
                        + y_coord
                        + "'"
                        + ' in '
                        "'" + word + "'" + ' is not a number'
                    )

                current_group_point_list.append((x_coord, y_coord))
            # print(group_name)
            # print(words[1:])
            if (
                group_name in self.rm.proximity_points
            ):  # To Support mutiple line assigment of the same group
                self.rm.proximity_points[group_name].extend(current_group_point_list)
            else:
                self.rm.proximity_points[group_name] = current_group_point_list

    def _read_priorities(self):
        agent_type_list = self.rm.agents.getAllAgentTypes()
        for lnum, line in self.sections['[PRIORITIES]']:
            words, comments = _split_line(line)

            if words is None or len(words) < 1:
                continue

            if not len(words) >= 3:
                raise ValueError(
                    'Syntax error in line: '
                    + str(lnum)
                    + '\n'
                    + 'Inadequate parametrs to define priority. There must be at least three parametrs, '
                    + repr(len(words))
                    + ' is given.'
                    + '\n'
                    + "Example= 'CREW TYPE   PriorityType[1 or 2], Action:DamageGroup"
                )

            agent_type = words[0]

            if agent_type not in agent_type_list:
                raise ValueError(
                    'Logical error in line: '
                    + str(lnum)
                    + '\n'
                    + 'Crew type '
                    + "'"
                    + agent_type
                    + "'"
                    + ' is not defiend in the crew section.'
                )

            try:
                priority_type = int(words[1])
            except:
                try:
                    priority_type = int(float(words[1]))
                except:
                    raise ValueError(
                        'Syntax error in line: '
                        + str(lnum)
                        + '\n'
                        + "'"
                        + priority_type
                        + "'"
                        + ' is not an acceptable priority type. Priority type must be either 1 or 2 to define the first or secondary prioirty consecutively.'
                        + '\n'
                        + "Example= 'CREW TYPE   Prioritytype[1 or 2], Action:DamageGroup"
                    )

            if priority_type not in [1, 2]:
                raise ValueError(
                    'Syntax error in line: '
                    + str(lnum)
                    + '\n'
                    + "'"
                    + priority_type
                    + "'"
                    + ' is not an acceptable priority type. Priority type must be either 1 or 2 to define the first or secondary prioirty consecutively.'
                    + '\n'
                    + "Example= 'CREW TYPE   Prioritytype[1 or 2], Action:DamageGroup"
                )

            arg = []
            for word in words[2:]:
                if priority_type == 1:
                    if word.find(':') == -1:
                        raise ValueError(
                            'Syntax error in line: '
                            + str(lnum)
                            + '\n'
                            + "The devider (':') is lacking. The primary priority "
                            + "'"
                            + word
                            + "'"
                            + ' is not an acceptable Primary Priority. A Priority Priority is a consisted of an Action:DamageGroup.'
                            + '\n'
                            + "Example= 'CREW TYPE   Prioritytype[1], Action:DamageGroup"
                        )
                    split_temp = word.split(':')

                    if len(split_temp) > 2:
                        raise ValueError(
                            'Syntax error in line: '
                            + str(lnum)
                            + '\n'
                            + "More than one devider (':') In the Primary Priority. The primary priority "
                            + "'"
                            + word
                            + "'"
                            + ' is not an acceptable Primary Priority. A Priority Priority is a consisted of an Action:DamageGroup.'
                            + '\n'
                            + "Example= 'CREW TYPE   Prioritytype[1], Action:DamageGroup"
                        )

                    action = split_temp[0]
                    damage_group = split_temp[1]

                    if damage_group not in self.rm.entity:
                        raise ValueError(
                            'Logical error in line: '
                            + str(lnum)
                            + '\n'
                            + 'DamageGroup '
                            + "'"
                            + damage_group
                            + "'"
                            + ' is not an defined. A Priority Priority is a consisted of an Action:DamageGroup.'
                            + '\n'
                            + "Example= 'CREW TYPE   Prioritytype[1], Action:DamageGroup"
                        )

                    if action not in self.rm.sequence[self.rm.entity[damage_group]]:
                        raise ValueError(
                            'Logical error in line: '
                            + str(lnum)
                            + '\n'
                            + 'Action '
                            + "'"
                            + action
                            + "'"
                            + ' is not an defined in Action Sequence. A Priority Priority is a consisted of an Action:DamageGroup.'
                            + '\n'
                            + "Example= 'CREW TYPE   Prioritytype[1], Action:DamageGroup"
                        )

                    arg.append((action, damage_group))

                elif priority_type == 2:
                    if (
                        word not in self.rm.proximity_points
                        and word not in self.rm.reserved_priority_names
                    ):
                        raise ValueError(
                            'Logical error in line: '
                            + str(lnum)
                            + '\n'
                            + 'Secondary Priority '
                            + "'"
                            + word
                            + "'"
                            + ' is not defined as a Point Group and is not a Reserved Secondary Priority.'
                            + '\n'
                            + "Example= 'CREW TYPE   Prioritytype[2] ['Point Group' or 'Reserved Secondary Priority']"
                        )
                    arg.append(word)
                else:
                    raise ValueError('Uknown Priority type: ' + repr(priority_type))

            self.rm.priority.addData(agent_type, priority_type, arg)

        for crew_type in self.rm.priority._data:
            priority_list = self.rm.priority._data[crew_type]
            primary_priority_order_list = priority_list[1]
            secondary_priority_order_list = priority_list[2]
            if len(primary_priority_order_list) != len(
                secondary_priority_order_list
            ):
                raise ValueError(
                    'Logical error. The number of Primary Priority and Secondary Primary does not match for Crew Trye: '
                    + repr(crew_type)
                )

        not_defined = []
        for agent_type in agent_type_list:
            if not self.rm.priority.isAgentTypeInPriorityData(agent_type):
                not_defined.append(agent_type)

        if len(not_defined) > 0:
            raise ValueError(
                'Logical error. The following agent types are not defined in the prioirty sections:\n'
                + repr(not_defined)
            )

    def _read_jobs(self):
        jobs_definition = []
        for lnum, line in self.sections['[JOBS]']:
            cur_job_definition = {}
            words, comments = _split_line(line)

            if words is not None and len(words) > 0:
                if not len(words) >= 3:
                    raise ValueError(
                        'Not enough arguments. error in line: ' + str(lnum)
                    )
                agent_type = words[0]

                action_entity = words[1]
                if not action_entity.find(':') != -1:
                    raise ValueError(
                        'There must be an action and entity seprated by : in line '
                        + str(lnum)
                    )
                split_temp = action_entity.split(':')
                action = split_temp[0]
                entity = split_temp[1]

                definer_arg = words[2]
                if not definer_arg.find(':') != -1:
                    raise ValueError(
                        'There must be an Time Definer and Argument seprated by : in line '
                        + str(lnum)
                    )
                split_temp = definer_arg.split(':')
                definer = split_temp[0]
                argument = split_temp[1]

                if definer.upper() == 'FIXED':
                    try:
                        argument = int(argument)
                    except:
                        print('exeption handled in _read_jobs')
                else:
                    raise ValueError('Definer is not recognized: ' + definer)

                effect = None
                if len(words) >= 4:
                    effect = words[3]

                cur_job_definition = {
                    'agent_type': agent_type,
                    'entity': entity,
                    'action': action,
                    'time_argument': argument,
                    'effect': effect,
                }
                jobs_definition.append(cur_job_definition)
        self.rm.jobs.setJob(jobs_definition)

    def _read_define(self):
        job = {}
        used_jobs = self.rm.jobs._job_list.effect.unique().tolist()
        if None in used_jobs:
            used_jobs.remove(None)

        # for key in used_effect:
        # job[key]=[]
        for lnum, line in self.sections['[DEFINE]']:
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                # if not len(words) >= 3:
                # raise ValueError('Not enough arguments. error in line: ' + str(lnum))
                job_name = words[0]
                if job_name not in used_jobs:
                    raise ValueError(
                        'Effect name not recognized in line '
                        + str(lnum)
                        + ' : '
                        + job_name
                    )
                try:
                    method_name = float(words[1])
                except:
                    method_name = words[1]

                res_list = []
                flag = False

                if method_name == 'FILE':
                    file_data = self._read_file_effect(words[2:], job_name)
                    self.rm.jobs.addEffect(job_name, 'DATA', file_data)
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
                                    raise ValueError(
                                        'Either pipe size or pipe size factor can be defined'
                                    )
                                res['PIPESIZE'] = float(val)

                            elif arg == 'PIPESIZEFACTOR':
                                if 'PIPESIZE' in res:
                                    raise ValueError(
                                        'Either pipe size or pipe size factor can be defined'
                                    )
                                val = float(val)
                                if val > 1 or val < 0:
                                    raise ValueError(
                                        'Pipe Size Factor must be bigger than 0 and less than or eqal to 1: '
                                        + str(val)
                                    )
                                res['PIPESIZEFACTOR'] = float(val)
                            elif arg == 'CV':
                                if val == 'TRUE' or val == '1':
                                    val = True
                                elif val == 'FALSE' or val == '0':
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
                                    val == float(val)
                                except Exception as e:
                                    print(
                                        'The value for PIPELENGTH must be a number'
                                    )
                                    raise e
                                res['PIPELENGTH'] = val
                            elif arg == 'PIPEFRICTION':
                                try:
                                    val == float(val)
                                except Exception as e:
                                    print(
                                        'The value for PIPEFRICTION must be a number'
                                    )
                                    raise e
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
                                if val == 'TRUE' or val == '1':
                                    val = True
                                elif val == 'FALSE' or val == '0':
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
                            raise ValueError(
                                'REPAIR at this stage does not accept any argument'
                            )

                        elif main_arg == 'ISOLATE_DN':
                            if arg == 'PIDR':  # Post Incident Demand Ratio
                                if (
                                    val[0] != '('
                                    or val[-1] != ')'
                                    or val.find(',') == -1
                                ):
                                    ValueError(
                                        'After PIDR the format must be like (CONDIION,VALUE)'
                                    )

                                val = val.strip('(').strip(')')
                                val_split = val.split(',')
                                _con = val_split[0].upper()
                                _con_val = float(val_split[1])

                                if not (
                                    _con == 'BG'
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
                            raise ValueError(
                                'REPAIR at this stage does not accept any argument'
                            )

                        elif method_name.upper() == 'DEFAULT':
                            try:
                                arg = int(arg)
                            except:
                                pass

                            if main_arg == 'METHOD_PROBABILITY':
                                val = float(val)

                                if val < 0:
                                    raise ValueError(
                                        'Probability cannot be less than zero. '
                                        + ' In line  '
                                        + lnum
                                        + ' probability: '
                                        + val
                                    )
                                elif val > 1:
                                    raise ValueError(
                                        'Probability cannot be bigger than 1. '
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
                                self.rm.jobs.addEffectDefaultValue(temp)
                                # temp={'effect_definition_name':effect_name, 'method_name':arg,'argument':'METHOD_PROBABILITY','value':val}
                            elif main_arg == 'FINALLY':
                                if val.upper() == 'NULL':
                                    val = None
                                else:
                                    val = None
                                    print(
                                        'WARNING: At default line in FINALL section, the third argument is not NULL: '
                                        + str(val)
                                        + 'The value is ignored antywhere'
                                    )
                                self.rm.jobs._final_method[job_name] = arg
                            elif main_arg == 'ONLYONCE':
                                try:
                                    val = float(val)
                                except:
                                    pass

                                if job_name in self.rm.jobs._once:
                                    self.rm.jobs._once[job_name].append(val)
                                else:
                                    self.rm.jobs._once[job_name] = [val]
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
                if flag == False:
                    self.rm.jobs.addEffect(job_name, method_name, res_list)

        # for self.rm.effects.pruneData()

    def _read_file_effect(self, file_info, effect_name):
        res = {}

        file_handle = file_info[0]
        file_data = file_info[1:]

        data = self.rm._files[file_handle]

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
                arg == 'ELEMENT_NAME'
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
                self.rm.jobs._time_overwrite.update(
                    pd.Series(index=_key, data=time_overwrite_data).to_dict()
                )

            else:
                raise ValueError('Unrecognized argument in pair: ' + _arg)
        res = pd.DataFrame(res)
        # print(res)
        return res

    def _read_config(self):
        """
        reads config files which contains general specification of
        configurations

        Raises
        ------
        RuntimeError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        edata = OrderedDict()
        self._crew_file_name = []
        self._crew_file_type = []
        for lnum, line in self.sections['[FILES]']:
            edata['lnum'] = lnum
            words, comments = _split_line(line)
            if words is not None and len(words) > 0:
                if len(words) < 2:
                    edata['key'] = words[0]
                    raise RuntimeError(
                        '%(fname)s:%(lnum)-6d %(sec)13s no value provided for %(key)s'
                        % edata
                    )
                key = words[0].upper()

                if key == 'DEMAND_NODES':
                    self._demand_Node_file_name = words[1]
                    self._read_demand_nodes()
                if key == 'CREW':
                    self._crew_file_type.append(words[1])
                    self._crew_file_name.append(words[2])
                    self._read_crew()

    def _read_demand_nodes(self):
        titles = []
        ntitle = 0
        lnum = 0
        dtemp = []
        with io.open(self._demand_Node_file_name, 'r', encoding='utf-8') as f:
            for line in f:
                lnum += 1
                line = line.strip()
                nwords = len(line.split())
                words = line.split()
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.upper().startswith('NODEID'):
                    title = words.copy()
                    ntitle = len(
                        words
                    )  # we need this to confirm that every line has data for every title(column)
                    continue
                elif nwords != ntitle:
                    raise ValueError(
                        '%{fname}s:%(lnum)d: Number of data does not match number of titles'
                    )
                elif nwords == ntitle:
                    dtemp.append(words)
                else:
                    raise ValueError(
                        '%{fname}s:%(lnum)d:This error must nnever happen'
                    )
            self.demand_node = pd.DataFrame(dtemp, columns=title)

    def _read_crew(self):
        titles = []
        ntitle = 0
        lnum = 0
        dtemp = []
        with io.open(self._crew_file_name[-1], 'r', encoding='utf-8') as f:
            for line in f:
                lnum += 1
                line = line.strip()
                nwords = len(line.split())
                words = line.split()
                if len(line) == 0 or nwords == 0:
                    # Blank line
                    continue
                elif line.upper().startswith('DISTYARDID'):
                    title = words.copy()
                    ntitle = len(
                        words
                    )  # we need this to confirm that every line has data for every title(column)
                    continue
                elif nwords != ntitle:
                    raise ValueError(
                        '%{fname}s:%(lnum)d: Number of data does not match number of titles'
                    )
                elif nwords == ntitle:
                    dtemp.append(words)
                else:
                    raise ValueError(
                        '%{fname}s:%(lnum)d:This error must nnever happen'
                    )
            self.crew_data[self._crew_file_type[-1]] = pd.DataFrame(
                dtemp, columns=title
            )
