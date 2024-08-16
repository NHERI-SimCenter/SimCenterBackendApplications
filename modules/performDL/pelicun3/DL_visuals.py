#  # noqa: INP001, D100
# Copyright (c) 2023 Leland Stanford Junior University
# Copyright (c) 2023 The Regents of the University of California
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

import argparse
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from textwrap import wrap
from zipfile import ZipFile

import colorlover as cl
import numpy as np
import pandas as pd
from pelicun.base import convert_to_MultiIndex, pelicun_path
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# start_time = time.time()


def plot_fragility(comp_db_path, output_path, create_zip='0'):  # noqa: C901, D103
    if create_zip == '1':
        output_path = output_path[:-4]

    if os.path.exists(output_path):  # noqa: PTH110
        shutil.rmtree(output_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    # frag_df = convert_to_MultiIndex(pd.read_csv(resource_dir + '/' + frag_DB_file, index_col=0), axis=1)
    frag_df = convert_to_MultiIndex(pd.read_csv(comp_db_path, index_col=0), axis=1)

    comp_db_meta = comp_db_path[:-3] + 'json'

    if Path(comp_db_meta).is_file():
        with open(comp_db_meta) as f:  # noqa: PTH123
            frag_meta = json.load(f)
    else:
        frag_meta = None

    # for comp_id in frag_df.index[:20]:
    # for comp_id in frag_df.index[400:420]:
    # for comp_id in frag_df.index[438:439]:
    # for comp_id in frag_df.index[695:705]:
    for comp_id in frag_df.index:
        comp_data = frag_df.loc[comp_id]
        if frag_meta != None:  # noqa: E711
            if comp_id in frag_meta.keys():  # noqa: SIM118
                comp_meta = frag_meta[comp_id]
            else:
                comp_meta = None
        else:
            comp_meta = None
        # print(json.dumps(comp_meta, indent=2))

        fig = go.Figure()
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'xy'}, {'type': 'table'}]],
            column_widths=[0.4, 0.6],
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        limit_states = [
            val for val in comp_data.index.unique(level=0) if 'LS' in val
        ]

        # mapping to a color in a sequential color scale
        colors = {
            1: [
                cl.scales['3']['seq']['Reds'][2],
            ],
            2: cl.scales['3']['seq']['Reds'][1:],
            3: cl.scales['4']['seq']['Reds'][1:],
            4: cl.scales['5']['seq']['Reds'][1:],
            5: cl.scales['5']['seq']['Reds'],
        }

        if comp_data.loc[('Incomplete', '')] != 1:  # noqa: RUF031
            p_min, p_max = 0.01, 0.9
            d_min = np.inf
            d_max = -np.inf

            LS_count = 0  # noqa: N806
            for LS in limit_states:  # noqa: N806
                if comp_data.loc[(LS, 'Family')] == 'normal':  # noqa: RUF031
                    d_min_i, d_max_i = norm.ppf(
                        [p_min, p_max],
                        loc=comp_data.loc[(LS, 'Theta_0')],  # noqa: RUF031
                        scale=comp_data.loc[(LS, 'Theta_1')]  # noqa: RUF031
                        * comp_data.loc[(LS, 'Theta_0')],  # noqa: RUF031
                    )
                elif comp_data.loc[(LS, 'Family')] == 'lognormal':  # noqa: RUF031
                    d_min_i, d_max_i = np.exp(
                        norm.ppf(
                            [p_min, p_max],
                            loc=np.log(comp_data.loc[(LS, 'Theta_0')]),  # noqa: RUF031
                            scale=comp_data.loc[(LS, 'Theta_1')],  # noqa: RUF031
                        )
                    )
                else:
                    continue

                LS_count += 1  # noqa: N806

                d_min = np.min([d_min, d_min_i])
                d_max = np.max([d_max, d_max_i])

            demand_vals = np.linspace(d_min, d_max, num=100)

            for i_ls, LS in enumerate(limit_states):  # noqa: N806
                if comp_data.loc[(LS, 'Family')] == 'normal':  # noqa: RUF031
                    cdf_vals = norm.cdf(
                        demand_vals,
                        loc=comp_data.loc[(LS, 'Theta_0')],  # noqa: RUF031
                        scale=comp_data.loc[(LS, 'Theta_1')]  # noqa: RUF031
                        * comp_data.loc[(LS, 'Theta_0')],  # noqa: RUF031
                    )
                elif comp_data.loc[(LS, 'Family')] == 'lognormal':  # noqa: RUF031
                    cdf_vals = norm.cdf(
                        np.log(demand_vals),
                        loc=np.log(comp_data.loc[(LS, 'Theta_0')]),  # noqa: RUF031
                        scale=comp_data.loc[(LS, 'Theta_1')],  # noqa: RUF031
                    )
                else:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=demand_vals,
                        y=cdf_vals,
                        mode='lines',
                        line=dict(width=3, color=colors[LS_count][i_ls]),  # noqa: C408
                        name=LS,
                    ),
                    row=1,
                    col=1,
                )

        else:
            fig.add_trace(
                go.Scatter(
                    x=[
                        0,
                    ],
                    y=[
                        0,
                    ],
                    mode='lines',
                    line=dict(width=3, color=colors[1][0]),  # noqa: C408
                    name='Incomplete Fragility Data',
                ),
                row=1,
                col=1,
            )

        table_vals = []

        for LS in limit_states:  # noqa: N806
            if (
                np.all(  # noqa: E712
                    pd.isna(
                        comp_data[LS][
                            ['Theta_0', 'Family', 'Theta_1', 'DamageStateWeights']
                        ].values
                    )
                )
                == False
            ):
                table_vals.append(  # noqa: PERF401
                    np.insert(
                        comp_data[LS][
                            ['Theta_0', 'Family', 'Theta_1', 'DamageStateWeights']
                        ].values,
                        0,
                        LS,
                    )
                )

        table_vals = np.array(table_vals).T

        ds_list = []
        ds_i = 1
        for dsw in table_vals[-1]:
            if pd.isna(dsw) == True:  # noqa: E712
                ds_list.append(f'DS{ds_i}')
                ds_i += 1

            else:
                w_list = dsw.split('|')
                ds_list.append(
                    '<br>'.join(
                        [
                            f'DS{ds_i + i} ({100.0 * float(w):.0f}%)'
                            for i, w in enumerate(w_list)
                        ]
                    )
                )
                ds_i += len(w_list)

        for i in range(1, 5):
            table_vals[-i] = table_vals[-i - 1]
        table_vals[1] = np.array(ds_list)

        font_size = 16
        if ds_i > 8:  # noqa: PLR2004
            font_size = 8.5

        fig.add_trace(
            go.Table(
                columnwidth=[50, 70, 65, 95, 80],
                header=dict(  # noqa: C408
                    values=[
                        '<b>Limit<br>State</b>',
                        '<b>Damage State(s)</b>',
                        '<b> Median<br>Capacity</b>',
                        '<b>  Capacity<br>Distribution</b>',
                        '<b>  Capacity<br>Dispersion</b>',
                    ],
                    align=['center', 'left', 'center', 'center', 'center'],
                    fill=dict(color='rgb(200,200,200)'),  # noqa: C408
                    line=dict(color='black'),  # noqa: C408
                    font=dict(color='black', size=16),  # noqa: C408
                ),
                cells=dict(  # noqa: C408
                    values=table_vals,
                    height=30,
                    align=['center', 'left', 'center', 'center', 'center'],
                    fill=dict(color='rgba(0,0,0,0)'),  # noqa: C408
                    line=dict(color='black'),  # noqa: C408
                    font=dict(color='black', size=font_size),  # noqa: C408
                ),
            ),
            row=1,
            col=2,
        )

        x_loc = 0.4928
        y_loc = 0.697 + 0.123
        ds_offset = 0.086
        info_font_size = 10

        if ds_i > 8:  # noqa: PLR2004
            x_loc = 0.4928
            y_loc = 0.705 + 0.123
            ds_offset = 0.0455
            info_font_size = 9

        for i_ls, ds_desc in enumerate(ds_list):
            if comp_meta != None:  # noqa: E711
                ls_meta = comp_meta['LimitStates'][f'LS{i_ls + 1}']

                y_loc = y_loc - 0.123

                if '<br>' in ds_desc:
                    ds_vals = ds_desc.split('<br>')

                    for i_ds, ds_name in enumerate(ds_vals):  # noqa: B007
                        ds_id = list(ls_meta.keys())[i_ds]

                        if ls_meta[ds_id].get('Description', False) != False:  # noqa: E712
                            ds_description = '<br>'.join(
                                wrap(ls_meta[ds_id]['Description'], width=70)
                            )
                        else:
                            ds_description = ''

                        if ls_meta[ds_id].get('RepairAction', False) != False:  # noqa: E712
                            ds_repair = '<br>'.join(
                                wrap(ls_meta[ds_id]['RepairAction'], width=70)
                            )
                        else:
                            ds_repair = ''

                        if ds_repair != '':
                            ds_text = f'<b>{ds_id}</b><br>{ds_description}<br><br><b>Repair Action</b><br>{ds_repair}'
                        else:
                            ds_text = f'<b>{ds_id}</b><br>{ds_description}'

                        y_loc_ds = y_loc - 0.018 - i_ds * ds_offset

                        fig.add_annotation(
                            text='<b>*</b>',
                            hovertext=ds_text,
                            xref='paper',
                            yref='paper',
                            axref='pixel',
                            ayref='pixel',
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(size=info_font_size),  # noqa: C408
                            showarrow=False,
                            ax=0,
                            ay=0,
                            x=x_loc,
                            y=y_loc_ds,
                        )

                    y_loc = y_loc_ds - 0.008

                else:
                    # assuming a single Damage State
                    ds_id = list(ls_meta.keys())[0]  # noqa: RUF015

                    if ls_meta[ds_id].get('Description', False) != False:  # noqa: E712
                        ds_description = '<br>'.join(
                            wrap(ls_meta[ds_id]['Description'], width=70)
                        )
                    else:
                        ds_description = ''

                    if ls_meta[ds_id].get('RepairAction', False) != False:  # noqa: E712
                        ds_repair = '<br>'.join(
                            wrap(ls_meta[ds_id]['RepairAction'], width=70)
                        )
                    else:
                        ds_repair = ''

                    if ds_repair != '':
                        ds_text = f'<b>{ds_id}</b><br>{ds_description}<br><br><b>Repair Action</b><br>{ds_repair}'
                    else:
                        ds_text = f'<b>{ds_id}</b><br>{ds_description}'

                    fig.add_annotation(
                        text='<b>*</b>',
                        hovertext=ds_text,
                        xref='paper',
                        yref='paper',
                        axref='pixel',
                        ayref='pixel',
                        xanchor='left',
                        yanchor='bottom',
                        font=dict(size=info_font_size),  # noqa: C408
                        showarrow=False,
                        ax=0,
                        ay=0,
                        x=x_loc,
                        y=y_loc,
                    )

        shared_ax_props = dict(  # noqa: C408
            showgrid=True,
            linecolor='black',
            gridwidth=0.05,
            gridcolor='rgb(192,192,192)',
        )

        demand_unit = comp_data.loc[('Demand', 'Unit')]  # noqa: RUF031
        if demand_unit == 'unitless':
            demand_unit = '-'
        fig.update_xaxes(
            title_text=f"{comp_data.loc[('Demand', 'Type')]} [{demand_unit}]",  # noqa: RUF031
            **shared_ax_props,
        )

        fig.update_yaxes(
            title_text='P(LS≥ls<sub>i</sub>)', range=[0, 1.01], **shared_ax_props
        )

        fig.update_layout(
            # title = f'{comp_id}',
            margin=dict(b=5, r=5, l=5, t=5),  # noqa: C408
            height=300,
            width=950,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )

        with open(f'{output_path}/{comp_id}.html', 'w') as f:  # noqa: PTH123
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # store the source database file(s) in the output directory for future reference
    shutil.copy(comp_db_path, Path(output_path) / Path(comp_db_path).name)

    if frag_meta is not None:
        shutil.copy(comp_db_meta, Path(output_path) / Path(comp_db_meta).name)

    if create_zip == '1':
        files = [f'{output_path}/{file}' for file in os.listdir(output_path)]

        with ZipFile(output_path + '.zip', 'w') as zip:  # noqa: A001
            for file in files:
                zip.write(file, arcname=Path(file).name)

        shutil.rmtree(output_path)

    print('Successfully generated component vulnerability figures.')  # noqa: T201


def plot_repair(comp_db_path, output_path, create_zip='0'):  # noqa: C901, D103, PLR0912, PLR0915
    # TODO:  # noqa: TD002
    # change limit_states names

    if create_zip == '1':
        output_path = output_path[:-4]

    # initialize the output dir

    # if exists, remove it
    if os.path.exists(output_path):  # noqa: PTH110
        shutil.rmtree(output_path)

    # then create it
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # open the input component database
    repair_df = convert_to_MultiIndex(
        convert_to_MultiIndex(pd.read_csv(comp_db_path, index_col=0), axis=1), axis=0
    )

    # The metadata is assumed to be stored at the same location under the same
    # name, in a JSON file
    comp_db_meta = comp_db_path[:-3] + 'json'

    # check if the metadata is there and open it
    if Path(comp_db_meta).is_file():
        with open(comp_db_meta) as f:  # noqa: PTH123
            repair_meta = json.load(f)
    else:
        # otherwise, assign None to facilitate checks later
        repair_meta = None

    # perform the plotting for each component independently
    for comp_id in repair_df.index.unique(level=0):  # [410:418]:
        # perform plotting for each repair consequence type independently
        for c_type in repair_df.loc[comp_id].index:
            # load the component-specific part of the database
            comp_data = repair_df.loc[(comp_id, c_type)]  # noqa: RUF031

            # and the component-specific metadata - if it exists
            if repair_meta != None:  # noqa: E711
                if comp_id in repair_meta.keys():  # noqa: SIM118
                    comp_meta = repair_meta[comp_id]
                else:
                    comp_meta = None
            else:
                comp_meta = None

            # start plotting

            # create a figure
            fig = go.Figure()

            # create two subplots, one for the curve and one for the tabular data
            fig = make_subplots(
                rows=1,
                cols=3,
                specs=[
                    [{'type': 'xy'}, {'type': 'xy'}, {'type': 'table'}],
                ],
                shared_yaxes=True,
                column_widths=[0.45, 0.05, 0.52],
                horizontal_spacing=0.02,
                vertical_spacing=0.02,
            )

            # initialize the table collecting parameters
            table_vals = []

            # get all potential limit state labels
            limit_states = [
                val for val in comp_data.index.unique(level=0) if 'DS' in val
            ]

            # check for each limit state
            for LS in limit_states:  # noqa: N806
                fields = ['Theta_0', 'Family', 'Theta_1']

                comp_data_LS = comp_data[LS]  # noqa: N806

                for optional_label in ['Family', 'Theta_1']:
                    if optional_label not in comp_data_LS.index:
                        comp_data_LS[optional_label] = None

                # if any of the fields above is set
                if np.all(pd.isna(comp_data_LS[fields].values)) == False:  # noqa: E712
                    # Then we assume that is valuable information that needs to be
                    # shown in the table while the other fields will show 'null'
                    table_vals.append(np.insert(comp_data_LS[fields].values, 0, LS))

            # transpose the table to work well with plotly's API
            table_vals = np.array(table_vals).T

            # copy the collected parameters into another object
            model_params = deepcopy(table_vals)

            # replace parameters for multilinear functions with 'varies'
            for ds_i, val in enumerate(table_vals[1]):
                if '|' in str(val):
                    table_vals[1][ds_i] = 'varies'
                elif pd.isna(val) == True:  # noqa: E712
                    table_vals[1][ds_i] = 'N/A'
                else:
                    conseq_val = float(val)
                    if conseq_val < 1:
                        table_vals[1][ds_i] = f'{conseq_val:.4g}'
                    elif conseq_val < 10:  # noqa: PLR2004
                        table_vals[1][ds_i] = f'{conseq_val:.3g}'
                    elif conseq_val < 1e6:  # noqa: PLR2004
                        table_vals[1][ds_i] = f'{conseq_val:.0f}'
                    else:
                        table_vals[1][ds_i] = f'{conseq_val:.3g}'

            # round dispersion parameters to 2 digits
            table_vals[-1] = [
                f'{float(sig):.2f}' if pd.isna(sig) == False else 'N/A'  # noqa: E712
                for sig in table_vals[-1]
            ]

            # replace missing distribution labels with N/A
            table_vals[-2] = [
                family if pd.isna(family) == False else 'N/A'  # noqa: E712
                for family in table_vals[-2]
            ]

            # converted simultaneous damage models might have a lot of DSs
            if table_vals.shape[1] > 8:  # noqa: PLR2004
                lots_of_ds = True
            else:
                lots_of_ds = False

            # set the font size
            font_size = 16 if lots_of_ds == False else 11  # noqa: E712

            # create the table

            # properties shared between consequence types
            c_pad = (9 - len(c_type)) * ' '  # noqa: F841
            table_header = [
                '<b>Damage<br>  State</b>',
                '<b>Median<br>Conseq.</b>',
                '<b>   Conseq.<br>Distribution</b>',
                '<b>  Conseq.<br>Dispersion</b>',
            ]
            cell_alignment = ['center', 'center', 'center', 'center']
            column_widths = [45, 45, 60, 55]

            fig.add_trace(
                go.Table(
                    columnwidth=column_widths,
                    header=dict(  # noqa: C408
                        values=table_header,
                        align=cell_alignment,
                        fill=dict(color='rgb(200,200,200)'),  # noqa: C408
                        line=dict(color='black'),  # noqa: C408
                        font=dict(color='black', size=16),  # noqa: C408
                    ),
                    cells=dict(  # noqa: C408
                        values=table_vals,
                        height=30 if lots_of_ds == False else 19,  # noqa: E712
                        align=cell_alignment,
                        fill=dict(color='rgba(0,0,0,0)'),  # noqa: C408
                        line=dict(color='black'),  # noqa: C408
                        font=dict(color='black', size=font_size),  # noqa: C408
                    ),
                ),
                row=1,
                col=3,
            )

            # get the number (and label) of damage states
            limit_states = model_params[0]

            # mapping to a color in a sequential color scale
            colors = {
                1: [
                    cl.scales['3']['seq']['PuBu'][2],
                ],
                2: cl.scales['3']['seq']['PuBu'][1:],
                3: cl.scales['4']['seq']['PuBu'][1:],
                4: cl.scales['6']['seq']['PuBu'][2:],
                5: cl.scales['7']['seq']['PuBu'][2:],
                6: cl.scales['7']['seq']['PuBu'][1:],
                7: cl.scales['7']['seq']['PuBu'],
                # Simultaneous elevators have a lot of DSs and need special
                # treatment
                15: (
                    cl.scales['9']['seq']['PuBu']
                    + cl.scales['8']['seq']['YlGnBu'][::-1][1:-1]
                ),
            }

            if comp_data.loc[('Incomplete', '')] != 1:  # noqa: RUF031
                # set the parameters for displaying uncertainty
                p_min, p_max = 0.16, 0.84  # +- 1 std  # noqa: F841

                # initialize quantity limits
                q_min = 0
                q_max = -np.inf

                # walk through median parameters
                for mu_capacity in model_params[1]:
                    # if any of them is quantity dependent
                    if '|' in str(mu_capacity):
                        # then parse the quantity limits
                        q_lims = np.array(
                            mu_capacity.split('|')[1].split(','), dtype=float
                        )

                        # Add the lower and upper limits to get a q_max that
                        # will lead to a nice plot
                        q_max = np.max([np.sum(q_lims), q_max])

                # if none of the medians is quantity-dependent,
                if q_max == -np.inf:
                    # Set q_max to 1.0 to scale the plot appropriately
                    q_max = 1.0

                # anchor locations for annotations providing DS information
                x_loc = 0.533 if lots_of_ds == False else 0.535  # noqa: E712
                y_space = 0.088 if lots_of_ds == False else 0.0543  # noqa: E712
                y_loc = 0.784 + y_space if lots_of_ds == False else 0.786 + y_space  # noqa: E712
                info_font_size = 10 if lots_of_ds == False else 9  # noqa: E712

                # x anchor for annotations providing median function data
                x_loc_func = 0.697 if lots_of_ds == False else 0.689  # noqa: E712

                need_x_axis = False

                for ds_i, mu_capacity in enumerate(model_params[1]):
                    # first, check if the median is a function:
                    if '|' in str(mu_capacity):
                        need_x_axis = True

                        # get the consequence (Y) and quantity (X) values
                        c_vals, q_vals = np.array(
                            [vals.split(',') for vals in mu_capacity.split('|')],
                            dtype=float,
                        )

                    else:
                        c_vals = np.array(
                            [
                                mu_capacity,
                            ],
                            dtype=float,
                        )
                        q_vals = np.array(
                            [
                                0.0,
                            ],
                            dtype=float,
                        )

                    # add one more value to each end to represent the
                    # constant parts
                    q_vals = np.insert(q_vals, 0, q_min)
                    c_vals = np.insert(c_vals, 0, c_vals[0])

                    q_vals = np.append(q_vals, q_max)
                    c_vals = np.append(c_vals, c_vals[-1])

                    # plot the median consequence
                    fig.add_trace(
                        go.Scatter(
                            x=q_vals,
                            y=c_vals,
                            mode='lines',
                            line=dict(  # noqa: C408
                                width=3,
                                color=colors[np.min([len(model_params[1]), 7])][
                                    ds_i % 7
                                ],
                            ),
                            name=model_params[0][ds_i],
                            legendgroup=model_params[0][ds_i],
                        ),
                        row=1,
                        col=1,
                    )

                    # check if dispersion is prescribed for this consequence
                    dispersion = model_params[3][ds_i]
                    if (pd.isna(dispersion) == False) and (dispersion != 'N/A'):  # noqa: E712
                        dispersion = float(dispersion)

                        if model_params[2][ds_i] == 'normal':
                            std_plus = c_vals * (1 + dispersion)
                            std_minus = c_vals * (1 - dispersion)

                            std_plus_label = 'mu + std'
                            std_minus_label = 'mu - std'
                        elif model_params[2][ds_i] == 'lognormal':
                            std_plus = np.exp(np.log(c_vals) + dispersion)
                            std_minus = np.exp(np.log(c_vals) - dispersion)

                            std_plus_label = 'mu + lnstd'
                            std_minus_label = 'mu - lnstd'
                        else:
                            continue

                        # plot the std lines
                        fig.add_trace(
                            go.Scatter(
                                x=q_vals,
                                y=std_plus,
                                mode='lines',
                                line=dict(  # noqa: C408
                                    width=1,
                                    color=colors[np.min([len(model_params[1]), 7])][
                                        ds_i % 7
                                    ],
                                    dash='dash',
                                ),
                                name=model_params[0][ds_i] + ' ' + std_plus_label,
                                legendgroup=model_params[0][ds_i],
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                        fig.add_trace(
                            go.Scatter(
                                x=q_vals,
                                y=std_minus,
                                mode='lines',
                                line=dict(  # noqa: C408
                                    width=1,
                                    color=colors[np.min([len(model_params[1]), 7])][
                                        ds_i % 7
                                    ],
                                    dash='dash',
                                ),
                                name=model_params[0][ds_i] + ' ' + std_minus_label,
                                legendgroup=model_params[0][ds_i],
                                showlegend=False,
                            ),
                            row=1,
                            col=1,
                        )

                        # and plot distribution pdfs on top
                        if model_params[2][ds_i] == 'normal':
                            sig = c_vals[-1] * dispersion
                            q_pdf = np.linspace(
                                np.max(
                                    [norm.ppf(0.025, loc=c_vals[-1], scale=sig), 0]
                                ),
                                norm.ppf(0.975, loc=c_vals[-1], scale=sig),
                                num=100,
                            )
                            c_pdf = norm.pdf(q_pdf, loc=c_vals[-1], scale=sig)

                        elif model_params[2][ds_i] == 'lognormal':
                            q_pdf = np.linspace(
                                np.exp(
                                    norm.ppf(
                                        0.025,
                                        loc=np.log(c_vals[-1]),
                                        scale=dispersion,
                                    )
                                ),
                                np.exp(
                                    norm.ppf(
                                        0.975,
                                        loc=np.log(c_vals[-1]),
                                        scale=dispersion,
                                    )
                                ),
                                num=100,
                            )
                            c_pdf = norm.pdf(
                                np.log(q_pdf),
                                loc=np.log(c_vals[-1]),
                                scale=dispersion,
                            )

                        c_pdf /= np.max(c_pdf)

                        fig.add_trace(
                            go.Scatter(
                                x=c_pdf,
                                y=q_pdf,
                                mode='lines',
                                line=dict(  # noqa: C408
                                    width=1,
                                    color=colors[np.min([len(model_params[1]), 7])][
                                        ds_i % 7
                                    ],
                                ),
                                fill='tozeroy',
                                name=model_params[0][ds_i] + ' pdf',
                                legendgroup=model_params[0][ds_i],
                                showlegend=False,
                            ),
                            row=1,
                            col=2,
                        )

                    # adjust y_loc for annotations
                    y_loc = y_loc - y_space

                    # add annotations for median function parameters, if needed
                    if '|' in str(mu_capacity):
                        c_vals, q_vals = (
                            vals.split(',') for vals in mu_capacity.split('|')
                        )

                        func_text = f'<b>Multilinear Function Breakpoints</b><br>Medians: {", ".join(c_vals)}<br>Quantities: {", ".join(q_vals)}'

                        fig.add_annotation(
                            text='<b>*</b>',
                            hovertext=func_text,
                            xref='paper',
                            yref='paper',
                            axref='pixel',
                            ayref='pixel',
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(size=info_font_size),  # noqa: C408
                            showarrow=False,
                            ax=0,
                            ay=0,
                            x=x_loc_func,
                            y=y_loc,
                        )

                    # check if metadata is available
                    if comp_meta != None:  # noqa: E711
                        ds_meta = comp_meta['DamageStates'][f'DS{ds_i + 1}']

                        if ds_meta.get('Description', False) != False:  # noqa: E712
                            ds_description = '<br>'.join(
                                wrap(ds_meta['Description'], width=55)
                            )
                        else:
                            ds_description = ''

                        if ds_meta.get('RepairAction', False) != False:  # noqa: E712
                            ds_repair = '<br>'.join(
                                wrap(ds_meta['RepairAction'], width=55)
                            )
                        else:
                            ds_repair = ''

                        if ds_repair != '':
                            ds_text = f'<b>{model_params[0][ds_i]}</b><br>{ds_description}<br><br><b>Repair Action</b><br>{ds_repair}'
                        else:
                            ds_text = (
                                f'<b>{model_params[0][ds_i]}</b><br>{ds_description}'
                            )

                        fig.add_annotation(
                            text='<b>*</b>',
                            hovertext=ds_text,
                            xref='paper',
                            yref='paper',
                            axref='pixel',
                            ayref='pixel',
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(size=info_font_size),  # noqa: C408
                            showarrow=False,
                            ax=0,
                            ay=0,
                            x=x_loc,
                            y=y_loc,
                        )

            else:
                # add an empty figure still; to highlight incomplete data
                fig.add_trace(
                    go.Scatter(
                        x=[
                            0,
                        ],
                        y=[
                            0,
                        ],
                        mode='lines',
                        line=dict(width=3, color=colors[1][0]),  # noqa: C408
                        name=f'Incomplete Repair {c_type} Consequence Data',
                    ),
                    row=1,
                    col=2,
                )

            shared_ax_props = dict(  # noqa: C408
                showgrid=True,
                linecolor='black',
                gridwidth=0.05,
                gridcolor='rgb(220,220,220)',
            )

            quantity_unit = comp_data.loc[('Quantity', 'Unit')]
            if quantity_unit in ['unitless', '1 EA', '1 ea']:
                quantity_unit = '-'
            elif quantity_unit.split()[0] == '1':
                quantity_unit = quantity_unit.split()[1]

            dv_unit = comp_data.loc[('DV', 'Unit')]  # noqa: RUF031
            if dv_unit == 'unitless':
                dv_unit = '-'

            # layout settings
            fig.update_layout(
                # minimize margins
                margin=dict(b=50, r=5, l=5, t=5),  # noqa: C408
                # height and width targets single-column web view
                height=400,
                width=950,
                # transparent background and paper
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                # legend on to allow turning DSs off
                showlegend=True,
                xaxis1=dict(
                    title_text=f'Damage Quantity [{quantity_unit}]',
                    range=[q_min, q_max],
                    **shared_ax_props,
                )
                if need_x_axis == True  # noqa: E712
                else dict(showgrid=False, showticklabels=False),  # noqa: C408
                yaxis1=dict(
                    title_text=f'{c_type} [{dv_unit}]',
                    rangemode='tozero',
                    **shared_ax_props,
                ),
                xaxis2=dict(  # noqa: C408
                    showgrid=False,
                    showticklabels=False,
                    title_text='',
                ),
                yaxis2=dict(showgrid=False, showticklabels=False),  # noqa: C408
                # position legend to top of the figure
                legend=dict(  # noqa: C408
                    yanchor='top',
                    xanchor='right',
                    font=dict(size=12),  # noqa: C408
                    orientation='v',
                    y=1.0,
                    x=-0.08,
                ),
            )

            # save figure to html
            with open(f'{output_path}/{comp_id}-{c_type}.html', 'w') as f:  # noqa: PTH123
                # Minimize size by not saving javascript libraries which means
                # internet connection is required to view the figure.
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # store the source database file(s) in the output directory for future reference
    shutil.copy(comp_db_path, Path(output_path) / Path(comp_db_path).name)

    if repair_meta is not None:
        shutil.copy(comp_db_meta, Path(output_path) / Path(comp_db_meta).name)

    if create_zip == '1':
        files = [f'{output_path}/{file}' for file in os.listdir(output_path)]

        with ZipFile(output_path + '.zip', 'w') as zip:  # noqa: A001
            for file in files:
                zip.write(file, arcname=Path(file).name)

        shutil.rmtree(output_path)

    print('Successfully generated component repair consequence figures.')  # noqa: T201


def check_diff(comp_db_path, output_path):  # noqa: D103
    # if the output path already exists
    if os.path.exists(output_path):  # noqa: PTH110
        # check for both the csv and json files
        for ext in ['csv', 'json']:
            comp_db = comp_db_path[:-3] + ext

            if not Path(comp_db).is_file():
                continue

            # get the name of the db file
            source_path = Path(comp_db).parents[0]
            comp_db = Path(comp_db).name

            # check if a file with the same name exists in the output dir
            if comp_db in os.listdir(output_path):
                # open the two files and compare them
                with open(Path(source_path) / comp_db) as f1, open(  # noqa: PTH123
                    Path(output_path) / comp_db
                ) as f2:
                    if ext == 'csv':
                        new_file = f1.readlines()
                        old_file = f2.readlines()

                        # compare every line in the new file with the old file
                        # if every line exists in the old file, continue with the next file
                        if new_file == old_file:
                            continue

                        # if at least one line does not match, we need to generate
                        else:  # noqa: RET507
                            return True

                    elif ext == 'json':
                        new_file = json.load(f1)
                        old_file = json.load(f2)

                        # if the two dicts are identical, continue with the next file
                        if sorted(new_file.items()) == sorted(old_file.items()):
                            continue

                        # otherwise, we need to generate
                        else:  # noqa: RET507
                            return True

            # if there is no db file in the output dir, we need to generate
            else:
                return True

        # if both files were identical, there is no need to generate
        return False

    # if the output path does not exist, we need to generate
    else:  # noqa: RET505
        return True


def main(args):  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument('viz_type')
    parser.add_argument('comp_db_path')
    parser.add_argument(
        '-o', '--output_path', default='./comp_viz/'
    )  # replace with None
    parser.add_argument('-z', '--zip', default='0')

    args = parser.parse_args(args)

    if args.viz_type in ['fragility', 'repair']:
        comp_db_path = args.comp_db_path
        output_path = args.output_path

        # verify that comp_db_path points to a file
        if not Path(comp_db_path).is_file():
            raise FileNotFoundError('comp_db_path does not point to a file.')  # noqa: EM101, TRY003

        if check_diff(comp_db_path, output_path):
            if args.viz_type == 'fragility':
                plot_fragility(comp_db_path, output_path, args.zip)

            elif args.viz_type == 'repair':
                plot_repair(comp_db_path, output_path, args.zip)

        else:
            print('No need to generate, figures already exist in the output folder.')  # noqa: T201

    elif args.viz_type == 'query':
        if args.comp_db_path == 'default_db':
            print(pelicun_path)  # noqa: T201

    # print("--- %s seconds ---" % (time.time() - start_time))


# python3 DL_visuals.py repair /Users/adamzs/SimCenter/applications/performDL/pelicun3/pelicun/resources/SimCenterDBDL/loss_repair_DB_FEMA_P58_2nd.csv

if __name__ == '__main__':
    main(sys.argv[1:])
