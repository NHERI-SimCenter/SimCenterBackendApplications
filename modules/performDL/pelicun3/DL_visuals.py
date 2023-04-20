# -*- coding: utf-8 -*-
#
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

import os, sys, json
import argparse
import shutil
from pathlib import Path
from textwrap import wrap

import numpy as np
from scipy.stats import norm
import pandas as pd

import colorlover as cl

from pelicun.base import convert_to_MultiIndex

from plotly import graph_objects as go
from plotly.subplots import make_subplots

import time

#start_time = time.time()

def plot_fragility(comp_db_path, output_path):

    #resource_dir = '/Users/adamzs/Repos/pelicun/pelicun/resources'

    #frag_DB_file = 'fragility_DB_FEMA_P58_2nd.csv'
    #frag_meta_file = 'fragility_DB_FEMA_P58_2nd.json'

    #frag_DB_file = 'fragility_DB_Hazus_EQ.csv'
    #frag_meta_file = 'fragility_DB_Hazus_EQ.json'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    Path(output_path).mkdir(parents=True, exist_ok=True);

    #frag_df = convert_to_MultiIndex(pd.read_csv(resource_dir + '/' + frag_DB_file, index_col=0), axis=1)
    frag_df = convert_to_MultiIndex(pd.read_csv(comp_db_path, index_col=0), axis=1)

    comp_db_meta = comp_db_path[:-3]+'json'

    if Path(comp_db_meta).is_file():
        with open(comp_db_meta, 'r') as f:
            frag_meta = json.load(f)
    else:
        frag_meta = None

    #for comp_id in frag_df.index[:20]:
    #for comp_id in frag_df.index[400:420]:
    #for comp_id in frag_df.index[438:439]:
    #for comp_id in frag_df.index[695:705]:
    for comp_id in frag_df.index:

        comp_data = frag_df.loc[comp_id]
        if frag_meta != None:
            if comp_id in frag_meta.keys():
                comp_meta = frag_meta[comp_id]
            else:
                comp_meta = None
        else:
            comp_meta = None
        #print(json.dumps(comp_meta, indent=2))

        fig = go.Figure()
        fig = make_subplots(
            rows=1, cols=2,
            specs = [[{"type":"xy"},{"type":"table"}]],
            column_widths = [0.4, 0.6],
            horizontal_spacing = 0.02,
            vertical_spacing=0.02
            )

        limit_states = [val for val in comp_data.index.unique(level=0) if 'LS' in val]

        # mapping to a color in a sequential color scale
        colors = {
            1: [cl.scales['3']['seq']['Reds'][2],],
            2: cl.scales['3']['seq']['Reds'][1:],
            3: cl.scales['4']['seq']['Reds'][1:],
            4: cl.scales['5']['seq']['Reds'][1:],
            5: cl.scales['5']['seq']['Reds']
        }

        if comp_data.loc[('Incomplete','')] != 1:

            p_min, p_max = 0.01, 0.9
            d_min = np.inf
            d_max = -np.inf            

            LS_count = 0
            for LS in limit_states:
                if comp_data.loc[(LS,'Family')] == 'normal':
                    d_min_i, d_max_i = norm.ppf([p_min, p_max], 
                                            loc=comp_data.loc[(LS,'Theta_0')], 
                                            scale=comp_data.loc[(LS,'Theta_1')]*comp_data.loc[(LS,'Theta_0')])
                elif comp_data.loc[(LS,'Family')] == 'lognormal':
                    d_min_i, d_max_i = np.exp(norm.ppf([p_min, p_max], 
                                                       loc=np.log(comp_data.loc[(LS,'Theta_0')]), 
                                                       scale=comp_data.loc[(LS,'Theta_1')]))
                else:
                    continue

                LS_count += 1
                    
                d_min = np.min([d_min, d_min_i])
                d_max = np.max([d_max, d_max_i])
                 
            demand_vals = np.linspace(d_min, d_max, num=100)

            for i_ls, LS in enumerate(limit_states):

                if comp_data.loc[(LS,'Family')] == 'normal':
                    cdf_vals = norm.cdf(demand_vals, 
                                        loc=comp_data.loc[(LS,'Theta_0')], 
                                        scale=comp_data.loc[(LS,'Theta_1')]*comp_data.loc[(LS,'Theta_0')])
                elif comp_data.loc[(LS,'Family')] == 'lognormal':
                    cdf_vals = norm.cdf(np.log(demand_vals), 
                                        loc=np.log(comp_data.loc[(LS,'Theta_0')]), 
                                        scale=comp_data.loc[(LS,'Theta_1')])
                else:
                    continue            

                fig.add_trace(go.Scatter(
                    x = demand_vals,
                    y = cdf_vals,
                    mode = 'lines',
                    line = dict(
                        width=3,
                        color=colors[LS_count][i_ls]
                    ),
                    name = LS,
                ), row=1, col=1)

        else:
            fig.add_trace(go.Scatter(
                x = [0,],
                y = [0,],
                mode = 'lines',
                line = dict(
                    width=3,
                    color=colors[1][0]
                ),
                name = 'Incomplete Fragility Data',
            ), row=1, col=1)

        table_vals = []

        for LS in limit_states:

            if np.all(pd.isna(comp_data[LS][['Theta_0','Family','Theta_1','DamageStateWeights']].values)) == False:
                table_vals.append(np.insert(comp_data[LS][['Theta_0','Family','Theta_1','DamageStateWeights']].values, 0, LS))

        table_vals = np.array(table_vals).T

        ds_list = []
        ds_i = 1
        for dsw in table_vals[-1]:

            if pd.isna(dsw) == True:
                ds_list.append(f'DS{ds_i}')
                ds_i += 1

            else:
                w_list = dsw.split('|')
                ds_list.append('<br>'.join([f'DS{ds_i+i} ({100.0 * float(w):.0f}%)' 
                    for i, w in enumerate(w_list)]))
                ds_i += len(w_list)

        for i in range(1,5):
            table_vals[-i] = table_vals[-i-1]
        table_vals[1] = np.array(ds_list)

        font_size = 16        
        if ds_i > 8:
            font_size = 8.5
        
        fig.add_trace(go.Table(
            columnwidth = [50,70,65,95,80],
            header=dict(
                values=['<b>Limit<br>State</b>',
                        '<b>Damage State(s)</b>',
                        '<b> Median<br>Capacity</b>',
                        '<b>  Capacity<br>Distribution</b>',
                        '<b>  Capacity<br>Dispersion</b>'],
                align=['center','left','center','center','center'],
                fill = dict(color='rgb(200,200,200)'),
                line = dict(color='black'),
                font = dict(color='black', size=16)
                ),
            cells=dict(
                values=table_vals,  
                height = 30,              
                align=['center','left','center','center','center'],
                fill = dict(color='rgba(0,0,0,0)'),
                line = dict(color='black'),
                font = dict(color='black', size=font_size)
                )
            ), row=1, col=2)

        x_loc = 0.4928
        y_loc = 0.697 + 0.123
        ds_offset = 0.086
        info_font_size = 10

        if ds_i > 8:
            x_loc = 0.4928
            y_loc = 0.705 + 0.123
            ds_offset = 0.0455
            info_font_size = 9

        for i_ls, ds_desc in enumerate(ds_list):

            if comp_meta != None:
                ls_meta = comp_meta['LimitStates'][f'LS{i_ls+1}']

                y_loc = y_loc - 0.123

                if '<br>' in ds_desc:

                    ds_vals = ds_desc.split('<br>')

                    for i_ds, ds_name in enumerate(ds_vals):                    

                        ds_id = list(ls_meta.keys())[i_ds]

                        if ls_meta[ds_id].get('Description', False) != False:
                            ds_description = '<br>'.join(wrap(ls_meta[ds_id]["Description"], width=70))
                        else:
                            ds_description = ''
                        
                        if ls_meta[ds_id].get('RepairAction', False) != False:
                            ds_repair = '<br>'.join(wrap(ls_meta[ds_id]["RepairAction"], width=70))
                        else:
                            ds_repair = ''

                        ds_text = f'<b>{ds_id}</b><br>{ds_description}<br><br><b>Repair Action</b><br>{ds_repair}'

                        y_loc_ds = y_loc - 0.018 - i_ds*ds_offset

                        fig.add_annotation(
                        text=f'<b>*</b>',
                        hovertext=ds_text,
                        xref='paper', yref='paper',
                        axref='pixel', ayref='pixel',
                        xanchor = 'left', yanchor='bottom',
                        font=dict(size=info_font_size),
                        showarrow = False,
                        ax = 0, ay = 0,
                        x = x_loc, y = y_loc_ds)

                    y_loc = y_loc_ds - 0.008


                else:

                    # assuming a single Damage State
                    ds_id = list(ls_meta.keys())[0]

                    if ls_meta[ds_id].get('Description', False) != False:
                        ds_description = '<br>'.join(wrap(ls_meta[ds_id]["Description"], width=70))
                    else:
                        ds_description = ''
                    
                    if ls_meta[ds_id].get('RepairAction', False) != False:
                        ds_repair = '<br>'.join(wrap(ls_meta[ds_id]["RepairAction"], width=70))
                    else:
                        ds_repair = ''

                    ds_text = f'<b>{ds_id}</b><br>{ds_description}<br><br><b>Repair Action</b><br>{ds_repair}'

                    fig.add_annotation(
                        text=f'<b>*</b>',
                        hovertext=ds_text,
                        xref='paper', yref='paper',
                        axref='pixel', ayref='pixel',
                        xanchor = 'left', yanchor='bottom',
                        font=dict(size=info_font_size),
                        showarrow = False,
                        ax = 0, ay = 0,
                        x = x_loc, y = y_loc)
            
        shared_ax_props = dict(
            showgrid = True,
            linecolor = 'black',
            gridwidth = 0.05,
            gridcolor = 'rgb(192,192,192)'
        )

        demand_unit = comp_data.loc[('Demand','Unit')]
        if demand_unit == 'unitless':
            demand_unit = '-'
        fig.update_xaxes(
            title_text=f"{comp_data.loc[('Demand','Type')]} [{demand_unit}]",
            **shared_ax_props)


        fig.update_yaxes(title_text=f'P(LS≥ls<sub>i</sub>)', 
                         range=[0,1.01],
                         **shared_ax_props)
            
        fig.update_layout(
            #title = f'{comp_id}',
            margin=dict(b=5,r=5,l=5,t=5),
            height=300,
            width=950,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor = 'rgba(0,0,0,0)',
            showlegend=False
        )

        with open(f'{output_path}{comp_id}.html', "w") as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    print("Successfully generated component vulnerability figures.")

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('viz_type')
    parser.add_argument('comp_db_path')
    parser.add_argument('-o', '--output_path', 
        default="./comp_viz/") #replace with None

    args = parser.parse_args(args)

    if args.viz_type == 'fragility':
        plot_fragility(args.comp_db_path, args.output_path)

    #print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':

    main(sys.argv[1:])
