import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import sys

from ENPMDA import MDDataFrame

def export_plotly(plotly_df,
                  title,
                  output,
                  sel_tic = [0, 1],
                  append=False):
    print('output: ', output)
    if len(sel_tic) != 2:
        raise ValueError('sel_tic must be a list of two integers.')
    print('sel_tic: ', sel_tic)

    struc_state_dic = {
            'BGT': 'CLOSED',
            'EPJ': 'DESENSITIZED',
            '7ekt': 'I (7EKT)',
            'EPJPNU': 'OPEN',
    }

    tic_values = []
    for tic in sel_tic:
        tic_values.append(plotly_df[plotly_df.traj_time%10000 == 0][f'tic_{tic}'].values)

    fig = go.Figure()
    fig.update_layout(margin=dict(t=150))

    default_size = 10
    highlighted_size_delta = 10

    plot_states = ['BGT', 'EPJ', 'EPJPNU']
    for system, df in plotly_df.groupby('system'):
        pathway = df.pathway.unique()[0]
        pathway_text = ' to '.join([struc_state_dic[path][0] for path in pathway.split('_')])
        seed = df.seed.unique()[0]
        x = df[df.traj_time%10000 == 0][f'tic_{sel_tic[0]}'].values
        y = df[df.traj_time%10000 == 0][f'tic_{sel_tic[1]}'].values
        t = df[df.traj_time%10000 == 0]['frame'].values

        fig.add_trace(
            go.Scattergl(x=x, y=y,
                name=f'SEED_{seed}',
                mode='lines+markers',
                visible='legendonly',
                legendgroup=pathway_text,
                legendgrouptitle_text=pathway_text,
                showlegend=True,
                line=dict(
                    width=2,
                    color='black',
                ),
                marker=dict(
                    color=t,
                    colorscale='Purp',
                    size=10,
                    opacity=1,
                    showscale=False)
            )
        )
        if seed == 0 and pathway.split('_')[0] in plot_states:
            fig.add_annotation(x=x[10], y=y[10],
            text=struc_state_dic[pathway.split('_')[0]],
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="center",
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.4)
            plot_states.remove(pathway.split('_')[0])


    def update_trace(trace, points, selector):
        # this list stores the points which were clicked on
        # in all but one trace they are empty
        print(points)
        if len(points.point_inds) == 0:
            return
        
        for i,_ in enumerate(fig.data):
            fig.data[i]['marker']['size'] = default_size + highlighted_size_delta


    # we need to add the on_click event to each trace separately       
    for i in range(len(fig.data[1:])):
        fig.data[i].on_click(update_trace)

    fig.update_xaxes(title_text="IC 1")
    fig.update_yaxes(title_text="IC 2")
    fig.update_layout(
        autosize=True,
        width=1300,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
        title=f"{title} \n Color (Time)\n Size (MSM weight)",
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="#000000"
            ),
        xaxis_range=[-1.5, 1.3],
        yaxis_range=[-2.1, 2.9],
        legend=dict(x=1.1, y=0.95),
        legend_groupclick="toggleitem",
    #    legend_groupclick="togglegroup",
        legend_orientation="h",
        updatemenus=[
            dict(
                active=0,
            buttons=list([
                dict(label="Only FES",
                     method="restyle",
                     visible=True,
                     args=[{"visible": [True] + ['legendonly'] * (len(fig.data) - 1)}],
                        ),
            ]
        )
    )
    ]
    )

    if append:
        with open(output, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    else:
        with open(output, 'w') as f:
            f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))

    print(f'Exported {output}')