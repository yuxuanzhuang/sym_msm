import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import sys

from ENPMDA import MDDataFrame

def generate_tica_csv(
                      md_dataframe: MDDataFrame,
                      msm_obj,
                      sel_tics=[0, 1],
                      output='tica.csv'
                      ):
    """Generate tica.csv for plotly_fes.py by projecting the
    md_dataframe onto the selected two tica components.
    md_dataframe: MDDataFrame
    msm_obj: MSM object
    sel_tics: list of int
    output: str
    """
    plotly_tica_output = msm_obj.transform_feature_trajectories(md_dataframe)
    plotly_df = md_dataframe.dataframe.copy()

    plotly_tica_concatenated = np.concatenate(plotly_tica_output[::5])
    for tic in sel_tics:
        plotly_df[f'tic_{tic}'] = plotly_tica_concatenated[:, tic]

    plotly_df['msm_weight'] = 0
    # if trajectory_weights exist, use them to weight the tica data.
    if hasattr(msm_obj, 'trajectory_weights'):
        plotly_df.iloc[plotly_df[plotly_df.frame >= msm_obj.start * msm_obj.md_dataframe.stride].index, -1] = np.concatenate(msm_obj.trajectory_weights[::5])
    plotly_df.to_csv(output)


# copy from pyemma plot2d
def _to_free_energy(z, minener_zero=False):
    """Compute free energies from histogram counts.
    Parameters
    ----------
    z : ndarray(T)
        Histogram counts.
    minener_zero : boolean, optional, default=False
        Shifts the energy minimum to zero.
    Returns
    -------
    free_energy : ndarray(T)
        The free energy values in units of kT.
    """
    pi = z / float(z.sum())
    free_energy = np.inf * np.ones(shape=z.shape)
    nonzero = pi.nonzero()
    free_energy[nonzero] = -np.log(pi[nonzero])
    if minener_zero:
        free_energy[nonzero] -= np.min(free_energy[nonzero])
    return free_energy

def export_plotly(tica_csv,
                  title,
                  output,
                  sel_tic = [0, 1],
                  append=False):
    print('tica_csv: ', tica_csv)
    print('output: ', output)
    if len(sel_tic) != 2:
        raise ValueError('sel_tic must be a list of two integers.')
    print('sel_tic: ', sel_tic)
    # load tica data
    try:
        plotly_df = pd.read_csv(tica_csv)
    except:
        print(f'No tica data found. Generate {tica_csv} first.')
        exit()

    struc_state_dic = {
            'BGT': 'CLOSED',
            'EPJ': 'DESENSITIZED',
            '7ekt': 'I (7EKT)',
            'EPJPNU': 'OPEN',
    }

    tic_values = []
    for tic in sel_tic:
        tic_values.append(plotly_df[plotly_df.traj_time%10000 == 0][f'tic_{tic}'].values)

    if not (plotly_df['msm_weight'] == 0).all():
        weights = plotly_df[plotly_df.traj_time%10000 == 0]['msm_weight']
    else:
        weights = None

    z, xedge, yedge = np.histogram2d(
            tic_values[0], tic_values[1], bins=50, weights=weights)

    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])

    z = np.maximum(z, np.min(z[z.nonzero()])).T
    f = _to_free_energy(z, minener_zero=True)

    fig = go.Figure()
    fig.update_layout(margin=dict(t=150))

#    fig = go.FigureWidget()
#    fig.layout.hovermode = 'closest'
#    fig.layout.hoverdistance = -1 #ensures no "gaps" for selecting sparse data
    default_size = 10
    highlighted_size_delta = 10

    if True:
        fig.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=f,
                zmax=10,
                zmin=0,
                zmid=3,
                ncontours=10,
                colorscale = 'Earth',
                showscale=False)
        )

        fig.update_traces(
                        contours_coloring="fill",
                        #contours_coloring = 'lines',
                        contours_showlabels = True)

    plot_states = ['BGT', 'EPJ', 'EPJPNU']
    for system, df in plotly_df.groupby('system'):
        pathway = df.pathway.unique()[0]
        pathway_text = ' to '.join([struc_state_dic[path][0] for path in pathway.split('_')])
        seed = df.seed.unique()[0]
        x = df[df.traj_time%10000 == 0][f'tic_{sel_tic[0]}'].values
        y = df[df.traj_time%10000 == 0][f'tic_{sel_tic[1]}'].values
        t = df[df.traj_time%10000 == 0]['frame'].values

        if not (df['msm_weight'] == 0).all():
            weights = df[df.traj_time%10000 == 0]['msm_weight'].values
            weights_norm = (weights - np.min(weights)) / np.ptp(weights)
        else:
            weights_norm = np.ones(len(x))

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
                    size=weights_norm * 10,
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

def export_plotly_3d(tica_csv,
                     title,
                     output,
                     sel_tic = [0, 1, 2],
                     colored_by=[],
                     append=False):
    print('tica_csv: ', tica_csv)
    print('output: ', output)

    # load tica data
    try:
        plotly_df = pd.read_csv(tica_csv)
    except:
        print(f'No tica data found. Generate {tica_csv} first.')
        exit()

    fig = go.Figure()

    tic_values = []
    for tic in sel_tic:
        tic_values.append(plotly_df[plotly_df.traj_time%10000 == 0][f'tic_{tic}'].values)

    fig.add_trace(go.Scatter3d(
                x=tic_values[0], y=tic_values[1], z=tic_values[2],
                mode='markers',
                showlegend=False,

                marker=dict(
                    color='black',
                    size=3,
                    opacity=0.4,
                    showscale=False)
            )
        )

    struc_state_dic = {
                'BGT': 'CLOSED',
                'EPJ': 'DESENSITIZED',
                '7ekt': 'I (7EKT)',
                'EPJPNU': 'OPEN',
        }

    plot_states = ['BGT', 'EPJ', 'EPJPNU']

    annotations=[]

    for system, df in plotly_df.groupby('system'):
        pathway = df.pathway.unique()[0]
        pathway_text = ' to '.join([struc_state_dic[path][0] for path in pathway.split('_')])
        seed = df.seed.unique()[0]
        x = df[df.traj_time%10000 == 0][f'tic_{sel_tic[0]}'].values
        y = df[df.traj_time%10000 == 0][f'tic_{sel_tic[1]}'].values
        z = df[df.traj_time%10000 == 0][f'tic_{sel_tic[2]}'].values
        t = df[df.traj_time%10000 == 0]['frame'].values
    #    weights = df[df.traj_time%10000 == 0]['msm_weight'].values

    #    weights_norm = (weights - np.min(weights)) / np.ptp(weights)

        fig.add_trace(
            go.Scatter3d(x=x, y=y, z=z,
                name=f'{pathway_text}_{seed}',
                mode='lines+markers',
                visible='legendonly',
                legendgroup=pathway_text,
                legendgrouptitle_text=pathway_text,
                showlegend=True,
                line=dict(
                    width=1,
                    color='black',
                ),
                marker=dict(
                    color=t,
                    colorscale='Purp',
                    size=10,
    #                size=weights_norm * 10,
                    opacity=1,
                    showscale=False)
            )
        )

        if seed == 0 and pathway.split('_')[0] in plot_states:
            annotations.append(
                dict(
                    showarrow=False,
                    x=x[10],
                    y=y[10],
                    z=z[10],
                    text=struc_state_dic[pathway.split('_')[0]],
                    xanchor="left",
                    opacity=1,
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
                )
            )
            plot_states.remove(pathway.split('_')[0])

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.update_layout(scene=dict(xaxis=dict(title_text="IC 1"),
                                yaxis=dict(title_text="IC 2"),
                                zaxis=dict(title_text="IC 3"),
                                annotations=annotations))

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

def main(args):
    import argparse

    parser = argparse.ArgumentParser(description='Export plotly html from tica csv file')
    parser.add_argument('-tica_csv', type=str, default='plotly_tica.csv')
    parser.add_argument('-output', type=str, default='plotly_fes.html')
    parser.add_argument('-title', type=str, default='FES')
    args = parser.parse_args()
    export_plotly(tica_csv=args.tica_csv,
                  output=args.output,
                  title=args.title)

if __name__ == "__main__":
   main(sys.argv[1:])