import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from textwrap import dedent as d
import sys
import os

import log_analyzer as la

app = dash.Dash(__name__)
log_folder = sys.argv[-1]
assert os.path.isdir(log_folder), 'Last argument was not a folder'

log_folder = log_folder + '/'
metrics_to_plot = ['loss', 'map', 'gmauc', 'lp_map', 'lp_auc', 'auc']

logs, csv, best_log_file = la.parse_all_logs_in_folder(log_folder)

def plot_metric(metric_name, log):
    plot_train = metric_name == 'loss'
    print(metric_name, log)
    fig = la.plot_metric(logs[log]['metrics'][metric_name], metric_name, plot_train)
    return fig

graphs = [dcc.Graph(id='graph-'+metric_name, figure=plot_metric(metric_name, best_log_file))
          for metric_name in metrics_to_plot]

app.layout = html.Div(children=[
    html.H1(children='DGNN Log'),
    html.Label('Log'),
        dcc.Dropdown(
            id='log-select',
            options=[{'label': i , 'value': i} for i in logs.keys()],
            value=best_log_file,
            multi=False
        ),
    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d('Best Epoch')),
            html.Pre(id='best-epoch')
        ])
    ]),
    html.Div(children=graphs)
])

@app.callback(
    Output('best-epoch', 'children'),
    [Input('log-select', 'value')])
def update_best_epoch(log):
    return logs[log]['res_map']['best_epoch']

# Hacky, but works
def add_graph_updates(metric_name):
    @app.callback(
        Output('graph-'+metric_name, 'figure'),
        [Input('log-select', 'value')])
    def update_graphs(log):
        return plot_metric(metric_name, log)

    return update_graphs

for metric_name in metrics_to_plot:
    globals()['update_graph_{}'.format(metric_name)] = add_graph_updates(metric_name)

if __name__ == '__main__':
    app.run_server(debug=True)
