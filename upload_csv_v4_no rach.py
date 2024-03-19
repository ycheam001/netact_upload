# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 09:09:43 2024

@author: cheamyk
"""

import dash
import time
from dash import html, dcc # callback_context,
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import base64
import io
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import netact_add_celltype 
import topn10_summary_v2
import insights_to_netact_v2

app = dash.Dash(__name__)
app.title = 'Upload: Network Outage Impact (5G)'
pd.set_option('mode.chained_assignment', None)

def create_trace(traces_e, df, color, name, threshold):
    "create lines for map scatterbox"

    bts = df.cellinfo.astype(str)
    lat = df.lat
    lon = df.lon
    kpi1 = df.kpi_display.astype(str)
    cell = df.cellinfo
    trace = name
    
    # Assign colors based on KPI1 values
    color2 = ['red' if val < threshold else 'green' for val in df['kpi_display']]
    # color2 = df['kpi_display'].apply(color_mapper)

    traces_e.append(go.Scattermapbox(
                    lat=lat,
                    lon=lon,
                    mode='markers',
                    marker=go.scattermapbox.Marker(
                        size = 8,
                        # size=df.kpi_display.apply(lambda x: x * 40 if x < 0.3 else x * 10),
                        color = color2
                        ),
                    showlegend=True,
                    name=trace,
                    customdata = bts +'<br>Cell: ' + cell +'<br>KPI Value: ' + kpi1 +'<br>Lat: ' + lat.astype(str)+'<br>Lon: ' + lon.astype(str),
                    hovertemplate='Serving Site: %{customdata}',
                    text=df['lat'],
                    ))
    

def create_layout(figx, title):
    "Create map layout"
    figx.update_layout(title=title,
                       mapbox_style="open-street-map",
                       mapbox_center_lat=bts_lat_avg,
                       mapbox_center_lon=bts_lon_avg,
                       mapbox_zoom=10,
                       height=600,
                       autosize=True,
                       legend=dict(x=0, y=0, orientation='v'),
                       uirevision=True)
    figx.update_layout(margin={"r":0, "t":0, "l":0, "b":10})


def cellid_cal(df_bts):
    "Calculate Cell ID"
    # ## SA (Hexa)
    # df_bts.loc[df_bts.rat == 'SA', 'site']  = df_bts.loc[df_bts.rat == 'SA']['enb']\
    # .astype(str).apply(lambda x:  math.floor((int(x,16))/8192))
    # df_bts.loc[df_bts.rat == 'SA', 'cell']  = df_bts.loc[df_bts.rat == 'SA']['enb']\
    # .astype(str).apply(lambda x:  math.floor((int(x,16)))) - df_bts['site'].apply(lambda x: x*8192)
    
    ## SA (Decimal)
    df_bts.loc[df_bts.rat == 'SA', 'site']  = df_bts.loc[df_bts.rat == 'SA']['enb']\
    .astype(str).apply(lambda x:  math.floor((int(x))/8192))
    df_bts.loc[df_bts.rat == 'SA', 'cell']  = df_bts.loc[df_bts.rat == 'SA']['enb']\
    .astype(str).apply(lambda x:  math.floor((int(x)))) - df_bts['site'].apply(lambda x: x*8192)
    
    ## LTE
    df_bts.loc[df_bts.rat == 'LTE', 'site'] = df_bts.loc[df_bts.rat == 'LTE']['enb']\
    .astype(str).apply(lambda x:  math.floor((int(x,16))/256))
    df_bts.loc[df_bts.rat == 'LTE', 'cell'] = df_bts.loc[df_bts.rat == 'LTE']['enb']\
    .astype(str).apply(lambda x:  math.floor((int(x,16)))) - df_bts['site'].apply(lambda x: x*256)
    ## Combine node + cell
    df_bts['cellinfo'] = df_bts['site'].apply(lambda x: str(int(x))) +'_' + df_bts['cell']\
    .apply(lambda x: str(int(x)))

#%% Location info
df1_n1 = pd.read_excel('Location Info\\Location_2.xlsx',sheet_name='n1')
df1_n78 = pd.read_excel('Location Info\\Location_2.xlsx',sheet_name='n78')
cellid_cal(df1_n1)
cellid_cal(df1_n78)

# Ensure aligned indices if necessary
df1_n1 = df1_n1.set_index(['site', 'cell'])
df1_n78 = df1_n78.set_index(['site', 'cell'])

df1 = df1_n78

# Calculate centre of Data points
bts_lat_avg = (df1.lat.max() + df1.lat.min())/2
bts_lon_avg = (df1.lon.max() + df1.lon.min())/2


navbar = html.Div([
    dbc.Row(
        [
            dbc.Col(html.Img(src='assets/logo.png', height="50px", style={'margin-right':'10px'})),
            dbc.Col(dbc.NavbarBrand("MapView (Outage Impact) - NetAct CSV Report Viewer")),
        ],
        style={
            'background-color': '#FAC898',
            'height': '70px',
            'display': 'flex',
            'justify-content': 'center',
            'align-items': 'center',
            'margin': '10px',
            'borderWidth': '1px',
            }
    ),
    html.Div([
        # html.A("4G", href="http://127.0.0.1:8004/", style={"width": "50px", "margin-right": "60px"}),
        # html.A("5G", href="http://127.0.0.1:8006/", style={"width": "50px", "margin-left": "60px"}),
    ], style={"display": "flex"})#, "justify-content": "space-between"})
], style={"background-color": "#f8f9fa", "padding": "10px"})

options = []
for i in range(1000):  # 20 steps from 0.05 to 0.95
    value = round(0.001 + i * 0.001, 3)  # Round to two decimal places
    label = str(value)  # Convert value to string
    options.append({'label': label, 'value': value})

#%% Webpage
app.layout = html.Div([
    navbar,
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '30px',
            'lineHeight': '30px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    dcc.Loading(
        id="loading-3",
        type="circle",
        fullscreen=True,
        children=[
            html.Div(id='output-data-upload'),
    ]), ## end of loading for upload
])

#%% Upload Logic
def parse_contents(contents, filename):
    # content_type, content_string = contents.split(',')
    if ',' in contents:
        content_type, content_string = contents.split(',')
    else:
        # content_type = None
        content_string = contents
    time_start = datetime.now()
    print("Contents of {}: {}".format(filename, content_string[0:20]))

    decoded = base64.b64decode(content_string)
    print(decoded)
    try:
        if 'csv' in filename:
            print('1. idenitified csv')
            # Assume that the user uploaded a CSV file
            # df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            header = pd.read_csv(io.StringIO(decoded.decode('utf-8')), nrows=1, sep= ';').columns
            df2 = pd.read_csv(io.StringIO(decoded.decode('utf-8')), skiprows = 2, names = header, sep= ';') # , sep= ';'
            print('2. Processed csv')
            print("[CSV] Header: {}".format(header))
        elif 'xls' in filename:
            print('1. idenitified excel')
            # Assume that the user uploaded an excel file
            header = pd.read_excel(io.BytesIO(decoded), nrows=1).columns
            df2 = pd.read_excel(io.BytesIO(decoded), skiprows = 1, names = header)
            print('2. Processed excel')
            print("[EXCEL] Header: {}".format(header))
            # df2 = pd.read_excel(io.BytesIO(decoded))
    ## load location
        df1 = df1_n78
        
        time_mid = datetime.now()

    ## Start Processing of uploaded file
        ## netact
        netact_type = "csv"
        df2['site'] = df2['NRBTS name'].astype(int)
        df2['cell'] = df2['NRCEL name'].astype(int)
        if netact_type == "excel":
            df2['Time'] = df2['Period start time']
        else:
            df2['Time'] = df2['PERIOD_START_TIME']
        netact_add_celltype.add_type(df2)
        df2['Time'] = pd.to_datetime(df2['Time'])
        
        # Format time
        time_list = df2.Time.unique()
        time_dict = [{'label': pd.to_datetime(time_list[i]).strftime("%Y-%m-%d %H:%M:%S"), 'value': time_list[i]} for i in range(len(time_list))]
        
        # Ensure aligned indices if necessary
        df3 = df2[df2['Time'] == time_dict[2]["label"]]
        df3 = df3.set_index(['site', 'cell'])

        numeric_dtypes = ['int64', 'float64']
        dropdown_dict = [{'label': col, 'value': col} for col in df3.columns[1:] if df3[col].dtype.name in numeric_dtypes]
        dropdown_t = [col for col in df2.columns[1:] if df2[col].dtype.name in numeric_dtypes]

        t_value = insights_to_netact_v2.main(df3, dropdown_t)
        kpi_select = dropdown_dict[2]["value"]
        
    ## Do the matching
        missing_indices = []  # Initialize list to store missing indices
          
        ## Match KPI from stats to location
        try:
            # Try to assign values from df2 to df1 without KeyErrors
            df1['kpi_display'] = df3.loc[df1.index, kpi_select].fillna(np.nan) 

        except KeyError as e:
                missing_indices.append(e.args[0])  # Add missing index to the list
                # Filter df1 to keep only indices present in df2
                df1 = df1[df1.index.isin(df3.index)]
                # Now you can safely assign values from df2 to df1 without KeyErrors
                df1['kpi_display'] = df3.loc[df1.index, kpi_select].fillna(np.nan)

        threshold = df3[kpi_select].max()*0.2
    ## Map plot
        # RSRP plot
        traces1 = []
        create_trace(traces1, df1, 'green', kpi_select, threshold)
        dict_of_fig = dict({'data': traces1})
        fig = go.Figure(dict_of_fig)
        create_layout(fig, 'test')

    ## Create Pivot Data
        df2_n = df2[df2['Channel'] == 'n78 outdoor']   
        df4 = df2_n.pivot_table(values=kpi_select,  # Replace with column containing values
                                      index='Time',    # Replace with the column for row labels
                                      #columns='Channel',  # Replace with the column for column labels
                                      aggfunc='mean') 

        days = len(df4)
        
        
        kpi_check_plot ='Cont based RACH stp SR'
        
        if kpi_check_plot in dropdown_t:
            # create dataframe ofr TopN 10
            # df2_n = df2[df2['Channel'] == 'n78 outdoor']
            df_top10n = topn10_summary_v2.top10n(df2_n)
            ## Graph for overall bottom 10
            fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Frequency of Low RACH SR over {} days'.format(days), 'Average RACH SR of cells frequently appearing in TopN Bottom 10'), 
                                  specs=[[{"secondary_y": True}, {"secondary_y": False}]], horizontal_spacing = 0.1)
        
        
            fig2.add_trace(
                go.Bar(
                    x=df_top10n.index,  # Use bin edges as x values
                    y=df_top10n['Count'],  # Use percentages as y values
                    name = "Count",
                ),
                secondary_y=False,
                row=1, col=1
            )
        
            fig2.add_trace(
                go.Scatter(
                    x=df_top10n.index,  # Use bin edges as x values
                    y=df_top10n['%'],  # Use percentages as y values
                    name = "%",
                ),
                secondary_y=True,
                row=1, col=1
            )
        
            fig2.add_trace(
                go.Bar(
                    x=df_top10n.index,  # Use bin edges as x values
                    y=df_top10n['Avg RACH SR'],  # Use percentages as y values
                    name = "Avg RACH SR",
                ),
                row=1, col=2
            )
            # fig2.layout.title.text = "TopN 10 Lowest for {}".format(days)
        
        
            fig2['layout']['xaxis']['title']='gNB'
            fig2['layout']['xaxis2']['title']='gNB'
            fig2['layout']['yaxis']['title']='Count'
            fig2['layout']['yaxis2']['title']='Succes Rate %'
            fig2['layout']['yaxis3']['title']='Percentage'
        
            fig2.update_layout(
                height=500,
                margin=go.layout.Margin(l=2, r=2, t=25, b=5),  # Adjust margins here
            )
        else:
            fig2 = go.Figure(
                    layout=go.Layout(
                        height=10, 
                        # width=800,
                        margin=go.layout.Margin(l=2, r=2, t=15, b=2),  # Adjust margins here
                        )
                )
        
        ## Overall graph
        fig1 = go.Figure(data=[
            go.Scatter(
                x=df4.index,  # Use bin edges as x values
                y=df4[kpi_select],  # Use percentages as y values
            )
        ])
        
        # print(df2)
        print('3. Complete')
        time_end = datetime.now()
        time_1 = (time_mid-time_start).total_seconds()
        time_2 = (time_end-time_mid).total_seconds()
        print("Stage 1: {}s".format(time_1))
        print("Stage 2: {}s".format(time_2))
        global df2_g 
        
        df2_g = df2
        
        return html.Div([
            html.H5(filename),
            html.Hr(),  # horizontal line
            html.Div([
                html.Div([
                    html.Label("Time:"),
                    dcc.Dropdown(
                        id='hour-slider2',
                        options= time_dict,
                        value=time_dict[0]["value"]
                    ),
                ], style={'width': '15%', 'display': 'inline-block'}),
                html.Div([], style={'width': '3%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("RAT:"),
                    dcc.Dropdown(
                        id='dropdown-rat',
                        options=[
                            {'label': '4G', 'value': '4g'},
                            {'label': '5G', 'value': '5g'}
                        ],
                        value='5g'
                    ),
                ], style={'width': '10%', 'display': 'inline-block'}),
                html.Div([], style={'width': '3%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Layer:"),
                    dcc.Dropdown(
                        id='dropdown-layer',
                        options=[
                            {'label': 'n78', 'value': 'n78'},
                            {'label': 'n1', 'value': 'n1'}
                        ],
                        value='n78'
                    ),
                ], style={'width': '10%', 'display': 'inline-block'}),
                html.Div([], style={'width': '3%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("KPI:"),
                    dcc.Dropdown(
                        id='dropdown-kpi',
                        options=dropdown_dict,
                        value=dropdown_dict[2]["value"] 
                    ),
                ], style={'width': '25%', 'display': 'inline-block'}),
                html.Div([], style={'width': '3%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Threshold:"),
                    # dcc.Input(id="dropdown-threshold", type="number", placeholder="", 
                    #           min=0.05, max=1, step=0.01, value=0.9, debounce = True),
                    dcc.Dropdown(
                        id='dropdown-threshold',
                        options=options,
                        value=0.9  # Default selected value
                    ),
                ], style={'width': '15%', 'display': 'inline-block'}),
                #     html.P(id='slider-output-container'),
            ]),
            html.Hr(),  # horizontal line
            dcc.Loading(
                id="loading-1",
                type="default",
                fullscreen=True, 
                children=[
                    html.P(id='slider-output-container'),
                    html.Div([
                        dcc.Graph(id='user-graph1', figure=fig),                        
                    ], style={'width': '68%', 'float': 'left'}),
            ]), ## end of dcc loading 1
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H4('TopN 10 Lowest: '),
                        html.Table(id='topn-output'),
                    ], style={'width': '50%', "display": "inline-block", 'float': 'left'}),
                    
                    html.Div([
                        html.H4('Statistics: '),
                        dcc.Loading(
                            id="loading-2",
                            type="circle",
                            children=[
                                html.P(id='time-output'),
                                html.P(id='count-total-output'),
                                html.P(id='count-below-output'),
                                html.P(id='failure-output'),
                        ]), ## end of dcc loading 2
                    ], style={'width': '45%', "display": "inline-block", 'margin-left':'5px'}), #'display': 'inline-block', 
                ], style ={"display": "flex", "flex-direction": "row"}),

                html.Div([
                    dcc.Loading(
                        id="loading-3",
                        type="circle",
                        children=[
                            dcc.Graph(id='kpi-plot1', figure=fig1),
                    ]), ## end of dcc loading 3
                ], style ={"display": "flex", "flex-direction": "column"}),
                
            ], style={'width': '30%', 'display': 'inline-block', 'margin-left':'25px'}),
            
            html.Hr(),  # horizontal line
            html.Div([
                html.H3("TopN 10 Lowest for {}".format(days)),
                dcc.Graph(id='kpi-plot2', figure=fig2),
            ]),
            html.Div([
                html.H4('List of KPIs and Recommended Thresholds'),
                html.Table(
                    [
                        html.Tr([html.Th(col) for col in ['KPI', 'Threshold', 'Min', 'Max', 'Mean']])
                    ]
                    + 
                    [
                        html.Tr([
                            html.Td(dropdown_dict[i]["value"]),
                            html.Td(t_value[i]),
                            html.Td(df2[dropdown_dict[i]["value"]].min()),
                            html.Td(round(df2[dropdown_dict[i]["value"]].max(),1)),
                            html.Td(round(df2[dropdown_dict[i]["value"]].mean(),2)),
                            # html.Td(dropdown_dict.get(dropdown_dict[i][0], 'N/A')),
                        ]) for i in range(len(dropdown_dict))
                    ], style={"border": "1px solid #ddd"}
                ),
            ]),
            html.Hr(),  # horizontal line
            html.P("header : {}".format(header)),
            html.Hr(),  # horizontal line
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in df2.columns])] +

                # Body
                [html.Tr([
                    html.Td(df2.iloc[i][col]) for col in df2.columns
                ]) for i in range(min(len(df2), 10))]
            )
        ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file: ' + filename
            
        ])
    except KeyError as e_1:
        print(e_1)
        return html.Div([
            'KeyError: The selected_rf column does not exist in the DataFrame.'
        ])
    except ValueError as e_1:
        print(e_1)
        return html.Div([
            'ValueError: There is a ValueError in the filter condition.'
        ])
    except TypeError as e_1:
        print(e_1)
        return html.Div([
            'TypeError: There is a TypeError in the filter condition.'
        ])
    except NameError as e_1:
        print(e_1)
        return html.Div([
            'NameError: One or more variables are not defined.'
        ])




@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def update_output(contents, filename):
    print(filename)
    if contents is not None:
        children = parse_contents(contents, filename)
        return children
#%%

# Define callback for uploading and displaying data
@app.callback(
    [#Output('table', 'data'),
     #Output('table', 'columns'),
     Output('slider-output-container', 'children'),
     Output('user-graph1', 'figure'),
     Output('kpi-plot1', 'figure')],
     Output("time-output", "children"),
     Output("count-total-output", "children"),
     Output("count-below-output", "children"),
     Output("failure-output", "children"),
     Output("topn-output", "children"),
    [#Input('upload-data', 'contents'),
     Input('dropdown-kpi', 'value'),
     Input('dropdown-layer', 'value'),
     Input('dropdown-threshold', 'value'),
     Input('hour-slider2', 'value'),]
)
def update_table(kpi, layer, threshold, selected_hour):
        
        time.sleep(2)
        print(selected_hour)
        missing_indices = []  # Initialize list to store missing indices
    
        if selected_hour is not None:
            # df['Real Time'] = pd.to_datetime(df['Real Time'])
            # df3 = df2[df2['Time'].dt.hour == selected_hour]
            selected_hour = pd.to_datetime(selected_hour).strftime("%Y-%m-%d %H:%M:%S")
            df3 = df2_g[df2_g['Time'] == selected_hour]
            
            # 2024-01-31 13:00:00:00
            # 2024-01-30 18:00:00

        # df3 = df3.set_index('enb')
        # Set aligned indices for efficient matching
        df3 = df3.set_index(['site', 'cell'])
        print(df3.take([2]))
        
        # df1_2 = df1
        if layer is not None:
            if layer == 'n1':
                df1_2 = df1_n1
                df3 = df3[df3['Channel'] == 'n1 outdoor']
                df2_n = df2_g[df2_g['Channel'] == 'n1 outdoor']
            else:
                df1_2 = df1_n78
                df3 = df3[df3['Channel'] == 'n78 outdoor']
                df2_n = df2_g[df2_g['Channel'] == 'n78 outdoor']
        # print(df1_2.take([2]))
        

        try:
            # Attempt filling with error handling
            df1_2['kpi_display'] = df3.loc[df1_2.index, kpi].fillna(np.nan)  # Fill missing values with NaN
            print('Filtering Success')
            print('Filtering Completed, MapView to be updated')

        except KeyError as e:
            missing_indices.append(e.args[0])  # Add missing index to the list
            # Filter df1 to keep only indices present in df3
            df1_2 = df1_2[df1_2.index.isin(df3.index)]
            print('Filtering Fail')
            # Now you can safely assign values from df2 to df1 without KeyErrors
            df1_2['kpi_display'] = df3.loc[df1_2.index, kpi].fillna(np.nan)

        
        
        ## Check
        # print(df1_2)              ## dataframe: to display on map
        print(len(missing_indices)) ## location with no match
        # print(df1)                ## dataframe: location 
        # print(df3)                ## dataframe: kpi
        
        lat_mid= (df1.lat.max() + df1.lat.min())/2
        lon_mid= (df1.lon.max() + df1.lon.min())/2
        
        threshold2= df2_g[kpi].max()*threshold
        # threshold2 = threshold
        thresholdmax= df3[kpi].max()
        thresholdmin= df3[kpi].min()
        print(thresholdmax)
        print(thresholdmin)
        
        count_total = df1_2['kpi_display'].count()
        count_fail = df1_2['kpi_display'].lt(threshold2).sum()
        failure = (count_fail/count_total)*100
        lowest_10_indices = df3[kpi].nsmallest(10).index
        
         
        table = html.Table(
            [
                html.Tr([html.Th(col) for col in ['No.', 'Site', 'Cell', 'Value']], style={"border": "1px solid black", "padding":"2px"})
            ]
            + 
            [
                html.Tr([
                    html.Td(i+1, style={"border": "1px solid black", "padding":"2px"}),
                    html.Td(lowest_10_indices[i][0], style={"border": "1px solid black", "padding":"2px"}),
                    html.Td(lowest_10_indices[i][1], style={"border": "1px solid black", "padding":"2px"}),
                    html.Td(df3.loc[lowest_10_indices[i],kpi], style={"border": "1px solid black", "padding":"2px"})
                ]) for i in range(len(lowest_10_indices))
            ], style={"border": "1px solid black", "padding":"2px"} #1px solid #dddblack
        ),

        
        traces_1 = []
        create_trace(traces_1, df1_2, 'blue', kpi, threshold2)
        
                
        layout = go.Layout(
            mapbox_style='open-street-map',
            mapbox_center_lat=lat_mid,
            mapbox_center_lon=lon_mid,
            mapbox_zoom=10,
            margin={"r":0, "t":0, "l":0, "b":10},
            height=600,
            autosize=True,
            legend=dict(x=0, y=0, orientation='v'),
            uirevision=True)
        
        
        df4 = df2_n.pivot_table(values=kpi,#kpi,  # Replace with column containing values
                                     index='Time',    # Replace with the column for row labels
                                     #columns='Channel',  # Replace with the column for column labels
                                     aggfunc='mean') 
        
        fig1_update = go.Figure(
            data=[
                go.Scatter(
                    x=df4.index,  # Use the same x values
                    y=df4[kpi],  # CDF as y values
                    name = kpi,
                ),
                go.Line(  # New line for horizontal reference
                    x=[df4.index.min(), df4.index.max()],
                    y=[threshold2, threshold2],
                    line=dict(color='red', dash='dash'),  # Customize line style
                    name = "Threshold",
                ),
            ],
            layout=go.Layout(
                height=260, 
                # width=800,
                margin=go.layout.Margin(l=2, r=2, t=15, b=2),  # Adjust margins here
                )
        )

        if not df3.empty: # [{:.2f} ->  {:.2f} t:{:.2f}]  threshold, thresholdmin, thresholdmax,
            return ' KPI: {} for {} '.format(kpi, layer),{'data': traces_1, 'layout':layout}, fig1_update, 'Time: {}'.format(selected_hour),\
                'Total Count: {}'.format(count_total), 'Count Below threshold [{}]: {}'.format(threshold2, count_fail),\
                '% Below threshold: {:.2f}'.format(failure), table #'TopN: {}'.format(for item in lowest_10_indices)
                
        else:
            return 'No Data', {'data': [], 'layout':[]}, fig1_update
#%%

if __name__ == '__main__':
    app.run_server(debug=False, port=8007, host='0.0.0.0')