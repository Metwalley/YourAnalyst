import pandas as pd
import joblib
from datetime import timedelta
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Load Data & Pre-trained Model --------------------------------

# Paths
TRAINING_DATA_PATH = "ready_for_training.csv"
EDA_DATA_PATH = "dash.csv"
MODEL_PATH = "trained_model.pkl"

# Load datasets
df_all = pd.read_csv(TRAINING_DATA_PATH)
df_eda = pd.read_csv(EDA_DATA_PATH)

# Convert dates
df_all['Date'] = pd.to_datetime(
    df_all[['Date_year','Date_month','Date_day']]
    .rename(columns={'Date_year':'year','Date_month':'month','Date_day':'day'})
)

df_eda['Date'] = pd.to_datetime(df_eda['Date'])

# Load trained model
sales_model = joblib.load(MODEL_PATH)

# --- 2. Prediction Helper --------------------------------------------

def predict_sales(store_id, last_date, periods, df_ref):
    # Build future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    fut = pd.DataFrame({'Date': future_dates})

    # Static store info
    store_info = df_ref[df_ref['Store']==store_id].iloc[-1]
    fut['Store'] = store_id
    fut['StoreType'] = store_info['StoreType']
    fut['StoreType'] = store_info['StoreType']
    fut['CompetitionDistance'] = store_info['CompetitionDistance']
    fut['Promo2'] = store_info['Promo2']
    fut['Promo2SinceWeek'] = store_info['Promo2SinceWeek']
    fut['Promo2SinceYear'] = store_info['Promo2SinceYear']

    # Default dynamic features
    fut['Customers'] = df_ref[df_ref['Store']==store_id]['Customers'].median()
    fut['Open'] = 1
    fut['Promo'] = 0
    fut['SchoolHoliday'] = 0

    # One-hot intervals
    for col in ['PromoInterval_0','PromoInterval_1','PromoInterval_2','PromoInterval_3']:
        fut[col] = store_info.get(col, False)
    # One-hot holiday & assortment
    for col in ['StateHoliday_0','Assortment_0','Assortment_1','Assortment_2']:
        fut[col] = store_info.get(col, False)
    # Date features
    fut['DayOfWeek'] = fut['Date'].dt.dayofweek + 1
    fut['Date_year'] = fut['Date'].dt.year
    fut['Date_month'] = fut['Date'].dt.month
    fut['Date_day'] = fut['Date'].dt.day
    # One-hot day of week
    for i in range(1,8): fut[f'DayOfWeek_{i}'] = (fut['DayOfWeek']==i)

    # Select features in training order
    feature_cols = [
        'Store','StoreType','CompetitionDistance','Promo2','Promo2SinceWeek','Promo2SinceYear',
        'Customers','Open','Promo','SchoolHoliday',
        'Date_year','Date_month','Date_day',
        'PromoInterval_0','PromoInterval_1','PromoInterval_2','PromoInterval_3',
        'StateHoliday_0','Assortment_0','Assortment_1','Assortment_2',
        'DayOfWeek_1','DayOfWeek_2','DayOfWeek_3','DayOfWeek_4','DayOfWeek_5','DayOfWeek_6','DayOfWeek_7'
    ]
    X = fut[feature_cols]

    # Predict
    yhat = sales_model.predict(X)
    return pd.DataFrame({'ds': fut['Date'], 'yhat': yhat})

# --- 3. Dash App Initialization --------------------------------------

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "Sales Forecasting & Optimization"

# Custom styles
CARD_STYLE = {
    'border': '1px solid #444',
    'borderRadius': '5px',
    'padding': '15px',
    'marginBottom': '15px',
    'backgroundColor': '#303030'
}

app.layout = dbc.Container([
    html.H1("📈 Sales Forecasting & Optimization", className="text-center my-4"),
    
    # Tabs
    dbc.Tabs(id='tabs', active_tab='tab-explore', children=[
        dbc.Tab(label='Data Exploration', tab_id='tab-explore'),
        dbc.Tab(label='Forecast', tab_id='tab-forecast'),
        dbc.Tab(label='Optimize', tab_id='tab-optimize'),
    ], className="mb-4"),
    
    # Tab content
    html.Div(id='tab-content')
], fluid=True, style={'padding': '20px'})

# --- 4. Callbacks -------------------------------------------------------

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(tab):
    if tab == 'tab-explore':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.H4("Filters", className="card-title"),
                        
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='eda-date-range',
                            min_date_allowed=df_eda['Date'].min(),
                            max_date_allowed=df_eda['Date'].max(),
                            start_date=df_eda['Date'].min(),
                            end_date=df_eda['Date'].max(),
                            className="mb-3"
                        ),
                        
                        html.Label("Store Type:"),
                        dcc.Dropdown(
                            id='store-type-filter',
                            options=[{'label': typ, 'value': typ} for typ in sorted(df_eda['StoreType'].unique())],
                            multi=True,
                            placeholder="All Store Types",
                            className="mb-3"
                        ),
                        
                        html.Label("Assortment Type:"),
                        dcc.Dropdown(
                            id='assortment-filter',
                            options=[{'label': ass, 'value': ass} for ass in sorted(df_eda['Assortment'].unique())],
                            multi=True,
                            placeholder="All Assortments",
                            className="mb-3"
                        ),
                        
                        html.Label("Promo Status:"),
                        dcc.Dropdown(
                            id='promo-filter',
                            options=[
                                {'label': 'Promo Active', 'value': 1},
                                {'label': 'Promo Inactive', 'value': 0}
                            ],
                            multi=True,
                            placeholder="All Promo Statuses",
                            className="mb-3"
                        ),
                        
                        html.Label("Day of Week:"),
                        dcc.Dropdown(
                            id='dow-filter',
                            options=[{'label': f'Day {i}', 'value': i} for i in range(1,8)],
                            multi=True,
                            placeholder="All Days",
                            className="mb-3"
                        ),
                        
                        dbc.Button("Apply Filters", id='apply-filters', color="primary", className="w-100")
                    ], style=CARD_STYLE)
                ], width=3),
                
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sales-trend'), width=12, className="mb-4"),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sales-by-store-type'), width=6),
                        dbc.Col(dcc.Graph(id='sales-by-assortment'), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sales-by-promo'), width=6),
                        dbc.Col(dcc.Graph(id='sales-by-day'), width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='sales-vs-customers'), width=6),
                        dbc.Col(dcc.Graph(id='comp-distance-effect'), width=6),
                    ])
                ], width=9)
            ])
        ], fluid=True)
    
    elif tab == 'tab-forecast':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.H4("Forecast Settings", className="card-title"),
                        html.Label("Store:"),
                        dcc.Dropdown(
                            id='store-dropdown', 
                            clearable=False,
                            options=[{'label': f"Store {i}", 'value': i} for i in sorted(df_all['Store'].unique())],
                            value=sorted(df_all['Store'].unique())[0],
                            className="mb-3"
                        ),
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='date-range',
                            min_date_allowed=df_all['Date'].min(),
                            max_date_allowed=df_all['Date'].max(),
                            start_date=df_all['Date'].min(),
                            end_date=df_all['Date'].max(),
                            className="mb-3"
                        ),
                        html.Label("Forecast Horizon (days):"),
                        dcc.Slider(
                            id='horizon-slider', 
                            min=7, max=90, step=1, value=30,
                            marks={7:'7',30:'30',60:'60',90:'90'},
                            className="mb-3"
                        ),
                    ], style=CARD_STYLE)
                ], width=3),
                dbc.Col([
                    dcc.Graph(id='forecast-chart')
                ], width=9)
            ])
        ], fluid=True)
    
    elif tab == 'tab-optimize':
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        html.H4("Optimization Settings", className="card-title"),
                        html.Label("Store:"),
                        dcc.Dropdown(
                            id='opt-store-dropdown', 
                            clearable=False,
                            options=[{'label': f"Store {i}", 'value': i} for i in sorted(df_all['Store'].unique())],
                            value=sorted(df_all['Store'].unique())[0],
                            className="mb-3"
                        ),
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id='opt-date-range',
                            min_date_allowed=df_all['Date'].min(),
                            max_date_allowed=df_all['Date'].max(),
                            start_date=df_all['Date'].min(),
                            end_date=df_all['Date'].max(),
                            className="mb-3"
                        ),
                        html.Label("Forecast Horizon (days):"),
                        dcc.Slider(
                            id='opt-horizon-slider', 
                            min=7, max=90, step=1, value=30,
                            marks={7:'7',30:'30',60:'60',90:'90'},
                            className="mb-3"
                        ),
                        html.Label("Safety Stock Percentage:"),
                        dcc.Slider(
                            id='safety-stock-slider',
                            min=5, max=50, step=5, value=20,
                            marks={5:'5%', 10:'10%', 20:'20%', 30:'30%', 40:'40%', 50:'50%'}
                        ),
                    ], style=CARD_STYLE)
                ], width=3),
                dbc.Col([
                    dcc.Graph(id='optimize-chart')
                ], width=9)
            ])
        ], fluid=True)

# EDA Callbacks
@app.callback(
    [Output('sales-trend', 'figure'),
     Output('sales-by-store-type', 'figure'),
     Output('sales-by-assortment', 'figure'),
     Output('sales-by-promo', 'figure'),
     Output('sales-by-day', 'figure'),
     Output('sales-vs-customers', 'figure'),
     Output('comp-distance-effect', 'figure')],
    [Input('apply-filters', 'n_clicks')],
    [State('eda-date-range', 'start_date'),
     State('eda-date-range', 'end_date'),
     State('store-type-filter', 'value'),
     State('assortment-filter', 'value'),
     State('promo-filter', 'value'),
     State('dow-filter', 'value')]
)
def update_eda_charts(n_clicks, start_date, end_date, store_types, assortments, promos, days):
    # Apply filters
    filtered_df = df_eda.copy()
    
    # Date filter
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & 
                                 (filtered_df['Date'] <= end_date)]
    
    # Store type filter
    if store_types and len(store_types) > 0:
        filtered_df = filtered_df[filtered_df['StoreType'].isin(store_types)]
    
    # Assortment filter
    if assortments and len(assortments) > 0:
        filtered_df = filtered_df[filtered_df['Assortment'].isin(assortments)]
    
    # Promo filter
    if promos and len(promos) > 0:
        filtered_df = filtered_df[filtered_df['Promo'].isin(promos)]
    
    # Day of week filter
    if days and len(days) > 0:
        filtered_df = filtered_df[filtered_df['DayOfWeek'].isin(days)]
    
    # Create figures with dark theme
    def apply_dark_theme(fig):
        fig.update_layout(
            plot_bgcolor='#303030',
            paper_bgcolor='#303030',
            font_color='white',
            xaxis=dict(gridcolor='#555'),
            yaxis=dict(gridcolor='#555')
        )
        return fig
    
    # 1. Sales Trend
    trend_df = filtered_df.groupby('Date')['Sales'].mean().reset_index()
    fig_trend = px.line(trend_df, x='Date', y='Sales', title='Average Sales Trend')
    fig_trend = apply_dark_theme(fig_trend)
    
    # 2. Sales by Store Type
    store_type_df = filtered_df.groupby('StoreType')['Sales'].mean().reset_index()
    fig_store_type = px.bar(store_type_df, x='StoreType', y='Sales', 
                           title='Average Sales by Store Type', color='StoreType')
    fig_store_type = apply_dark_theme(fig_store_type)
    
    # 3. Sales by Assortment
    assortment_df = filtered_df.groupby('Assortment')['Sales'].mean().reset_index()
    fig_assortment = px.bar(assortment_df, x='Assortment', y='Sales', 
                           title='Average Sales by Assortment Type', color='Assortment')
    fig_assortment = apply_dark_theme(fig_assortment)
    
    # 4. Sales by Promo
    promo_df = filtered_df.groupby('Promo')['Sales'].mean().reset_index()
    promo_df['Promo'] = promo_df['Promo'].map({0: 'No Promo', 1: 'Promo Active'})
    fig_promo = px.bar(promo_df, x='Promo', y='Sales', 
                      title='Average Sales by Promo Status', color='Promo')
    fig_promo = apply_dark_theme(fig_promo)
    
    # 5. Sales by Day of Week
    day_df = filtered_df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    fig_day = px.line(day_df, x='DayOfWeek', y='Sales', 
                     title='Average Sales by Day of Week', markers=True)
    fig_day.update_xaxes(tickvals=list(range(1,8)))
    fig_day = apply_dark_theme(fig_day)
    
    # 6. Sales vs Customers
    fig_customers = px.scatter(filtered_df, x='Customers', y='Sales', 
                              trendline='ols', title='Sales vs Customers',
                              color='StoreType')
    fig_customers = apply_dark_theme(fig_customers)
    
    # 7. Competition Distance Effect
    comp_df = filtered_df[filtered_df['CompetitionDistance'] < filtered_df['CompetitionDistance'].quantile(0.95)]
    fig_comp = px.scatter(comp_df, x='CompetitionDistance', y='Sales', 
                         trendline='ols', title='Sales vs Competition Distance',
                         color='StoreType')
    fig_comp = apply_dark_theme(fig_comp)
    
    return fig_trend, fig_store_type, fig_assortment, fig_promo, fig_day, fig_customers, fig_comp

# Forecast Callback
@app.callback(
    Output('forecast-chart', 'figure'),
    [Input('store-dropdown', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date'),
     Input('horizon-slider', 'value')]
)
def update_forecast_chart(store_id, start_date, end_date, horizon):
    # Filter historical data
    d = df_all[(df_all['Store']==store_id) &
               (df_all['Date']>=start_date) &
               (df_all['Date']<=end_date)].copy()
    
    # Create forecast
    last_date = d['Date'].max()
    fc_df = predict_sales(store_id, last_date, horizon, df_all)
    
    # Prepare data for plotting
    hist = d.rename(columns={'Date':'ds','Sales':'y'}).assign(type='history')
    fut = fc_df.rename(columns={'yhat':'y'}).assign(type='forecast')
    combined = pd.concat([hist[['ds','y','type']], fut])
    
    # Create figure
    fig = px.line(combined, x='ds', y='y', color='type', 
                 labels={'ds':'Date','y':'Sales','type':'Series'}, 
                 title=f'Actual vs Forecasted Sales ({horizon}d)')
    
    # Apply dark theme
    fig.update_layout(
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        font_color='white',
        xaxis=dict(gridcolor='#555'),
        yaxis=dict(gridcolor='#555')
    )
    
    return fig

# Optimization Callback
@app.callback(
    Output('optimize-chart', 'figure'),
    [Input('opt-store-dropdown', 'value'),
     Input('opt-date-range', 'start_date'),
     Input('opt-date-range', 'end_date'),
     Input('opt-horizon-slider', 'value'),
     Input('safety-stock-slider', 'value')]
)
def update_optimize_chart(store_id, start_date, end_date, horizon, safety_pct):
    # Filter historical data
    d = df_all[(df_all['Store']==store_id) &
               (df_all['Date']>=start_date) &
               (df_all['Date']<=end_date)].copy()
    
    # Create forecast
    last_date = d['Date'].max()
    fc_df = predict_sales(store_id, last_date, horizon, df_all)
    
    # Calculate reorder levels
    fc_df['safety_stock'] = fc_df['yhat'] * (safety_pct/100)
    fc_df['reorder_level'] = fc_df['yhat'] + fc_df['safety_stock']
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=fc_df['ds'], y=fc_df['yhat'], name="Forecasted Sales"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=fc_df['ds'], y=fc_df['reorder_level'], name="Reorder Level"),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=fc_df['ds'], y=fc_df['safety_stock'], name="Safety Stock"),
        secondary_y=True,
    )
    
    # Add figure title and labels
    fig.update_layout(
        title_text=f"Reorder Level Recommendation (Safety Stock: {safety_pct}%)",
        plot_bgcolor='#303030',
        paper_bgcolor='#303030',
        font_color='white',
        xaxis=dict(gridcolor='#555'),
        yaxis=dict(gridcolor='#555', title="Sales / Reorder Level"),
        yaxis2=dict(gridcolor='#555', title="Safety Stock")
    )
    
    return fig

# --- 5. Run -----------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
