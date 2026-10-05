"""
Module 4: Data Visualization for Statistics
- Matplotlib vs. Seaborn vs. Plotly / Dash Comparison
- Advanced Statistical Visualizations (Boxplots, Violin plots, Heatmaps, 3D Scatter)
- Interactive Chart Selection Wizard
Uses: climate_change_impact_on_agriculture_2024.csv
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, callback

import plotly.io as pio
pio.templates.default = "plotly_dark"

PLOT_LAYOUT = {
    'template': 'plotly_dark',
    'paper_bgcolor': '#081a13',
    'plot_bgcolor': '#0c241b',
    'font': {'color': '#f0fdf4', 'family': 'Plus Jakarta Sans, sans-serif'},
    'xaxis': {'gridcolor': 'rgba(52, 211, 153, 0.15)', 'zerolinecolor': 'rgba(52, 211, 153, 0.3)', 'tickfont': {'color': '#93c5b5'}, 'title_font': {'color': '#f0fdf4'}},
    'yaxis': {'gridcolor': 'rgba(52, 211, 153, 0.15)', 'zerolinecolor': 'rgba(52, 211, 153, 0.3)', 'tickfont': {'color': '#93c5b5'}, 'title_font': {'color': '#f0fdf4'}},
    'margin': {'l': 40, 'r': 30, 't': 40, 'b': 40}
}
DARK_LAYOUT = PLOT_LAYOUT

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "data", "climate_change_impact_on_agriculture_2024.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    if os.path.exists("data/climate_change_impact_on_agriculture_2024.csv"):
        return pd.read_csv("data/climate_change_impact_on_agriculture_2024.csv")
    return pd.DataFrame()

def layout():
    return html.Div([
        html.Div([
            html.Span("Phase 4 — Visual Intelligence", className="section-badge"),
            html.H2("Agro-Meteorological Visual Explorer & Analytics Suite", className="section-title"),
            html.P("Compare analytical visualization frameworks, explore multidimensional 3D climate-yield dynamics, and receive smart visual recommendations.", className="section-subtitle")
        ]),
        
        dcc.Tabs(id="m4-subtabs", value="tool-comparison", children=[
            dcc.Tab(label="1. Data Science Graphics Ecosystem", value="tool-comparison"),
            dcc.Tab(label="2. Multidimensional Agro-Climatic Gallery", value="chart-gallery"),
            dcc.Tab(label="3. Intelligent Chart Selection Guide", value="chart-wizard")
        ], className="mb-4"),
        
        html.Div(id="m4-content")
    ])

def comparison_layout():
    return html.Div([
        html.Div([
            # Matplotlib Card
            html.Div([
                html.Div([
                    html.H4("Matplotlib", className="text-primary font-weight-bold"),
                    html.Span("Low-level Procedural / Object-Oriented", className="badge bg-primary text-white mb-3"),
                    html.P("The foundational plotting engine in Python. Granular control over every axis, spine, tick, and pixel. Ideal for publication-ready static figures (LaTeX/PDF papers).", className="text-muted small"),
                    html.Ul([
                        html.Li("Exact canvas layout control"),
                        html.Li("Extensive 2D plotting APIs"),
                        html.Li("Static raster/vector exports")
                    ], className="small text-light pl-3")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            # Seaborn Card
            html.Div([
                html.Div([
                    html.H4("Seaborn", className="text-info font-weight-bold"),
                    html.Span("High-level Statistical Abstraction", className="badge bg-info text-white mb-3"),
                    html.P("Built on top of Matplotlib, designed specifically for pandas DataFrames and exploratory statistical modeling with built-in regression trendlines and distribution fits.", className="text-muted small"),
                    html.Ul([
                        html.Li("Built-in confidence intervals & KDE"),
                        html.Li("Violin, Boxen, Jointplot, PairGrid"),
                        html.Li("Cohesive color palettes")
                    ], className="small text-light pl-3")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            # Plotly & Dash Card
            html.Div([
                html.Div([
                    html.H4("Plotly & Dash", className="text-success font-weight-bold"),
                    html.Span("Interactive Web & Reactive Dashboards", className="badge bg-success text-white mb-3"),
                    html.P("Web-first declarative visualization engine (WebGL / D3.js) and full-stack reactive framework for creating production-ready analytic applications.", className="text-muted small"),
                    html.Ul([
                        html.Li("Dynamic zoom, pan, hover tooltips"),
                        html.Li("Multi-dimensional 3D & geospatial maps"),
                        html.Li("Reactive server-side callbacks")
                    ], className="small text-light pl-3")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4")
        ], className="row")
    ])

def chart_gallery_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Statistical Gallery Controls", className="control-label"),
                    html.P("Select advanced statistical visualizations on the 10,000 observation climate dataset.", className="control-desc"),
                    
                    html.Label("Chart Technique:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m4-gallery-choice",
                        options=[
                            {'label': 'Violin + Boxplot Overlay (Yield across Adaptation Strategies)', 'value': 'violin'},
                            {'label': 'Correlation Matrix Heatmap with Coefficients', 'value': 'heatmap'},
                            {'label': '3D Statistical Scatter (Temp vs Precipitation vs Yield)', 'value': '3d_scatter'},
                            {'label': '2D Density Contour / Heatmap', 'value': 'density_contour'}
                        ],
                        value='violin',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Group or Color Variable:", className="control-label"),
                    dcc.Dropdown(
                        id="m4-gallery-group",
                        options=[
                            {'label': 'Adaptation Strategies', 'value': 'Adaptation_Strategies'},
                            {'label': 'Crop Type', 'value': 'Crop_Type'},
                            {'label': 'Country', 'value': 'Country'}
                        ],
                        value='Adaptation_Strategies',
                        clearable=False,
                        className="mb-3"
                    )
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="m4-graph-gallery", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-lg-8 mb-4")
        ], className="row")
    ])

def chart_wizard_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Chart Selector Wizard", className="control-label"),
                    html.P("Input your variable types and analysis objective to get tailored statistical visual recommendations.", className="control-desc"),
                    
                    html.Label("Primary Variable (X):", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m4-wiz-x",
                        options=[
                            {'label': 'Continuous Numerical (e.g. Temperature, Yield)', 'value': 'continuous'},
                            {'label': 'Categorical / Nominal (e.g. Crop Type, Country)', 'value': 'categorical'},
                            {'label': 'Temporal / Time-Series (e.g. Year)', 'value': 'time'}
                        ],
                        value='continuous',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Secondary Variable (Y):", className="control-label"),
                    dcc.Dropdown(
                        id="m4-wiz-y",
                        options=[
                            {'label': 'None (Univariate Analysis)', 'value': 'none'},
                            {'label': 'Continuous Numerical', 'value': 'continuous'},
                            {'label': 'Categorical / Grouping', 'value': 'categorical'}
                        ],
                        value='continuous',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Core Analytical Goal:", className="control-label"),
                    dcc.Dropdown(
                        id="m4-wiz-goal",
                        options=[
                            {'label': 'Examine Distribution & Outliers', 'value': 'distribution'},
                            {'label': 'Uncover Correlation & Relationships', 'value': 'relationship'},
                            {'label': 'Compare Group Means & Medians', 'value': 'comparison'},
                            {'label': 'Part-to-Whole Composition', 'value': 'composition'}
                        ],
                        value='relationship',
                        clearable=False,
                        className="mb-3"
                    )
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.H4(id="m4-wiz-rec-title", className="text-info font-weight-bold mb-2"),
                        html.P(id="m4-wiz-rec-desc", className="text-light mb-3"),
                        dcc.Graph(id="m4-wiz-rec-graph", config={'displayModeBar': True, 'responsive': True})
                    ], className="stat-card")
                ])
            ], className="col-lg-8 mb-4")
        ], className="row")
    ])

def register_callbacks(app):
    @app.callback(
        Output("m4-content", "children"),
        Input("m4-subtabs", "value")
    )
    def render_m4_tab(tab):
        if tab == "tool-comparison":
            return comparison_layout()
        elif tab == "chart-gallery":
            return chart_gallery_layout()
        elif tab == "chart-wizard":
            return chart_wizard_layout()
        return html.Div()

    @app.callback(
        Output("m4-graph-gallery", "figure"),
        [Input("m4-gallery-choice", "value"),
         Input("m4-gallery-group", "value")]
    )
    def update_gallery(choice, group_col):
        df = load_data()
        if df.empty:
            df = pd.DataFrame({'Crop_Type': ['Wheat', 'Rice'], 'Crop_Yield_MT_per_HA': [3.0, 4.0]})
            
        group_col = group_col if group_col in df.columns else 'Adaptation_Strategies'
        fig = go.Figure()
        
        if choice == 'violin':
            fig = px.violin(
                df, x=group_col, y='Crop_Yield_MT_per_HA', color=group_col,
                box=True, points='outliers',
                title=f"Violin Plot with Embedded Boxplot: Crop Yield by {group_col.replace('_', ' ')}",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            
        elif choice == 'heatmap':
            num_cols = ['Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT', 
                        'Crop_Yield_MT_per_HA', 'Extreme_Weather_Events', 'Irrigation_Access_%', 
                        'Pesticide_Use_KG_per_HA', 'Fertilizer_Use_KG_per_HA', 'Soil_Health_Index', 
                        'Economic_Impact_Million_USD']
            corr = df[num_cols].corr()
            
            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Correlation Heatmap Matrix of 2024 Agronomic & Climatic Variables"
            )
            
        elif choice == '3d_scatter':
            sample_df = df.sample(min(800, len(df)), random_state=42)
            fig = px.scatter_3d(
                sample_df,
                x='Average_Temperature_C',
                y='Total_Precipitation_mm',
                z='Crop_Yield_MT_per_HA',
                color=group_col,
                size='Extreme_Weather_Events',
                title=f"3D Statistical Scatter: Temp vs. Precipitation vs. Yield (Color: {group_col})",
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor="#071912", gridcolor="rgba(52, 211, 153, 0.15)"),
                yaxis=dict(backgroundcolor="#071912", gridcolor="rgba(52, 211, 153, 0.15)"),
                zaxis=dict(backgroundcolor="#071912", gridcolor="rgba(52, 211, 153, 0.15)")
            ))
            
        elif choice == 'density_contour':
            fig = px.density_contour(
                df.sample(min(2000, len(df)), random_state=42),
                x='Average_Temperature_C', y='Crop_Yield_MT_per_HA',
                color='Adaptation_Strategies',
                marginal_x="histogram", marginal_y="histogram",
                title="2D Bivariate Density Contours: Temperature vs. Yield by Adaptation Strategy"
            )
            
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({'height': 480})
        fig.update_layout(layout_dict)
        return fig

    @app.callback(
        [Output("m4-wiz-rec-title", "children"),
         Output("m4-wiz-rec-desc", "children"),
         Output("m4-wiz-rec-graph", "figure")],
        [Input("m4-wiz-x", "value"),
         Input("m4-wiz-y", "value"),
         Input("m4-wiz-goal", "value")]
    )
    def update_wizard(x_type, y_type, goal):
        df = load_data()
        fig = go.Figure()
        
        if x_type == 'continuous' and y_type == 'none':
            title = "Recommended: Histogram with Kernel Density Overlay / Boxplot"
            desc = "Best for assessing skewness, modality, kurtosis, and identifying extreme outliers in continuous distributions."
            fig = px.histogram(df, x='Crop_Yield_MT_per_HA', nbins=30, marginal="rug", color_discrete_sequence=['#38bdf8'])
            
        elif x_type == 'continuous' and y_type == 'continuous':
            title = "Recommended: Scatter Plot with OLS Regression Trendline & Joint Marginals"
            desc = "Best for identifying linear or non-linear relationships, homoscedasticity, clustering, and correlation."
            fig = px.scatter(df.sample(min(500, len(df))), x='Total_Precipitation_mm', y='Crop_Yield_MT_per_HA', color='Adaptation_Strategies', trendline='ols')
            
        elif x_type == 'categorical' and y_type == 'continuous':
            title = "Recommended: Grouped Violin Plot or Boxplot with Error Bars"
            desc = "Best for comparing location (median/mean) and dispersion (IQR/variance) across distinct discrete categories."
            fig = px.box(df, x='Country', y='Crop_Yield_MT_per_HA', color='Country', points="outliers")
            
        elif x_type == 'time':
            title = "Recommended: Time-Series Line Plot with Rolling Average"
            desc = "Best for identifying secular trends, seasonality, cyclical shocks, and temporal autocorrelation."
            yearly = df.groupby('Year')[['Average_Temperature_C', 'Crop_Yield_MT_per_HA']].mean().reset_index()
            fig = px.line(yearly, x='Year', y='Average_Temperature_C', markers=True, line_shape='spline')
            
        else:
            title = "Recommended: Stacked or Grouped Bar Chart"
            desc = "Best for categorical cross-tabulations and comparing frequencies across subgroups."
            ct = pd.crosstab(df['Crop_Type'], df['Adaptation_Strategies']).reset_index()
            fig = px.bar(ct, x='Crop_Type', y=df['Adaptation_Strategies'].unique(), title="Adaptation Strategy Distribution by Crop")
            
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({'height': 400})
        fig.update_layout(layout_dict)
        return title, desc, fig
