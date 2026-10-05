"""
Module 1: Descriptive Statistics
- Measures of Central Tendency (Mean, Median, Mode)
- Measures of Variability (Range, IQR, Variance, Standard Deviation, Empirical Rule)
- Graphical Summaries (Bar Chart, Histogram with KDE, Donut Chart, Scatter with Marginals)
Uses: climate_change_impact_on_agriculture_2024.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, callback

import plotly.io as pio
pio.templates.default = "plotly_white"

PLOT_LAYOUT = {
    'template': 'plotly_white',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'family': 'Plus Jakarta Sans, sans-serif'},
    'xaxis': {'gridcolor': 'rgba(128,128,128,0.18)', 'zerolinecolor': 'rgba(128,128,128,0.3)'},
    'yaxis': {'gridcolor': 'rgba(128,128,128,0.18)', 'zerolinecolor': 'rgba(128,128,128,0.3)'},
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
            html.Span("Phase 1 — Agro-Climatic Baseline", className="section-badge"),
            html.H2("Crop Yield Baselines & Climate Variability Profiling", className="section-title"),
            html.P("Analyze global crop yield benchmarks, evaluate temperature and precipitation dispersion, and uncover regional farm distribution patterns.", className="section-subtitle")
        ]),
        
        dcc.Tabs(id="m1-subtabs", value="central-tendency", children=[
            dcc.Tab(label="1. Benchmark Yields & Central Trends", value="central-tendency"),
            dcc.Tab(label="2. Climate Dispersion & Anomaly Bands", value="variability"),
            dcc.Tab(label="3. Global Farm Distributions & Explorer", value="graphical-summaries")
        ], className="mb-4"),
        
        html.Div(id="m1-content")
    ])

def central_tendency_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Distribution Generator & Outlier Controls", className="control-label"),
                    html.P("Select a theoretical distribution shape and inject outliers to observe Mean vs. Median vs. Mode behavior.", className="control-desc"),
                    
                    html.Label("Distribution Shape:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m1-dist-type",
                        options=[
                            {'label': 'Symmetric (Normal Distribution)', 'value': 'normal'},
                            {'label': 'Right-Skewed (Positive Skew - e.g., Economic Loss)', 'value': 'right_skew'},
                            {'label': 'Left-Skewed (Negative Skew - e.g., Soil Health Index)', 'value': 'left_skew'},
                            {'label': 'Bimodal (Two Distinct Yield Clusters)', 'value': 'bimodal'},
                            {'label': 'Uniform Distribution (Even Spread)', 'value': 'uniform'}
                        ],
                        value='right_skew',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Sample Size (N):", className="control-label"),
                    dcc.Slider(
                        id="m1-sample-size",
                        min=50, max=1000, step=50, value=300,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Inject Outlier Magnitude:", className="control-label"),
                    dcc.Slider(
                        id="m1-outlier-val",
                        min=0, max=100, step=10, value=0,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Div([
                        html.H6("Guidance & Best Practices:", className="text-info mt-3 font-weight-bold"),
                        html.Ul([
                            html.Li([html.B("Mean: "), "Sensitive to extreme outliers. Best for symmetric, unskewed continuous data."]),
                            html.Li([html.B("Median: "), "Robust to outliers. Ideal for skewed distributions (economic damages, precipitation extremes)."]),
                            html.Li([html.B("Mode: "), "Most frequent value. Crucial for categorical and discrete modal patterns."])
                        ], className="small text-muted pl-3")
                    ], className="mt-3 p-3 bg-dark rounded border border-secondary")
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Arithmetic Mean (x̄)", className="stat-card-header"),
                            html.Div(id="m1-val-mean", className="stat-card-value text-primary"),
                            html.Div("Sum of values / N", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Median (50th %ile)", className="stat-card-header"),
                            html.Div(id="m1-val-median", className="stat-card-value text-success"),
                            html.Div("Middle value (robust)", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Estimated Mode", className="stat-card-header"),
                            html.Div(id="m1-val-mode", className="stat-card-value text-warning"),
                            html.Div("Peak density value", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m1-graph-central", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def variability_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Dispersion & Spread Controls", className="control-label"),
                    html.P("Explore how data spreads around the mean, standard deviation bands, and the Empirical 68-95-99.7% Rule.", className="control-desc"),
                    
                    html.Label("Target Mean (μ):", className="control-label mt-2"),
                    dcc.Slider(
                        id="m1-var-mean",
                        min=20, max=100, step=5, value=50,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Standard Deviation (σ):", className="control-label"),
                    dcc.Slider(
                        id="m1-var-std",
                        min=2, max=25, step=1, value=10,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Highlight Empirical Rule Bands:", className="control-label"),
                    dcc.RadioItems(
                        id="m1-empirical-band",
                        options=[
                            {'label': ' All Bands (68%, 95%, 99.7%)', 'value': 'all'},
                            {'label': ' 1σ (68.27% of Data)', 'value': '1sd'},
                            {'label': ' 2σ (95.45% of Data)', 'value': '2sd'},
                            {'label': ' 3σ (99.73% of Data)', 'value': '3sd'}
                        ],
                        value='all',
                        inline=False,
                        className="mb-3 custom-radio-group",
                        labelClassName="custom-radio-label",
                        inputClassName="custom-radio-input me-2"
                    ),
                    
                    html.Div([
                        html.H6("Variability Formulas:", className="text-info font-weight-bold"),
                        html.Div("Range = Max - Min", className="formula-badge d-block mb-1"),
                        html.Div("IQR = Q3 - Q1", className="formula-badge d-block mb-1"),
                        html.Div("Variance s² = Σ(x - x̄)² / (n - 1)", className="formula-badge d-block mb-1"),
                        html.Div("Std Dev s = √(s²)", className="formula-badge d-block mb-1"),
                        html.Div("CV = (s / x̄) × 100%", className="formula-badge d-block")
                    ], className="p-3 bg-dark rounded border border-secondary mt-3")
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Range & IQR", className="stat-card-header"),
                            html.Div(id="m1-disp-range-iqr", className="stat-card-value text-info", style={'fontSize': '1.35rem'}),
                            html.Div("Total Spread & Middle 50%", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Variance (s²)", className="stat-card-header"),
                            html.Div(id="m1-disp-variance", className="stat-card-value text-warning", style={'fontSize': '1.35rem'}),
                            html.Div("Average squared deviations", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Coeff. of Variation (CV)", className="stat-card-header"),
                            html.Div(id="m1-disp-cv", className="stat-card-value text-danger", style={'fontSize': '1.35rem'}),
                            html.Div("Relative Dispersion (s/μ)", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m1-graph-variability", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def graphical_summaries_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Visual Summary Explorer (Real Dataset)", className="control-label"),
                    html.P("Explore patterns across 10,000 agricultural observations from climate_change_impact_on_agriculture_2024.csv.", className="control-desc"),
                    
                    html.Label("Chart Type:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m1-chart-type",
                        options=[
                            {'label': 'Histogram with Density Curve (KDE)', 'value': 'histogram'},
                            {'label': 'Bar Chart (Mean Crop Yields by Crop)', 'value': 'bar'},
                            {'label': 'Donut / Pie Chart (Country Composition)', 'value': 'pie'},
                            {'label': 'Scatter Plot with Marginal Boxplots', 'value': 'scatter'}
                        ],
                        value='histogram',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    # Persistent container for Histogram controls (always in DOM)
                    html.Div(id="m1-hist-controls", children=[
                        html.Label("Histogram Bin Count:", className="control-label"),
                        dcc.Slider(id="m1-hist-bins", min=10, max=60, step=5, value=25, marks=None),
                        html.Label("Variable to Plot:", className="control-label mt-2"),
                        dcc.Dropdown(
                            id="m1-hist-var",
                            options=[
                                {'label': 'Average Temperature (°C)', 'value': 'Average_Temperature_C'},
                                {'label': 'Total Precipitation (mm)', 'value': 'Total_Precipitation_mm'},
                                {'label': 'Crop Yield (MT/HA)', 'value': 'Crop_Yield_MT_per_HA'},
                                {'label': 'Extreme Weather Events', 'value': 'Extreme_Weather_Events'},
                                {'label': 'Soil Health Index', 'value': 'Soil_Health_Index'},
                                {'label': 'Economic Impact (Million USD)', 'value': 'Economic_Impact_Million_USD'}
                            ],
                            value='Average_Temperature_C',
                            clearable=False
                        )
                    ], style={'display': 'block'}),
                    
                    # Persistent container for Scatter controls (always in DOM)
                    html.Div(id="m1-scatter-controls", children=[
                        html.Label("X-Axis Feature:", className="control-label"),
                        dcc.Dropdown(
                            id="m1-scatter-x",
                            options=[
                                {'label': 'Average Temperature (°C)', 'value': 'Average_Temperature_C'},
                                {'label': 'Total Precipitation (mm)', 'value': 'Total_Precipitation_mm'},
                                {'label': 'Fertilizer Use (KG/HA)', 'value': 'Fertilizer_Use_KG_per_HA'},
                                {'label': 'Soil Health Index', 'value': 'Soil_Health_Index'},
                                {'label': 'CO2 Emissions (MT)', 'value': 'CO2_Emissions_MT'}
                            ],
                            value='Average_Temperature_C',
                            clearable=False,
                            className="mb-2"
                        ),
                        html.Label("Y-Axis Feature:", className="control-label"),
                        dcc.Dropdown(
                            id="m1-scatter-y",
                            options=[
                                {'label': 'Crop Yield (MT/HA)', 'value': 'Crop_Yield_MT_per_HA'},
                                {'label': 'Economic Impact (Million USD)', 'value': 'Economic_Impact_Million_USD'}
                            ],
                            value='Crop_Yield_MT_per_HA',
                            clearable=False
                        )
                    ], style={'display': 'none'}),
                    
                    html.Div([
                        html.H6("Data Visualization Principle:", className="text-info font-weight-bold"),
                        html.P("Histograms display continuous frequencies; Bar charts compare discrete categories; Scatter plots uncover bivariate associations.", className="small text-muted mb-0")
                    ], className="p-3 bg-dark rounded border border-secondary mt-3")
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="m1-graph-summaries", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def register_callbacks(app):
    @app.callback(
        Output("m1-content", "children"),
        Input("m1-subtabs", "value")
    )
    def render_subtab(tab):
        if tab == "central-tendency":
            return central_tendency_layout()
        elif tab == "variability":
            return variability_layout()
        elif tab == "graphical-summaries":
            return graphical_summaries_layout()
        return html.Div("Select a tab")

    @app.callback(
        [Output("m1-val-mean", "children"),
         Output("m1-val-median", "children"),
         Output("m1-val-mode", "children"),
         Output("m1-graph-central", "figure")],
        [Input("m1-dist-type", "value"),
         Input("m1-sample-size", "value"),
         Input("m1-outlier-val", "value")]
    )
    def update_central_tendency(dist_type, n, outlier_val):
        np.random.seed(42)
        if dist_type == 'normal':
            data = np.random.normal(50, 10, n)
        elif dist_type == 'right_skew':
            data = np.random.exponential(scale=12, size=n) + 20
        elif dist_type == 'left_skew':
            data = 100 - (np.random.exponential(scale=12, size=n) + 20)
        elif dist_type == 'bimodal':
            d1 = np.random.normal(35, 6, int(n * 0.5))
            d2 = np.random.normal(70, 7, n - int(n * 0.5))
            data = np.concatenate([d1, d2])
        else: # uniform
            data = np.random.uniform(20, 80, n)
            
        if outlier_val and outlier_val > 0:
            num_outliers = max(2, int(n * 0.03))
            outliers = np.full(num_outliers, np.max(data) + outlier_val)
            data = np.concatenate([data, outliers])
            
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        
        # Kernel density estimation to find peak mode
        kde = stats.gaussian_kde(data)
        x_eval = np.linspace(min(data), max(data), 1000)
        y_kde = kde(x_eval)
        mode_val = float(x_eval[np.argmax(y_kde)])
        
        # Build Figure
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            histnorm='probability density',
            name='Data Distribution',
            nbinsx=35,
            marker=dict(color='rgba(56, 189, 248, 0.45)', line=dict(color='#38bdf8', width=1.5)),
            opacity=0.75
        ))
        
        fig.add_trace(go.Scatter(
            x=x_eval, y=y_kde,
            mode='lines',
            name='Kernel Density (KDE)',
            line=dict(color='#06b6d4', width=2.5)
        ))
        
        max_y = max(y_kde) * 1.15
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val], y=[0, max_y],
            mode='lines+text',
            name=f'Mean = {mean_val:.2f}',
            line=dict(color='#3b82f6', width=3, dash='solid'),
            text=['', f' Mean ({mean_val:.1f})'],
            textposition='top right'
        ))
        
        fig.add_trace(go.Scatter(
            x=[median_val, median_val], y=[0, max_y * 0.9],
            mode='lines+text',
            name=f'Median = {median_val:.2f}',
            line=dict(color='#10b981', width=3, dash='dash'),
            text=['', f' Median ({median_val:.1f})'],
            textposition='top right'
        ))
        
        fig.add_trace(go.Scatter(
            x=[mode_val, mode_val], y=[0, max_y * 0.8],
            mode='lines+text',
            name=f'Mode = {mode_val:.2f}',
            line=dict(color='#f59e0b', width=3, dash='dot'),
            text=['', f' Mode ({mode_val:.1f})'],
            textposition='top left'
        ))
        
        skewness = stats.skew(data)
        skew_label = "Symmetric" if abs(skewness) < 0.2 else ("Positively Skewed (Right Tail)" if skewness > 0 else "Negatively Skewed (Left Tail)")
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Central Tendency Comparison</b> (Skewness: {skewness:.2f} — {skew_label})',
            'xaxis_title': 'Observed Values',
            'yaxis_title': 'Probability Density',
            'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.1},
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"{mean_val:.2f}", f"{median_val:.2f}", f"{mode_val:.2f}", fig

    @app.callback(
        [Output("m1-disp-range-iqr", "children"),
         Output("m1-disp-variance", "children"),
         Output("m1-disp-cv", "children"),
         Output("m1-graph-variability", "figure")],
        [Input("m1-var-mean", "value"),
         Input("m1-var-std", "value"),
         Input("m1-empirical-band", "value")]
    )
    def update_variability(mu, sigma, band_type):
        np.random.seed(42)
        data = np.random.normal(mu, sigma, 1500)
        
        val_range = np.ptp(data)
        q75, q25 = np.percentile(data, [75 ,25])
        val_iqr = q75 - q25
        val_var = np.var(data, ddof=1)
        val_cv = (sigma / mu) * 100
        
        x_axis = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        y_pdf = stats.norm.pdf(x_axis, mu, sigma)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x_axis, y=y_pdf,
            mode='lines',
            name='Gaussian Distribution',
            line=dict(color='#38bdf8', width=2.5)
        ))
        
        def add_band(k, color, label):
            x_band = np.linspace(mu - k*sigma, mu + k*sigma, 200)
            y_band = stats.norm.pdf(x_band, mu, sigma)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_band, x_band[::-1]]),
                y=np.concatenate([y_band, np.zeros_like(y_band)]),
                fill='toself',
                fillcolor=color,
                line=dict(color='rgba(255,255,255,0)'),
                name=label,
                hoverinfo='text',
                text=f'Interval: [{mu - k*sigma:.1f}, {mu + k*sigma:.1f}]'
            ))
            
        if band_type in ['all', '3sd']:
            add_band(3, 'rgba(239, 68, 68, 0.25)', '±3σ (99.73%)')
        if band_type in ['all', '2sd']:
            add_band(2, 'rgba(245, 158, 11, 0.35)', '±2σ (95.45%)')
        if band_type in ['all', '1sd']:
            add_band(1, 'rgba(16, 185, 129, 0.45)', '±1σ (68.27%)')
            
        fig.add_vline(x=mu, line_width=2, line_dash="dash", line_color="#38bdf8", annotation_text=f"μ = {mu}")
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Empirical Rule & Dispersion Bounds</b> (μ = {mu}, σ = {sigma})',
            'xaxis_title': 'Value',
            'yaxis_title': 'Density',
            'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.05},
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"{val_range:.1f} / {val_iqr:.1f}", f"{val_var:.2f}", f"{val_cv:.1f}%", fig

    # Toggle options visibility without removing elements from DOM
    @app.callback(
        [Output("m1-hist-controls", "style"),
         Output("m1-scatter-controls", "style")],
        Input("m1-chart-type", "value")
    )
    def toggle_chart_controls(chart_type):
        if chart_type == 'histogram':
            return {'display': 'block'}, {'display': 'none'}
        elif chart_type == 'scatter':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'none'}, {'display': 'none'}

    @app.callback(
        Output("m1-graph-summaries", "figure"),
        [Input("m1-chart-type", "value"),
         Input("m1-hist-bins", "value"),
         Input("m1-hist-var", "value"),
         Input("m1-scatter-x", "value"),
         Input("m1-scatter-y", "value")]
    )
    def update_graphical_summaries(chart_type, bins, hist_var, scatter_x, scatter_y):
        df = load_data()
        if df.empty:
            df = pd.DataFrame({'Crop_Type': ['Wheat', 'Corn'], 'Crop_Yield_MT_per_HA': [3.5, 4.2]})
            
        fig = go.Figure()
        
        if chart_type == 'histogram':
            var_name = hist_var if (hist_var and hist_var in df.columns) else 'Crop_Yield_MT_per_HA'
            bin_count = bins or 25
            
            fig = px.histogram(
                df, x=var_name, nbins=bin_count, marginal="box",
                color_discrete_sequence=['#06b6d4'],
                title=f"Histogram with Marginal Boxplot: {var_name}"
            )
            fig.update_traces(marker_line_color='#0284c7', marker_line_width=1)
            
        elif chart_type == 'bar':
            agg = df.groupby('Crop_Type')['Crop_Yield_MT_per_HA'].agg(['mean', 'std', 'count']).reset_index()
            fig = px.bar(
                agg, x='Crop_Type', y='mean', error_y='std',
                color='Crop_Type',
                color_discrete_sequence=px.colors.qualitative.Prism,
                title="Bar Chart: Mean Crop Yields by Crop Type (with ±1 SD Error Bars)",
                labels={'mean': 'Mean Yield (MT/HA)', 'Crop_Type': 'Crop Type'}
            )
            
        elif chart_type == 'pie':
            counts = df['Country'].value_counts().reset_index()
            counts.columns = ['Country', 'Count']
            fig = px.pie(
                counts, values='Count', names='Country',
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Donut Chart: Geographical Distribution of Agricultural Observations"
            )
            
        elif chart_type == 'scatter':
            x_col = scatter_x if (scatter_x and scatter_x in df.columns) else 'Average_Temperature_C'
            y_col = scatter_y if (scatter_y and scatter_y in df.columns) else 'Crop_Yield_MT_per_HA'
            fig = px.scatter(
                df.sample(min(1000, len(df))), x=x_col, y=y_col, color='Crop_Type',
                marginal_x="histogram", marginal_y="violin",
                trendline="ols",
                title=f"Scatter Plot & Marginal Distributions: {x_col} vs. {y_col}",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({'height': 480})
        fig.update_layout(layout_dict)
        return fig
