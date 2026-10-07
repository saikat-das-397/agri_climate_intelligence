"""
Module 3: Inferential Statistics
- Sampling & Central Limit Theorem (CLT) Simulation
- Estimation & Confidence Intervals Lab (100 Sample Intervals Visualizer)
- Hypothesis Testing Suite (t-test, z-test, Chi-square test, p-values, Type I & II errors)
"""
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

def layout():
    return html.Div([
        html.Div([
            html.Span("Phase 3 — Inferential Agro-Analytics", className="section-badge"),
            html.H2("Climate Impact Estimation & Agronomic Hypothesis Lab", className="section-title"),
            html.P("Simulate multi-farm sampling under the Central Limit Theorem, construct yield benchmark confidence intervals, and test the statistical significance of climate anomalies.", className="section-subtitle")
        ]),
        
        html.Div([
            dcc.Tabs(
                id="m3-subtabs",
                value="clt-simulator",
                parent_className="subtabs-nav-container",
                className="subtabs-root",
                mobile_breakpoint=0,
                children=[
                    dcc.Tab(
                        label="1. Regional Farm Sampling (CLT)",
                        value="clt-simulator",
                        className="custom-subtab",
                        selected_className="custom-subtab--selected"
                    ),
                    dcc.Tab(
                        label="2. Yield Benchmark Confidence Intervals",
                        value="confidence-intervals",
                        className="custom-subtab",
                        selected_className="custom-subtab--selected"
                    ),
                    dcc.Tab(
                        label="3. Climate Shift Significance Testing",
                        value="hypothesis-testing",
                        className="custom-subtab",
                        selected_className="custom-subtab--selected"
                    ),
                    dcc.Tab(
                        label="4. Adaptation Independence & Risk Analysis",
                        value="chisq-power",
                        className="custom-subtab",
                        selected_className="custom-subtab--selected"
                    )
                ]
            )
        ], className="subtabs-bar-wrapper mb-4"),
        
        html.Div(id="m3-content")
    ])

def clt_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("CLT Simulation Parameters", className="control-label"),
                    html.P("The Central Limit Theorem guarantees that the sampling distribution of the mean approaches Gaussian regardless of parent distribution shape.", className="control-desc"),
                    
                    html.Label("Parent Population Shape:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m3-clt-parent",
                        options=[
                            {'label': 'Highly Skewed Exponential (λ=0.2)', 'value': 'exponential'},
                            {'label': 'Bimodal Mixture (Two distinct sub-populations)', 'value': 'bimodal'},
                            {'label': 'Uniform Distribution (Flat)', 'value': 'uniform'},
                            {'label': 'Standard Normal (Gaussian)', 'value': 'normal'}
                        ],
                        value='exponential',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Sample Size per Draw (n):", className="control-label"),
                    dcc.Slider(
                        id="m3-clt-n",
                        min=2, max=100, step=2, value=30,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Number of Samples Drawn (k):", className="control-label"),
                    dcc.Slider(
                        id="m3-clt-k",
                        min=100, max=2000, step=100, value=1000,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Div([
                        html.H6("CLT Mathematical Formulation:", className="text-info font-weight-bold"),
                        html.Div("X̄ ~ N( μ , σ / √n )", className="formula-badge d-block mb-1"),
                        html.Div("Standard Error SE = σ / √n", className="formula-badge d-block")
                    ], className="p-3 bg-dark rounded border border-secondary mt-3")
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Parent Population μ", className="stat-card-header"),
                            html.Div(id="m3-clt-pop-mean", className="stat-card-value text-muted"),
                            html.Div("True Mean of Parent", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Mean of Sample Means", className="stat-card-header"),
                            html.Div(id="m3-clt-samp-mean", className="stat-card-value text-primary"),
                            html.Div("E[X̄] Unbiased Estimator", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Standard Error (SE)", className="stat-card-header"),
                            html.Div(id="m3-clt-se", className="stat-card-value text-success"),
                            html.Div("Observed vs Theoretical σ/√n", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m3-graph-clt", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def confidence_intervals_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Confidence Intervals Simulation", className="control-label"),
                    html.P("Simulate 100 independent random samples from a population with true mean μ = 50. See how many 100(1-α)% CIs successfully capture μ.", className="control-desc"),
                    
                    html.Label("Confidence Level (1 - α):", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m3-ci-level",
                        options=[
                            {'label': '90% Confidence Level (z* = 1.645)', 'value': 0.90},
                            {'label': '95% Confidence Level (z* = 1.960)', 'value': 0.95},
                            {'label': '99% Confidence Level (z* = 2.576)', 'value': 0.99}
                        ],
                        value=0.95,
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Sample Size per Experiment (n):", className="control-label"),
                    dcc.Slider(
                        id="m3-ci-n",
                        min=15, max=150, step=15, value=45,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Button("Generate New 100 Samples", id="m3-ci-btn-resample", className="btn btn-primary-glow btn-block w-100 mb-3"),
                    
                    html.Div([
                        html.H6("Formula & Interpretation:", className="text-info font-weight-bold"),
                        html.Div("CI = x̄ ± z* × (s / √n)", className="formula-badge d-block mb-1"),
                        html.P("A 95% CI does not mean there is a 95% chance μ is inside this specific interval; it means 95% of all similarly constructed intervals across repeated samplings will capture μ.", className="small text-muted mb-0")
                    ], className="p-3 bg-dark rounded border border-secondary")
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Captured Population μ", className="stat-card-header"),
                            html.Div(id="m3-ci-stat-capture", className="stat-card-value text-success"),
                            html.Div("Success Rate out of 100 CIs", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Missed Population μ", className="stat-card-header"),
                            html.Div(id="m3-ci-stat-missed", className="stat-card-value text-danger"),
                            html.Div("Intervals outside true mean", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m3-graph-ci", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def hypothesis_testing_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Hypothesis Testing Lab (t-test / z-test)", className="control-label"),
                    html.P("Test whether climate change has significantly altered agricultural crop yield benchmarks.", className="control-desc"),
                    
                    html.Label("Test Type:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m3-ht-test-type",
                        options=[
                            {'label': 'One-Sample t-test (Compare Sample Mean vs Null μ₀)', 'value': 'one_sample_t'},
                            {'label': 'Two-Sample Independent t-test (Compare Two Groups)', 'value': 'two_sample_t'},
                            {'label': 'Two-Tailed Z-Test for Proportions', 'value': 'z_prop'}
                        ],
                        value='one_sample_t',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Significance Level (α):", className="control-label"),
                    dcc.Dropdown(
                        id="m3-ht-alpha",
                        options=[
                            {'label': 'α = 0.01 (99% Confidence)', 'value': 0.01},
                            {'label': 'α = 0.05 (95% Confidence - Standard)', 'value': 0.05},
                            {'label': 'α = 0.10 (90% Confidence)', 'value': 0.10}
                        ],
                        value=0.05,
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Hypothesized Mean / Benchmark (μ₀):", className="control-label"),
                    dcc.Slider(
                        id="m3-ht-mu0",
                        min=3.0, max=6.0, step=0.1, value=4.5,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Observed Sample Mean (x̄):", className="control-label"),
                    dcc.Slider(
                        id="m3-ht-xbar",
                        min=3.0, max=6.0, step=0.1, value=4.9,
                        marks=None,
                        className="mb-3"
                    ),
                    
                    html.Label("Sample Size (n):", className="control-label"),
                    dcc.Slider(
                        id="m3-ht-n",
                        min=10, max=100, step=5, value=35,
                        marks=None,
                        className="mb-3"
                    ),
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Test Statistic (t / z)", className="stat-card-header"),
                            html.Div(id="m3-ht-stat-val", className="stat-card-value text-warning"),
                            html.Div("Calculated Score", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("p-value", className="stat-card-header"),
                            html.Div(id="m3-ht-pval", className="stat-card-value text-info"),
                            html.Div("P(Extreme | H₀ true)", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Statistical Decision", className="stat-card-header"),
                            html.Div(id="m3-ht-decision", className="stat-card-value", style={'fontSize': '1.15rem'}),
                            html.Div("Alpha Threshold Evaluation", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 col-md-4 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m3-graph-ht", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def chisq_power_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Chi-Square Test of Independence", className="control-label"),
                    html.P("Examine association between Climate Risk Level and Crop Yield Outcomes (High vs Low Yield).", className="control-desc"),
                    
                    html.Label("Contingency Table Counts (Low Risk - High Yield):", className="control-label mt-2"),
                    dcc.Slider(id="m3-chi-c1", min=20, max=200, step=10, value=140, marks=None),
                    
                    html.Label("Low Risk - Low Yield:", className="control-label mt-2"),
                    dcc.Slider(id="m3-chi-c2", min=10, max=150, step=10, value=40, marks=None),
                    
                    html.Label("High Climate Risk - High Yield:", className="control-label mt-2"),
                    dcc.Slider(id="m3-chi-c3", min=10, max=150, step=10, value=50, marks=None),
                    
                    html.Label("High Climate Risk - Low Yield:", className="control-label mt-2"),
                    dcc.Slider(id="m3-chi-c4", min=20, max=200, step=10, value=130, marks=None),
                ], className="stat-card h-100")
            ], className="col-12 col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Chi-Square Statistic (χ²)", className="stat-card-header"),
                            html.Div(id="m3-chi-stat", className="stat-card-value text-warning"),
                            html.Div("Σ (O - E)² / E", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Chi-Square p-value", className="stat-card-header"),
                            html.Div(id="m3-chi-pval", className="stat-card-value text-info"),
                            html.Div("Degrees of Freedom df=1", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-12 col-sm-6 mb-3"),
                ], className="row g-2 g-md-3"),
                
                html.Div([
                    dcc.Graph(id="m3-graph-chisq", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-12 col-lg-8 mb-4")
        ], className="row g-3 g-md-4")
    ])

def register_callbacks(app):
    @app.callback(
        Output("m3-content", "children"),
        Input("m3-subtabs", "value")
    )
    def render_m3_tab(tab):
        if tab == "clt-simulator":
            return clt_layout()
        elif tab == "confidence-intervals":
            return confidence_intervals_layout()
        elif tab == "hypothesis-testing":
            return hypothesis_testing_layout()
        elif tab == "chisq-power":
            return chisq_power_layout()
        return html.Div()

    @app.callback(
        [Output("m3-clt-pop-mean", "children"),
         Output("m3-clt-samp-mean", "children"),
         Output("m3-clt-se", "children"),
         Output("m3-graph-clt", "figure")],
        [Input("m3-clt-parent", "value"),
         Input("m3-clt-n", "value"),
         Input("m3-clt-k", "value")]
    )
    def update_clt(parent, n, k):
        np.random.seed(42)
        n = n or 30
        k = k or 1000
        
        # Generate parent population
        pop_size = 50000
        if parent == 'exponential':
            pop = np.random.exponential(scale=5.0, size=pop_size)
            pop_name = "Exponential (Skewed)"
        elif parent == 'bimodal':
            d1 = np.random.normal(15, 3, int(pop_size * 0.5))
            d2 = np.random.normal(35, 4, pop_size - int(pop_size * 0.5))
            pop = np.concatenate([d1, d2])
            pop_name = "Bimodal Gaussian Mixture"
        elif parent == 'uniform':
            pop = np.random.uniform(10, 50, pop_size)
            pop_name = "Uniform Distribution"
        else: # normal
            pop = np.random.normal(30, 8, pop_size)
            pop_name = "Normal Distribution"
            
        pop_mean = float(np.mean(pop))
        pop_std = float(np.std(pop, ddof=0))
        theoretical_se = pop_std / np.sqrt(n)
        
        # Draw k samples of size n and compute sample means
        sample_matrix = np.random.choice(pop, size=(k, n), replace=True)
        sample_means = np.mean(sample_matrix, axis=1)
        mean_of_means = float(np.mean(sample_means))
        observed_se = float(np.std(sample_means, ddof=1))
        
        # Plotting side by side / overlapping
        fig = go.Figure()
        
        # Histogram of sample means
        fig.add_trace(go.Histogram(
            x=sample_means,
            histnorm='probability density',
            name=f'Sampling Dist of X̄ (k={k}, n={n})',
            marker=dict(color='rgba(16, 185, 129, 0.65)', line=dict(color='#10b981', width=1.5)),
            nbinsx=40
        ))
        
        # Overlay Theoretical Normal curve
        x_norm = np.linspace(pop_mean - 3.8*theoretical_se, pop_mean + 3.8*theoretical_se, 300)
        y_norm = stats.norm.pdf(x_norm, pop_mean, theoretical_se)
        fig.add_trace(go.Scatter(
            x=x_norm, y=y_norm,
            mode='lines',
            line=dict(color='#f43f5e', width=2.5),
            name=f'Theoretical N(μ={pop_mean:.2f}, SE={theoretical_se:.2f})'
        ))
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>CLT Sampling Distribution</b> (Parent: {pop_name}, Sample Size n={n})',
            'xaxis_title': 'Sample Mean (X̄)',
            'yaxis_title': 'Probability Density',
            'legend': {'orientation': 'h', 'y': -0.2, 'x': 0.05},
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"{pop_mean:.2f}", f"{mean_of_means:.2f}", f"{observed_se:.2f} (Theo: {theoretical_se:.2f})", fig

    @app.callback(
        [Output("m3-ci-stat-capture", "children"),
         Output("m3-ci-stat-missed", "children"),
         Output("m3-graph-ci", "figure")],
        [Input("m3-ci-level", "value"),
         Input("m3-ci-n", "value"),
         Input("m3-ci-btn-resample", "n_clicks")]
    )
    def update_ci(ci_level, n, n_clicks):
        # We can seed based on n_clicks
        seed = 42 + (n_clicks or 0)
        np.random.seed(seed)
        
        true_mu = 50.0
        pop_sigma = 12.0
        n = n or 45
        ci_level = ci_level or 0.95
        
        z_crit = stats.norm.ppf(1 - (1 - ci_level) / 2)
        
        k = 100
        samples = np.random.normal(true_mu, pop_sigma, size=(k, n))
        sample_means = np.mean(samples, axis=1)
        sample_stds = np.std(samples, axis=1, ddof=1)
        margins = z_crit * (sample_stds / np.sqrt(n))
        
        ci_lower = sample_means - margins
        ci_upper = sample_means + margins
        
        captured = (ci_lower <= true_mu) & (true_mu <= ci_upper)
        num_captured = int(np.sum(captured))
        num_missed = k - num_captured
        
        fig = go.Figure()
        
        # True mean line
        fig.add_vline(x=true_mu, line_width=3, line_color="#38bdf8", line_dash="solid", annotation_text=f"True μ = {true_mu:.1f}")
        
        for i in range(k):
            color = '#10b981' if captured[i] else '#ef4444'
            fig.add_trace(go.Scatter(
                x=[ci_lower[i], sample_means[i], ci_upper[i]],
                y=[i + 1, i + 1, i + 1],
                mode='lines+markers',
                marker=dict(size=[0, 4, 0], color=color),
                line=dict(color=color, width=1.5),
                hoverinfo='text',
                text=f"Sample #{i+1}: Mean={sample_means[i]:.2f}, CI=[{ci_lower[i]:.2f}, {ci_upper[i]:.2f}] ({'Captured' if captured[i] else 'MISSED'})",
                showlegend=False
            ))
            
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>100 Simulated {int(ci_level*100)}% Confidence Intervals</b> ({num_captured}/100 Captured True μ)',
            'xaxis_title': 'Observed Interval Estimate',
            'yaxis_title': 'Sample Replication #',
            'height': 480
        })
        fig.update_layout(layout_dict)
        
        return f"{num_captured}%", f"{num_missed}%", fig

    @app.callback(
        [Output("m3-ht-stat-val", "children"),
         Output("m3-ht-pval", "children"),
         Output("m3-ht-decision", "children"),
         Output("m3-ht-decision", "className"),
         Output("m3-graph-ht", "figure")],
        [Input("m3-ht-test-type", "value"),
         Input("m3-ht-alpha", "value"),
         Input("m3-ht-mu0", "value"),
         Input("m3-ht-xbar", "value"),
         Input("m3-ht-n", "value")]
    )
    def update_hypothesis_test(test_type, alpha, mu0, xbar, n):
        alpha = alpha or 0.05
        mu0 = mu0 or 4.5
        xbar = xbar or 4.9
        n = n or 35
        s = 1.1 # estimated standard deviation
        
        se = s / np.sqrt(n)
        t_stat = (xbar - mu0) / se
        df = n - 1
        
        # Two-tailed p-value
        pval = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
        
        decision = "Reject H₀ (Statistically Significant)" if pval < alpha else "Fail to Reject H₀ (Inconclusive)"
        decision_class = "stat-card-value text-danger" if pval < alpha else "stat-card-value text-muted"
        
        # Build curve
        x_vals = np.linspace(-4, 4, 500)
        y_vals = stats.t.pdf(x_vals, df=df)
        crit_t = stats.t.ppf(1 - alpha/2, df=df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color='#38bdf8', width=2), name='Null Distribution (H₀)'))
        
        # Shaded Rejection Regions
        x_left = np.linspace(-4, -crit_t, 100)
        y_left = stats.t.pdf(x_left, df=df)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_left, [-crit_t, -4]]),
            y=np.concatenate([y_left, [0, 0]]),
            fill='toself', fillcolor='rgba(239, 68, 68, 0.45)', line=dict(color='rgba(255,255,255,0)'),
            name=f'Rejection Region (α={alpha})'
        ))
        
        x_right = np.linspace(crit_t, 4, 100)
        y_right = stats.t.pdf(x_right, df=df)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_right, [4, crit_t]]),
            y=np.concatenate([y_right, [0, 0]]),
            fill='toself', fillcolor='rgba(239, 68, 68, 0.45)', line=dict(color='rgba(255,255,255,0)'),
            showlegend=False
        ))
        
        # Test stat line
        fig.add_vline(x=t_stat, line_width=3, line_color="#38bdf8", line_dash="solid", annotation_text=f"t_calc = {t_stat:.2f}")
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Null Distribution vs. Observed Test Statistic</b> (Critical t = ±{crit_t:.2f})',
            'xaxis_title': 't-score',
            'yaxis_title': 'Density',
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"t = {t_stat:.3f}", f"{pval:.4f}", decision, decision_class, fig

    @app.callback(
        [Output("m3-chi-stat", "children"),
         Output("m3-chi-pval", "children"),
         Output("m3-graph-chisq", "figure")],
        [Input("m3-chi-c1", "value"),
         Input("m3-chi-c2", "value"),
         Input("m3-chi-c3", "value"),
         Input("m3-chi-c4", "value")]
    )
    def update_chisq(c1, c2, c3, c4):
        obs = np.array([[c1 or 140, c2 or 40], [c3 or 50, c4 or 130]])
        chi2, p, dof, ex = stats.chi2_contingency(obs)
        
        # Bar comparison of Observed vs Expected
        cats = ['Low Risk: High Yield', 'Low Risk: Low Yield', 'High Risk: High Yield', 'High Risk: Low Yield']
        observed_flat = obs.flatten()
        expected_flat = ex.flatten()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Observed Counts', x=cats, y=observed_flat, marker_color='#06b6d4'))
        fig.add_trace(go.Bar(name='Expected Under H₀', x=cats, y=expected_flat, marker_color='#64748b'))
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Chi-Square Independence Test</b> (χ² = {chi2:.2f}, p-value = {p:.4e})',
            'yaxis_title': 'Count of Farms',
            'barmode': 'group',
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"{chi2:.2f}", f"{p:.4e}", fig
