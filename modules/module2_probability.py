"""
Module 2: Probability and Distributions
- Probability Fundamentals & Bayes' Theorem Lab
- Discrete Distributions (Binomial, Poisson)
- Continuous Distributions (Normal, Uniform, Exponential)
- Random Variables, Expectation E[X], Variance Var(X), and Law of Large Numbers (LLN)
"""
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, callback

DARK_LAYOUT = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(6, 19, 14, 0.65)',
    'font': {'color': '#f0fdf4', 'family': 'Plus Jakarta Sans, sans-serif'},
    'xaxis': {'gridcolor': 'rgba(52, 211, 153, 0.12)', 'zerolinecolor': 'rgba(52, 211, 153, 0.25)'},
    'yaxis': {'gridcolor': 'rgba(52, 211, 153, 0.12)', 'zerolinecolor': 'rgba(52, 211, 153, 0.25)'},
    'margin': {'l': 40, 'r': 30, 't': 40, 'b': 40}
}

def layout():
    return html.Div([
        html.Div([
            html.Span("Phase 2 — Stochastic Climate Risk", className="section-badge"),
            html.H2("Climate Anomaly Probabilities & Weather Shock Modeling", className="section-title"),
            html.P("Model early drought warning systems via Bayes' Theorem, quantify extreme weather occurrences, and analyze long-term yield loss expectations.", className="section-subtitle")
        ]),
        
        dcc.Tabs(id="m2-subtabs", value="bayes-fundamentals", children=[
            dcc.Tab(label="1. Drought Early Warning & Bayes' System", value="bayes-fundamentals"),
            dcc.Tab(label="2. Precipitation & Shock Distribution Sandbox", value="distributions"),
            dcc.Tab(label="3. Expected Loss Convergence & Law of Large Numbers", value="expectation-lln")
        ], className="mb-4"),
        
        html.Div(id="m2-content")
    ])

def bayes_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Bayes' Theorem & Conditional Probabilities", className="control-label"),
                    html.P("Real-world scenario: Early Climate Drought Alarm System.", className="control-desc"),
                    
                    html.Label("Prior Probability of Severe Drought P(D):", className="control-label mt-2"),
                    dcc.Slider(
                        id="m2-bayes-prior",
                        min=0.01, max=0.50, step=0.01, value=0.08,
                        marks={0.01: '1%', 0.08: '8%', 0.25: '25%', 0.50: '50%'},
                        className="mb-3"
                    ),
                    
                    html.Label("Sensor Sensitivity P(Alarm | Drought):", className="control-label"),
                    dcc.Slider(
                        id="m2-bayes-sens",
                        min=0.50, max=0.99, step=0.01, value=0.92,
                        marks={0.5: '50%', 0.75: '75%', 0.92: '92%', 0.99: '99%'},
                        className="mb-3"
                    ),
                    
                    html.Label("False Alarm Rate P(Alarm | No Drought):", className="control-label"),
                    dcc.Slider(
                        id="m2-bayes-fpr",
                        min=0.01, max=0.30, step=0.01, value=0.05,
                        marks={0.01: '1%', 0.05: '5%', 0.15: '15%', 0.30: '30%'},
                        className="mb-3"
                    ),
                    
                    html.Div([
                        html.H6("Bayes' Formula:", className="text-info font-weight-bold"),
                        html.Div("P(D|A) = [P(A|D) × P(D)] / P(A)", className="formula-badge d-block mb-1"),
                        html.Div("P(A) = P(A|D)P(D) + P(A|Dᶜ)P(Dᶜ)", className="formula-badge d-block")
                    ], className="p-3 bg-dark rounded border border-secondary mt-3")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.Div("Prior P(Drought)", className="stat-card-header"),
                            html.Div(id="m2-bayes-val-prior", className="stat-card-value text-muted"),
                            html.Div("Baseline probability", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Total P(Alarm)", className="stat-card-header"),
                            html.Div(id="m2-bayes-val-total", className="stat-card-value text-warning"),
                            html.Div("Denominator evidence", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-md-4 mb-3"),
                    
                    html.Div([
                        html.Div([
                            html.Div("Posterior P(Drought | Alarm)", className="stat-card-header"),
                            html.Div(id="m2-bayes-val-posterior", className="stat-card-value text-success"),
                            html.Div("Updated probability", className="stat-card-subtext")
                        ], className="stat-card")
                    ], className="col-md-4 mb-3"),
                ], className="row"),
                
                html.Div([
                    dcc.Graph(id="m2-graph-bayes", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-lg-8 mb-4")
        ], className="row")
    ])

def distributions_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Distribution Sandbox", className="control-label"),
                    html.P("Select discrete or continuous models, adjust parameters, and calculate exact probability intervals.", className="control-desc"),
                    
                    html.Label("Distribution Family:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m2-dist-family",
                        options=[
                            {'label': 'Normal / Gaussian (Continuous)', 'value': 'normal'},
                            {'label': 'Binomial Distribution (Discrete)', 'value': 'binomial'},
                            {'label': 'Poisson Distribution (Discrete)', 'value': 'poisson'},
                            {'label': 'Exponential Distribution (Continuous)', 'value': 'exponential'},
                            {'label': 'Uniform Distribution (Continuous)', 'value': 'uniform'}
                        ],
                        value='normal',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    # Normal Controls Container
                    html.Div(id="m2-ctrls-norm", children=[
                        html.Label("Mean (μ):", className="control-label"),
                        dcc.Slider(id="m2-norm-mu", min=-10, max=10, step=1, value=0, marks={-10: '-10', 0: '0', 10: '10'}),
                        html.Label("Std Dev (σ):", className="control-label mt-2"),
                        dcc.Slider(id="m2-norm-sigma", min=0.5, max=5, step=0.5, value=1.0, marks={0.5: '0.5', 1: '1', 3: '3', 5: '5'})
                    ], style={'display': 'block'}),
                    
                    # Binomial Controls Container
                    html.Div(id="m2-ctrls-binom", children=[
                        html.Label("Number of Trials (n):", className="control-label"),
                        dcc.Slider(id="m2-binom-n", min=5, max=50, step=1, value=20, marks={5: '5', 20: '20', 50: '50'}),
                        html.Label("Success Probability (p):", className="control-label mt-2"),
                        dcc.Slider(id="m2-binom-p", min=0.05, max=0.95, step=0.05, value=0.4, marks={0.1: '0.1', 0.5: '0.5', 0.9: '0.9'})
                    ], style={'display': 'none'}),
                    
                    # Poisson Controls Container
                    html.Div(id="m2-ctrls-pois", children=[
                        html.Label("Rate Parameter (λ):", className="control-label"),
                        dcc.Slider(id="m2-pois-lam", min=0.5, max=25, step=0.5, value=5.0, marks={1: '1', 5: '5', 15: '15', 25: '25'})
                    ], style={'display': 'none'}),
                    
                    # Exponential Controls Container
                    html.Div(id="m2-ctrls-exp", children=[
                        html.Label("Rate Parameter (λ):", className="control-label"),
                        dcc.Slider(id="m2-exp-lam", min=0.2, max=3.0, step=0.2, value=1.0, marks={0.2: '0.2', 1.0: '1.0', 3.0: '3.0'})
                    ], style={'display': 'none'}),
                    
                    # Uniform Controls Container
                    html.Div(id="m2-ctrls-unif", children=[
                        html.Label("Lower Bound (a):", className="control-label"),
                        dcc.Slider(id="m2-unif-a", min=-10, max=0, step=1, value=0, marks={-10: '-10', 0: '0'}),
                        html.Label("Upper Bound (b):", className="control-label mt-2"),
                        dcc.Slider(id="m2-unif-b", min=1, max=10, step=1, value=5, marks={1: '1', 5: '5', 10: '10'})
                    ], style={'display': 'none'}),
                    
                    html.Label("Calculate P(X ≤ Cutoff):", className="control-label mt-3"),
                    dcc.Slider(
                        id="m2-cutoff-val",
                        min=-10, max=50, step=0.5, value=1.0,
                        marks={-10: '-10', 0: '0', 10: '10', 25: '25', 50: '50'},
                        className="mb-3"
                    ),
                    
                    html.Div(id="m2-interval-prob-output", className="p-3 bg-dark rounded border border-info text-center font-weight-bold text-info")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="m2-graph-dist", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-lg-8 mb-4")
        ], className="row")
    ])

def expectation_lln_layout():
    return html.Div([
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Law of Large Numbers & Expectation", className="control-label"),
                    html.P("Watch the empirical sample average converge to the theoretical expected value E[X] as trial count increases.", className="control-desc"),
                    
                    html.Label("Random Experiment Type:", className="control-label mt-2"),
                    dcc.Dropdown(
                        id="m2-lln-type",
                        options=[
                            {'label': 'Fair 6-Sided Die (E[X] = 3.5)', 'value': 'die'},
                            {'label': 'Loaded Climate Event Die (Biased, E[X] = 4.8)', 'value': 'biased_die'},
                            {'label': 'Exponential Yield Waiting Time (λ=0.5, E[X]=2.0)', 'value': 'exp'}
                        ],
                        value='die',
                        clearable=False,
                        className="mb-3"
                    ),
                    
                    html.Label("Total Number of Trials:", className="control-label"),
                    dcc.Slider(
                        id="m2-lln-trials",
                        min=100, max=3000, step=100, value=1000,
                        marks={100: '100', 1000: '1000', 2000: '2000', 3000: '3000'},
                        className="mb-3"
                    ),
                    
                    html.Div([
                        html.H6("Key Theorems:", className="text-info font-weight-bold"),
                        html.P("Weak Law of Large Numbers states that for any ε > 0, P(|X̄ₙ - μ| ≥ ε) → 0 as n → ∞.", className="small text-muted mb-2"),
                        html.Div("E[X] = ∫ x f(x) dx  or  Σ x P(x)", className="formula-badge d-block mb-1"),
                        html.Div("Var(X) = E[(X - E[X])²]", className="formula-badge d-block")
                    ], className="p-3 bg-dark rounded border border-secondary mt-3")
                ], className="stat-card h-100")
            ], className="col-lg-4 mb-4"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="m2-graph-lln", config={'displayModeBar': True, 'responsive': True})
                ], className="stat-card")
            ], className="col-lg-8 mb-4")
        ], className="row")
    ])

def register_callbacks(app):
    @app.callback(
        Output("m2-content", "children"),
        Input("m2-subtabs", "value")
    )
    def render_m2_tab(tab):
        if tab == "bayes-fundamentals":
            return bayes_layout()
        elif tab == "distributions":
            return distributions_layout()
        elif tab == "expectation-lln":
            return expectation_lln_layout()
        return html.Div()

    @app.callback(
        [Output("m2-bayes-val-prior", "children"),
         Output("m2-bayes-val-total", "children"),
         Output("m2-bayes-val-posterior", "children"),
         Output("m2-graph-bayes", "figure")],
        [Input("m2-bayes-prior", "value"),
         Input("m2-bayes-sens", "value"),
         Input("m2-bayes-fpr", "value")]
    )
    def update_bayes(prior, sens, fpr):
        prior_d = prior or 0.08
        sens = sens or 0.92
        fpr = fpr or 0.05
        prior_nod = 1.0 - prior_d
        
        p_alarm_and_d = sens * prior_d
        p_alarm_and_nod = fpr * prior_nod
        p_alarm = p_alarm_and_d + p_alarm_and_nod
        posterior = p_alarm_and_d / p_alarm if p_alarm > 0 else 0
        
        categories = ['Prior P(Drought)', 'P(Alarm Triggered)', 'Posterior P(Drought | Alarm)']
        values = [prior_d * 100, p_alarm * 100, posterior * 100]
        colors = ['#64748b', '#f59e0b', '#10b981']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='outside'
        ))
        
        total_farms = 10000
        drought_farms = int(total_farms * prior_d)
        nodrought_farms = total_farms - drought_farms
        true_pos = int(drought_farms * sens)
        false_pos = int(nodrought_farms * fpr)
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Bayesian Probability Update</b> (In 10,000 Farms: {true_pos} True Alarms vs. {false_pos} False Alarms)',
            'yaxis_title': 'Probability (%)',
            'yaxis': {'range': [0, 110], 'gridcolor': 'rgba(255,255,255,0.08)'},
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        return f"{prior_d * 100:.1f}%", f"{p_alarm * 100:.1f}%", f"{posterior * 100:.1f}%", fig

    # Toggle distribution parameter containers
    @app.callback(
        [Output("m2-ctrls-norm", "style"),
         Output("m2-ctrls-binom", "style"),
         Output("m2-ctrls-pois", "style"),
         Output("m2-ctrls-exp", "style"),
         Output("m2-ctrls-unif", "style")],
        Input("m2-dist-family", "value")
    )
    def toggle_dist_controls(dist):
        styles = [{'display': 'none'}] * 5
        if dist == 'normal':
            styles[0] = {'display': 'block'}
        elif dist == 'binomial':
            styles[1] = {'display': 'block'}
        elif dist == 'poisson':
            styles[2] = {'display': 'block'}
        elif dist == 'exponential':
            styles[3] = {'display': 'block'}
        elif dist == 'uniform':
            styles[4] = {'display': 'block'}
        return styles[0], styles[1], styles[2], styles[3], styles[4]

    @app.callback(
        [Output("m2-graph-dist", "figure"),
         Output("m2-interval-prob-output", "children")],
        [Input("m2-dist-family", "value"),
         Input("m2-cutoff-val", "value"),
         Input("m2-norm-mu", "value"),
         Input("m2-norm-sigma", "value"),
         Input("m2-binom-n", "value"),
         Input("m2-binom-p", "value"),
         Input("m2-pois-lam", "value"),
         Input("m2-exp-lam", "value"),
         Input("m2-unif-a", "value"),
         Input("m2-unif-b", "value")]
    )
    def update_dist_plot(dist, cutoff, mu, sigma, bn, bp, lam, exp_lam, ua, ub):
        fig = go.Figure()
        cutoff = cutoff if cutoff is not None else 1.0
        
        if dist == 'normal':
            mu = mu if mu is not None else 0
            sigma = sigma if sigma is not None else 1
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 400)
            y = stats.norm.pdf(x, mu, sigma)
            prob = stats.norm.cdf(cutoff, mu, sigma)
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#38bdf8', width=2.5), name='PDF f(x)'))
            x_shade = np.linspace(mu - 4*sigma, cutoff, 200)
            y_shade = stats.norm.pdf(x_shade, mu, sigma)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_shade, [cutoff, x_shade[0]]]),
                y=np.concatenate([y_shade, [0, 0]]),
                fill='toself', fillcolor='rgba(14, 165, 233, 0.4)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'P(X ≤ {cutoff:.1f}) = {prob:.4f}'
            ))
            title_text = f"Normal Distribution N(μ={mu}, σ={sigma}) — Continuous PDF"
            
        elif dist == 'binomial':
            bn = bn or 20
            bp = bp or 0.4
            k = np.arange(0, bn + 1)
            pmf = stats.binom.pmf(k, bn, bp)
            prob = stats.binom.cdf(int(cutoff), bn, bp)
            colors = ['#10b981' if ki <= cutoff else '#334155' for ki in k]
            
            fig.add_trace(go.Bar(x=k, y=pmf, marker_color=colors, name='PMF P(X=k)'))
            title_text = f"Binomial Distribution Bin(n={bn}, p={bp}) — Discrete PMF"
            
        elif dist == 'poisson':
            lam = lam or 5.0
            k = np.arange(0, int(lam + 4*np.sqrt(lam)) + 2)
            pmf = stats.poisson.pmf(k, lam)
            prob = stats.poisson.cdf(int(cutoff), lam)
            colors = ['#f59e0b' if ki <= cutoff else '#334155' for ki in k]
            
            fig.add_trace(go.Bar(x=k, y=pmf, marker_color=colors, name='PMF P(X=k)'))
            title_text = f"Poisson Distribution Pois(λ={lam}) — Discrete PMF"
            
        elif dist == 'exponential':
            exp_lam = exp_lam or 1.0
            x = np.linspace(0, 8 / exp_lam, 400)
            y = stats.expon.pdf(x, scale=1.0/exp_lam)
            prob = stats.expon.cdf(cutoff, scale=1.0/exp_lam)
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#ec4899', width=2.5), name='PDF f(x)'))
            x_shade = np.linspace(0, max(0, cutoff), 200)
            y_shade = stats.expon.pdf(x_shade, scale=1.0/exp_lam)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_shade, [cutoff, 0]]),
                y=np.concatenate([y_shade, [0, 0]]),
                fill='toself', fillcolor='rgba(236, 72, 153, 0.4)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'P(X ≤ {cutoff:.1f}) = {prob:.4f}'
            ))
            title_text = f"Exponential Distribution Exp(λ={exp_lam}) — Waiting Times"
            
        else: # uniform
            ua = ua if ua is not None else 0
            ub = ub if ub is not None else 5
            if ub <= ua:
                ub = ua + 1
            x = np.linspace(ua - 2, ub + 2, 400)
            y = np.where((x >= ua) & (x <= ub), 1.0 / (ub - ua), 0)
            prob = stats.uniform.cdf(cutoff, loc=ua, scale=ub - ua)
            
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#8b5cf6', width=2.5), name='PDF f(x)'))
            x_shade = np.linspace(ua, min(max(ua, cutoff), ub), 200)
            y_shade = np.full_like(x_shade, 1.0 / (ub - ua))
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_shade, [cutoff, ua]]),
                y=np.concatenate([y_shade, [0, 0]]),
                fill='toself', fillcolor='rgba(139, 92, 246, 0.4)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'P(X ≤ {cutoff:.1f}) = {prob:.4f}'
            ))
            title_text = f"Uniform Distribution U(a={ua}, b={ub}) — Continuous PDF"
            
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>{title_text}</b>',
            'xaxis_title': 'Random Variable X',
            'yaxis_title': 'Density / Probability',
            'height': 420
        })
        fig.update_layout(layout_dict)
        
        prob_text = f"Cumulative Probability P(X ≤ {cutoff:.2f}) = {prob:.4f} ({prob*100:.2f}%)"
        return fig, prob_text

    @app.callback(
        Output("m2-graph-lln", "figure"),
        [Input("m2-lln-type", "value"),
         Input("m2-lln-trials", "value")]
    )
    def update_lln(exp_type, n_trials):
        np.random.seed(42)
        n_trials = n_trials or 1000
        trials_arr = np.arange(1, n_trials + 1)
        
        if exp_type == 'die':
            theoretical_exp = 3.5
            outcomes = np.random.choice([1, 2, 3, 4, 5, 6], size=n_trials)
        elif exp_type == 'biased_die':
            theoretical_exp = 4.8
            p = [0.05, 0.05, 0.1, 0.1, 0.3, 0.4]
            outcomes = np.random.choice([1, 2, 3, 4, 5, 6], size=n_trials, p=p)
        else: # exp
            theoretical_exp = 2.0
            outcomes = np.random.exponential(scale=2.0, size=n_trials)
            
        running_means = np.cumsum(outcomes) / trials_arr
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trials_arr, y=running_means,
            mode='lines',
            line=dict(color='#06b6d4', width=2),
            name='Sample Running Mean (X̄ₙ)'
        ))
        
        fig.add_trace(go.Scatter(
            x=[1, n_trials], y=[theoretical_exp, theoretical_exp],
            mode='lines',
            line=dict(color='#f43f5e', width=2.5, dash='dash'),
            name=f'Theoretical E[X] = {theoretical_exp:.2f}'
        ))
        
        layout_dict = DARK_LAYOUT.copy()
        layout_dict.update({
            'title': f'<b>Law of Large Numbers (LLN) Convergence over {n_trials} Trials</b> (Final X̄ = {running_means[-1]:.3f})',
            'xaxis_title': 'Number of Trials (n)',
            'yaxis_title': 'Cumulative Sample Average X̄ₙ',
            'height': 420
        })
        fig.update_layout(layout_dict)
        return fig
