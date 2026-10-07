"""
AgroClimate Analytics Platform
Global Agriculture & Climate Change Intelligence
1. Crop Baselines & Spread (Descriptive Statistics)
2. Climate Risk & Shocks (Probability & Distributions)
3. Yield Impact & Hypotheses (Inferential Statistics)
4. Visual Intelligence (Data Visualization)
5. Adaptation & Policy Synthesis (Capstone Empirical Study)
"""
import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

import plotly.io as pio
pio.templates.default = "plotly_white"

# Import modules
from modules import (
    module1_descriptive,
    module2_probability,
    module3_inferential,
    module4_visualization,
    module5_capstone
)

# Initialize Dash application with Bootswatch Flatly light theme and custom styles
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "AgroClimate Intelligence | Global Crop & Climate Analytics"
server = app.server

# Single-line Top Navigation Bar
navbar = html.Header([
    html.Div([
        # Left Brand Group
        html.Div([
            html.I(className="fa-solid fa-seedling brand-icon me-2"),
            html.Span("AGRI-CLIMATE INTELLIGENCE", className="brand-title"),
            html.Span("Global Resilience & Analytics", className="brand-badge d-none d-md-inline-block")
        ], className="navbar-brand-group"),
        
        # Right Links Group
        html.Div([
            html.Span("10,000 Records • 2000–2024", className="navbar-meta-badge d-none d-lg-inline-block me-3"),
            html.Button([
                html.I(className="fa-solid fa-moon me-1", id="theme-btn-icon"),
                html.Span("Dark Mode", id="theme-btn-label")
            ], id="theme-toggle-btn", className="btn-theme-toggle me-3", n_clicks=0),
            html.A([html.I(className="fa-solid fa-book-open me-1"), "Docs"], href="#", className="navbar-link me-3"),
            html.A([html.I(className="fa-brands fa-github me-1"), "Repo"], href="https://github.com", target="_blank", className="navbar-link")
        ], className="navbar-links-group")
    ], className="navbar-inner-container")
], className="navbar-custom mb-3")

# Main App Layout
app.layout = html.Div(id="app-container", className="theme-light", children=[
    dcc.Store(id="theme-store", data="light", storage_type="local"),
    navbar,
    
    html.Div([
        # Single-Line Main Navigation Tabs
        html.Div([
            dcc.Tabs(
                id="main-tabs",
                value="tab-module1",
                parent_className="tabs-nav-container",
                className="single-line-tabs",
                mobile_breakpoint=0,
                children=[
                    dcc.Tab(
                        label="1. Crop Baselines & Spread",
                        value="tab-module1",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                    dcc.Tab(
                        label="2. Climate Risk & Shocks",
                        value="tab-module2",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                    dcc.Tab(
                        label="3. Yield Impact & Hypotheses",
                        value="tab-module3",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                    dcc.Tab(
                        label="4. Visual Intelligence",
                        value="tab-module4",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                    dcc.Tab(
                        label="5. Adaptation & Policy Capstone",
                        value="tab-module5",
                        className="custom-tab",
                        selected_className="custom-tab--selected"
                    ),
                ]
            )
        ], className="tabs-bar-wrapper mb-4"),
        
        # Dynamic Main Tab Content with smooth transitions
        dcc.Loading(
            id="main-tab-loading",
            type="dot",
            color="#10b981",
            children=[
                html.Div(id="main-tab-content", className="mt-2 tab-fade-in")
            ]
        ),
        
        # Footer
        html.Footer([
            html.Hr(className="border-secondary mt-5 mb-3"),
            html.Div([
                html.P([
                    "AgroClimate Intelligence Platform • Powered by ",
                    html.B("Python, Dash & Plotly"),
                    " • Real Empirical Dataset: 10,000 Global Records (2000–2024)"
                ], className="text-muted small text-center mb-0")
            ])
        ], className="pb-4")
    ], className="container-fluid px-3 px-md-4")
])

# Register callbacks for all individual modules
module1_descriptive.register_callbacks(app)
module2_probability.register_callbacks(app)
module3_inferential.register_callbacks(app)
module4_visualization.register_callbacks(app)
module5_capstone.register_callbacks(app)

# Theme Switcher Callback
@app.callback(
    [Output("app-container", "className"),
     Output("theme-btn-icon", "className"),
     Output("theme-btn-label", "children"),
     Output("theme-store", "data")],
    Input("theme-toggle-btn", "n_clicks"),
    dash.State("theme-store", "data"),
    prevent_initial_call=False
)
def toggle_theme(n_clicks, current_theme):
    if n_clicks is None or n_clicks == 0:
        theme = current_theme or "light"
    else:
        theme = "dark" if current_theme == "light" else "light"
        
    if theme == "dark":
        return "theme-dark", "fa-solid fa-sun me-1", "Light Mode", "dark"
    else:
        return "theme-light", "fa-solid fa-moon me-1", "Dark Mode", "light"

# Main router callback
@app.callback(
    Output("main-tab-content", "children"),
    Input("main-tabs", "value")
)
def render_main_content(active_tab):
    if active_tab == "tab-module1":
        return module1_descriptive.layout()
    elif active_tab == "tab-module2":
        return module2_probability.layout()
    elif active_tab == "tab-module3":
        return module3_inferential.layout()
    elif active_tab == "tab-module4":
        return module4_visualization.layout()
    elif active_tab == "tab-module5":
        return module5_capstone.layout()
    return html.Div("Please select a module tab above.")

if __name__ == '__main__':
    print("Starting AgroClimate Intelligence Dashboard on http://127.0.0.1:8050 ...")
    app.run(debug=True, port=8050)
