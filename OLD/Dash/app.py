from dash_extensions.enrich import DashProxy
import dash_bootstrap_components as dbc
import plotly.io as pio

# Setup some theme stuff
THEME = ''
if THEME == 'dark':
    DASH_THEME = dbc.themes.DARKLY
    plotly_theme = 'plotly_dark'
else:
    DASH_THEME = dbc.themes.BOOTSTRAP
    plotly_theme = 'none'
pio.templates.default = plotly_theme

app = DashProxy(name=__name__, external_stylesheets=[DASH_THEME],
                transforms=[
                ])
server = app.server

# The names to display for all pages in App, and the html links to use for all pages of app
ALL_PAGES = {
    "Single Dat": '/single-dat-view',
    "Transition": '/transition',
    "Shared": "/shared",
    "Square Heating Entropy": "/square-entropy",
    # "Gamma Broadened": "/gamma-broadened",
}