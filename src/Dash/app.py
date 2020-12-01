import dash
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

app = dash.Dash(__name__, external_stylesheets=[DASH_THEME])
server = app.server

# The names to display for all pages in App, and the html links to use for all pages of app
ALL_PAGES = {
    "Single Dat": '/single-dat-view',
    "Test Page 2": '/second-page',
    "Transition": '/transition',
}