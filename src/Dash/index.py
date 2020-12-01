import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import logging
from src.Dash.app import app, ALL_PAGES

# Import any Pages to be added to app
from src.Dash.pages import single_dat_view, test_page_2, Transition

logger = logging.getLogger(__name__)

# One Div for whole page which can switch between Pages
index_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Set the app layout to index_layout only (the rest will be generated with the callback below)
app.layout = index_layout


# Make list of available pages (should be very similar to app.ALL_PAGES)
DEFAULT_PAGE = single_dat_view.layout
PAGES = {
    '/single-dat-view': single_dat_view.layout,
    '/second-page': test_page_2.layout,
    '/transition': Transition.layout,
}
if mismatch := set(PAGES.keys()).difference(set(ALL_PAGES.values())):
    logger.warning(f'These pages are mismatched between app.ALL_PAGES and index.PAGES:\n'
                   f'{mismatch}')


# Callback to be able to switch between whole pages
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname in PAGES:
        return PAGES[pathname]
    else:
        logger.warning(f'{pathname} not found, showing default page')
        return DEFAULT_PAGE


"""
Validation layout only, this just checks whether all the callbacks etc in other pages make sense, but this layout 
will not actually be used. Ideally all ids should be unique throughout the whole app, but it is sufficient that id's 
are unique between full pages since only one real Page will ever be loaded at a time. 
"""
app.validation_layout = html.Div([
    single_dat_view.layout,
    test_page_2.layout,
    index_layout
])

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)