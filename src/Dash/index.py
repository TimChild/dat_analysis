import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.Dash.app import app
from src.Dash.pages import single_dat_view, test_page_2

index_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.layout = index_layout


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/pages/single-dat-view':
        return single_dat_view.layout
    elif pathname == '/pages/second-page':
        return test_page_2.layout
    # elif pathname == '/pages/app2':
    #     return app2.layout
    else:
        return single_dat_view.layout


app.validation_layout = html.Div([
    single_dat_view.layout,
    test_page_2.layout,
    index_layout
])

if __name__ == '__main__':
    app.run_server(debug=True)
