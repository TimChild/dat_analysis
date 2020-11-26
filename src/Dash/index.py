import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from src.Dash.app import app
from src.Dash.pages import single_dat_view, test_page_2
from src.Dash.BaseClasses import BaseCallbacks

# index_layout = html.Div([
#     dcc.Location(id='url', refresh=False),
#     html.Div(id='page-content')
# ])

index_layout = html.Div([
    single_dat_view.BasePageLayout().top_bar_layout(),
    single_dat_view.layout,
    test_page_2.layout
])

app.layout = index_layout


app.validation_layout = html.Div([
    single_dat_view.layout,
    test_page_2.layout,
    index_layout
])

BaseCallbacks().set_test2_callback()
# @app.callback(Output('page-content', 'children'),
#               Input('url', 'pathname'))
# def display_page(pathname):
#     if pathname == '/pages/single-dat-view':
#         return single_dat_view.layout
#     elif pathname == '/pages/second-page':
#         return test_page_2.layout
#     # elif pathname == '/pages/app2':
#     #     return app2.layout
#     else:
#         return single_dat_view.layout

@app.callback(
    Output('div-page1', 'hidden'),
    Output('div-page2', 'hidden'),
    Input('but-page-switch', 'n_clicks')
)
def toggle_pages(clicks):
    if clicks:
        if clicks % 2:
            return True, False
        else:
            return False, True
    else:
        return True, False


if __name__ == '__main__':
    app.run_server(debug=True)
