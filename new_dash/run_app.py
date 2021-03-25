from dash_dashboard.app import get_app
from new_dash.pages import single_entropy

if __name__ == '__main__':
    app = get_app([single_entropy])
    app.run_server(port=8050, debug=True)