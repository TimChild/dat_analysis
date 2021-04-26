from dash_dashboard.app import get_app
from new_dash.pages import NRGdata, single_entropy


app = get_app([NRGdata, single_entropy])


if __name__ == '__main__':
    remote = False
    port, debug, host = 8057, True, '127.0.0.1'
    if remote is True:
        port, debug, host = 80, False, '0.0.0.0'

    app.run_server(debug=debug, port=port, host=host, threaded=True)


