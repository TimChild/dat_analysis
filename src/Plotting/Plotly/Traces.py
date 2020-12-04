"""
Place to start working on functions to return Plotly traces with some default settings that we often use
"""


import plotly.graph_objects as go


class Line(go.Scatter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



if __name__ == '__main__':
    pass