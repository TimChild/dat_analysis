import pandas as pd
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go

from src.plotting.plotly.dat_plotting import OneD


def load_df(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def print_column_names(df: pd.DataFrame) -> None:
    for i, r in enumerate(df.columns):
        print(f'{i}: {r}')


@dataclass
class BFData:
    time: np.ndarray
    oil: np.ndarray
    water_in: np.ndarray
    water_out: np.ndarray

    @classmethod
    def from_df(cls, df: pd.DataFrame, time_col: int, oil_col: int, water_in_col: int, water_out_col: int):
        arr = np.array(df)
        oil = arr[:, oil_col]
        win = arr[:, water_in_col]
        wo = arr[:, water_out_col]
        time = arr[:, time_col]
        return cls(time, oil, win, wo)


def plot_data(data: BFData, date: str) -> go.Figure:
    plotter = OneD(dat=None)
    plotter.MAX_POINTS = 1000000000
    fig = plotter.figure(xlabel='Time', ylabel='Temperature /C', title=f'PT Compressor Temperatures {date}')
    x = data.time
    for d, label in zip([data.oil, data.water_in, data.water_out], ['Oil', 'Water In', 'Water Out']):
        fig.add_trace(plotter.trace(data=d, x=x, name=label, mode='lines'))
    return fig


if __name__ == '__main__':
    filedate = '21-02-03'
    df = load_df(f"D:/OneDrive/GitHub/dat_analysis/dat_analysis/Analysis/Feb2021/Status_{filedate}.log")
    # print_column_names(df)

    data = BFData.from_df(df, time_col=1, oil_col=41, water_in_col=37, water_out_col=39)

    fig = plot_data(data, filedate)
    fig.show(renderer='browser')







