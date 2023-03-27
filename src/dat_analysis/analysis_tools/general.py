from typing import Callable
from dataclasses import dataclass
from ..useful_functions import sig_fig
from progressbar import progressbar
import pandas as pd


@dataclass
class ColumnDescriptor:
    """For laying out what will fill a column in a DataFrame when filled with a list of DatHDF objects

    i.e. A single DatHDF will be called to fill each entry of the column
    """

    name: str  # Name of column in DF
    function: Callable  # Function that takes `dat` and returns value for DF
    sig_fig: int = None

    def __call__(self, *args, **kwargs):
        v = self.function(*args, **kwargs)
        if self.sig_fig:
            v = sig_fig(v, self.sig_fig)
        return v


def dats_to_df(
    dats,
    column_descriptors: list[ColumnDescriptor],
    index_descriptor: ColumnDescriptor = None,
    show_progressbar=True,
):
    """Create DataFrame with dat info based on descriptors passed
    Examples:
        def  get_val(dat):
            return dat.Logs.dacs['DAC1']

        dac = ColumnDescriptor('Dac 1', get_val, sig_fig=2)

        df = dats_to_df(dats, [dac])
        # Creates df with datnum as index, and a single column with name 'Dac 1' and the DAC1 values from each dat in dats (where values are rounded to 2 s.f.)
    """
    cols = [d.name for d in column_descriptors]
    if not index_descriptor:
        index_descriptor = ColumnDescriptor("Datnum", lambda dat: dat.datnum)

    # Annoyingly, the threaded approach is slower :(
    # with ThreadPoolExecutor(max_workers=100) as executor:
    #     index = executor.map(index_descriptor, dats)
    #     values = executor.map(lambda dat: [d(dat) for d in column_descriptors], dats)

    index = []
    values = []
    if show_progressbar:
        dats = progressbar(dats)
    for dat in dats:
        index.append(index_descriptor(dat))
        values.append([d(dat) for d in column_descriptors])

    df = pd.DataFrame(
        data=values, columns=cols, index=pd.Index(index, name=index_descriptor.name)
    )
    return df


def get_unique_df_vals(df):
    """
    Creates a dataframe with same column headers, only containing unique values in the rows
    """
    df_unique = pd.DataFrame({})
    for column in df:
        df_unique[column] = df[column].unique()
    return df_unique


def summarize_df(df):
    print(df.nunique())
    print(f"\n")
    print("Unique Values for each column are:")
    for col in df.columns:
        uniques = pd.unique(df[col])
        if len(uniques) <= 10:
            print(f"{col}: {uniques}")
        else:
            print(f"{col}: {uniques[:10]}, ...")
