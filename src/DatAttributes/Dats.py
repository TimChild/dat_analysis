from typing import List, Tuple, Union
from src.Core import make_dat_standard
import src.DFcode.DatDF as DF
import src.PlottingFunctions as PF

# TODO: Where to store this as pickle? How to load from pickle?
# TODO: Add methods to this for printing information from dataframes (mostly just one df)


class Dats(object):
    """Holds list of (datnum, datname, dfname) with methods to interact with all dats in that list"""
    def __init__(self, dat_index_list: List[Tuple] = None, dat_list: List = None,
                 dfnames: Union[str, List[str]] = None):
        """Makes single list of dat identifiers given combination of indexs, dat objects, and either single or list of
        dfnames"""
        if dfnames is None:  # Default dfname
            dfnames = ['default'] * len(dat_index_list)
        elif type(dfnames) is str:  # Same dfname for each (datnum, datname)
            dfnames = [dfnames] * len(dat_index_list)
        elif type(dfnames) is list: # If providing dfname for each (datnum,datname)
            assert len(dfnames) == len(dat_index_list)
        self.dats = []
        if dat_index_list is None and dat_list is None:
            raise ValueError('"Dats" requires either list of indexes or list of dat objects')
        elif dat_index_list is not None:  # Make sure dats are in datdfs
            for dat_index, dfname in zip(dat_index_list, dfnames):
                datdf = DF.DatDF(dfname=dfname)
                if (dat_index[0], dat_index[1]) not in datdf.df.index:
                    dat = make_dat_standard(dat_index[0], dat_index[1], dfname=dfname)
                    datdf.update_dat(dat)
                    datdf.save()
                self.dats.append((dat_index[0], dat_index[1], dfname))
        if dat_list is not None:  # Add (datnum, datname, dfname) of dat objects to list
            for dat in dat_list:
                if (dat.datnum, dat.datname, dat.dfname) not in dat_index_list:  # Add only dats that haven't already been loaded
                    self.dats.append((dat.datnum, dat.datname, dat.dfname))
                else:
                    print(f'Dat{dat.datnum}[{dat.datname}] from df "{dat.dfname}" has already been loaded from dat_index_list')

    def __iter__(self):
        """Sets iter to start at first dat"""
        self.index = 0
        return self

    def __next__(self):
        """Iterates through dat objects"""  # TODO: Is this fast loading from dataframe/pickles each time??
        if self.index < len(self.dats):
            dat_id = self.dats[self.index]
            datnum, datname, dfname = dat_id[0], dat_id[1], dat_id[2]
            dat = make_dat_standard(datnum, datname, dfoption='load', dfname=dfname)
            return dat
        else:
            raise StopIteration





def plot_transition_fits(dats):
    """Plots i_sense data with di_gamma_fit over top"""
    fig, axs = PF.make_axes(len(dats)*2)

    row=0

    for i, dat in enumerate(dats):
        ax1 = axs[2*i]
        ax2 = axs[2*i + 1]
        x = dat.Data.x_array
        data1d = dat.Transition._data[row]
        PF.display_2d(x, dat.Data.y_array, dat.Transition._data, ax1, dat=dat)
        PF.display_1d(x, data1d, ax2, dat=dat, scatter=True, label='i_sense data')
        ax2.plot(x, dat.Transition._full_fits[row].best_fit, label='di_gamma_fit', c='r')