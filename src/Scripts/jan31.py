from src.Experiment import *
from src.CoreUtil import open_hdf5
import src.Configs.Main_Config as cfg
import src.DFcode.DatDF as DF

cfg.verbose = False

if __name__ == '__main__':

    hdf = open_hdf5(4, cfg.ddir)
    metadata = hdf['metadata']
    config = metadata.attrs['sc_config']
    logs = metadata.attrs['sweep_logs']
    dat = make_dat_standard(4)
    datdf = DF.DatDF()
    print(datdf.df)
    datdf.update_dat(dat)
    datdf.save()
    a = dat.Data.i_sense
    dat.Data.get_names()
    # dat.Data.get_names()