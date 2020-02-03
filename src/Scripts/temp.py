from src.Experiment import *
from src.DFcode.DatDF import DatDF
from src.DFcode.SetupDF import SetupDF

if __name__ == '__main__':
    setupdf = SetupDF()
    datdf = DatDF()
    dat = make_dat_standard(4, dfoption='overwrite', dattypes='pinch')
    datdf.update_dat(dat)
    datdf.save()
    dat.Pinch.plot()