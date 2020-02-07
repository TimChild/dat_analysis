from src.Scripts.StandardImports import *


if __name__ == '__main__':
    dat = make_dat_standard(20)
    datdf = DF.DatDF()
    setupdf = SF.SetupDF()
    dat.Logs.x_label = 'Gates(6,15) /mV'
    datdf.update_dat(dat)