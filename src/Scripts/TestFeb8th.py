from src.Scripts.StandardImports import *


if __name__ == '__main__':
    datdf = DF.DatDF()
    setupdf = SF.SetupDF()
    datdf.change_in_excel()
    # for datnum in range(25, 26):
    #     print(datnum)
    #     dat = make_dat_standard(datnum, dfoption='load')
    #     datdf.update_dat(dat)
