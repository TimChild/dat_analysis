from src.Scripts.StandardImports import *
from src.DatCode.Pinch import get_gates
import re

def main():
    datdf = DF.DatDF()
    sub_df = datdf.df[datdf.df[('dat_types', 'pinch')] == True]
    datids = sub_df.index
    fig, axs = PF.make_axes(len(datids))
    dats = [make_dat_standard(datid[0], datid[1], dfoption='load') for datid in datids]
    for dat, ax in zip(dats, axs):
        print(dat.datnum)
        gates = get_gates(dat.Logs.comments)  # this gets a list of gates from text like gates=(10,3)
        # dat.Logs.x_label = 'Gates ' + ','.join(gates) + ' /mV'
        dat.Logs.y_label = 'Current /nA'
        datdf.update_dat(dat)

        dat.Pinch.plot_current(dat, ax=ax)
        # plt.draw()
    fig = PF.add_standard_fig_info(plt.gcf())
    fig.suptitle('Pinch off for Nik v2 Device')


if __name__ == '__main__':
    main()