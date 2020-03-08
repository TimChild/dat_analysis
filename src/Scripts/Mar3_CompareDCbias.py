from src.Scripts.StandardImports import *
from src.DatCode.DCbias import plot_standard_dcbias
from src.Configs.Jan20Config import add_mag_to_logs

datnums = list(range(1026, 1037, 2))  # From sweep then scan
datnums2 = list(range(1158, 1199, 8))  # From HQPC steps
datnums3 = list(range(1301, 1304+1))  # From Magnet biases (50, -50, 100, 200mT)
datnums4 = list(range(1305, 1327, 2))  # All 200mT, 40nA max HQPC -550mV
datnums5 = list(range(1327, 1341+1, 2))  # Changing HQPC between -550 and -590, and Fields through [200, 100, 50, -50]mT
datnums7 = list(range(1373, 1393+1, 2))  # Same as above with -100 and -200mT and longer scans (601 lines per DCBias)

def add_fields_datnums3(dats):
    for dat, field in zip(dats, [50, -50, 100, 200]):
        dat.Logs.magy = field

def plot_multiple_i_sense(dats):
    # fig, axs = PF.make_axes(len(dats), plt_kwargs={'sharex':True})
    fig, axs = plt.subplots(5,2, sharex=True, figsize=(5, 10))
    axs = axs.flatten()
    for a, dat in zip(axs, dats):
        PF.display_2d(dat.Data.x_array, dat.Data.y_array, dat.Data.i_sense, a, dat=dat)
        PF.ax_text(a, f'Field={dat.Instruments.magy.field:.1f}mT')
        a.set_title(f'Field={dat.Instruments.magy.field:.1f}mT, HQPC={dat.Logs.fdacs[0]}mV')
    PF.add_standard_fig_info(plt.gcf())



def plot_dc_bias(dat):
    fig, axs = PF.make_axes(4)
    plot_standard_dcbias(dat,axs, plots=[1,2,3,4])
    fig.suptitle(f'Dat{dat.datnum} - Field={dat.Instruments.magy.field:.1f}mT')



if __name__ == '__main__':
    # dats = [make_dat_standard(num, dfoption='load') for num in datnums7[4:]]
    # add_mag_to_logs(dat)
    # plot_multiple_i_sense(dats)
    # for dat in dats:
    #     plot_dc_bias(dat)
    pass

