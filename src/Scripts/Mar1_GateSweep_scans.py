from src.Scripts.StandardImports import *
import src.DatCode.Entropy as E
import src.DatCode.Transition as T

""" 
This is for plotting dats 1025 to 1050

Set gates to sweep back and forth from -200mV to -800mV for about an hour, then did 5 repeated scans of DCbias and 
Entropy and repeated

Trying to see if transition becomes more stable after sweeping gates

"""



################################ This is just to look at dats
datnums = list(range(1030, 1035))
dcdat = 1026
# datnums = [1173]
# dcdat = 1166
for num in datnums:
    dat = make_dat_standard(num, dfoption='load')
    if 'entropy' in dat.dattype:
        if dat.Entropy.int_entropy_initialized is False:
            dcbias_dat = make_dat_standard(dcdat, dfoption='load')
            dT = dcbias_dat.DCbias.get_dt_at_current(dat.Instruments.srs1.out / 50 * np.sqrt(2))
            dat.Entropy.init_integrated_entropy_average(dT_mV=dT, dT_err=None, amplitude=dat.Transition.amp, amplitude_err=None)
            cfg.yes_to_all = True
            datdf.update_dat(dat)
            cfg.yes_to_all = False
            print(f'Dat{dat.datnum} had integrated entropy initialized with dT = {dT:.3f}mV and amp = {dat.Transition.amp:.3f}nA')
        fig, axs = PF.make_axes(5)
        E.plot_standard_entropy(dat, axs, plots=[1,2,4,10,11])
        T.plot_standard_transition(dat, [axs[-1]], plots=[1])
    else:
        print(f'dat{dat.datnum} is not an entropy dat. It has types [{dat.dattype}]')




#####################################
