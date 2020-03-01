from src.Sandbox import *
import src.DatCode.Transition as T
import src.DatCode.Entropy as E

def get_dat(datnum, **kwargs):
    return make_dat_standard(datnum, **kwargs)

def load():
    C.load_dats()


def int_test(dat, dt=0.1, plot=True):
    datamv = np.flip(dat.Entropy.entrav * 2 / 10)
    assert datamv.ndim == 1
    x = np.flip(dat.Data.x_array)
    dx = np.abs((x[-1] - x[0]) / len(x))
    datana = datamv * 1e-3 * 1e-8 * 1e9
    cumsum = np.nancumsum(datana)
    scaling = dx / dt / dat.Transition.amp
    print(f'scaling={scaling:.3f}, from dT = {dt:.3f}mV and amp = {dat.Transition.amp:.3f}nA')
    norm = cumsum * scaling

    print(f'Dat{dat.datnum}\nintegrated[-1] = {norm[-1]:.3f}\nratio(dS/ln(2)) = {norm[-1]/np.log(2):.2f}\nratio(dSmax/dSend) = {np.nanmax(norm)/norm[-1]:.3f}\nln3/ln2 = {np.log(3)/np.log(2):.3f}')

    if plot is True:
        PF.mpluse('qt')
        fig, ax = PF.make_axes(4)
        ax[0].plot(dat.Data.x_array, dat.Data.i_sense[0])
        ax[0].set_title('Charge sensor')

        ax[1].plot(x, datana)
        ax[1].set_title('entropyr /nA')

        ax[2].plot(x, norm)
        ax[2].set_title('Integrated entropy')
        fig.suptitle(f'Dat{dat.datnum}')
        plt.tight_layout()
        plt.show()

def update_and_save(dats):
    datdf = DF.DatDF()
    for dat in dats:
        datdf.update_dat(dat)
    datdf.save()



if __name__ == '__main__':
    # dat = get_dat(205)
    # dat = make_dat_standard(419, dfoption='overwrite')
    # int_test(dat, 0.1, plot=False)
    # PF.mpluse('qt')
    # dats = [make_dat_standard(num, dfoption='load') for num in [418, 419, 420]]
    # for dat in dats:
    #     dat.Entropy.init_integrated_entropy_average(dT_mV=0.1, dT_err=None, amplitude=dat.Transition.amp, amplitude_err=None)

    datdf = DF.DatDF()
    df = datdf.df

    # for num in [467, 470, 471, 474, 477, 481, 484, 487, 491, 494, 497, 501, 504, 507, 511, 514, 517, 521, 524, 527, 531, 534, 537, 541, 544, 547, 551, 554, 557, 561, 564, 567, 571, 574, 577, 581, 584, 587, 591, 594, 597, 601, 604, 607, 611, 614, 617, 621, 624, 627, 631, 634, 637, 641, 644, 647, 651, 654, 657, 661, 664, 667, 671, 674, 677, 681, 684, 687, 691, 694, 697, 701, 704, 707, 711, 714, 717, 719, 721, 723, 726, 728, 730, 733, 735, 737, 740, 742, 744, 747]:  # 253, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 414, 415, 416, 417, 418, 419, 420, 420, 423, 424, 425, 426, 427, 428, 429, 429, 432, 433, 434, 435, 436, 437, 438, 438, 441, 442, 443, 444, 445, 446, 447, 450, 451, 452, 453, 454, 455, 456, 461, 464,
    #
    #     dat = make_dat_standard(num, dfoption='load')
    #     print(f'Working on dat{dat.datnum}')
    #     cfg.yes_to_all = True
    #     if dat.Transition.version != T.Transition._Transition__version:
    #         print(f'Dat{dat.datnum}: Transition Reset')
    #         dat._reset_transition()
    #     if hasattr(dat, 'Entropy') is False:
    #         print(f'Dat{dat.datnum} did not have Entropy')
    #         continue
    #     if dat.Entropy.version != E.Entropy._Entropy__version:
    #         print(f'Dat{dat.datnum}: Entropy Reset')
    #         dat.Data.entx = dat.Data.entropy_x_2d[:]*dat.Instruments.srs3.sens/10*1e-3*1e-8*1e9
    #         dat.Data.enty = dat.Data.entropy_y_2d[:] * dat.Instruments.srs3.sens / 10 * 1e-3 * 1e-8 * 1e9
    #         dat._reset_entropy()
    #
    #     datdf.update_dat(dat)


