from src.Scripts.StandardImports import *


if __name__ == '__main__':
    dats = [C.DatHandler.get_dat(num, 'base') for num in range(36, 50)]


    fig, axs = plt.subplots(4,4)
    axs = axs.flatten()

    for dat in dats:
        print(f'Dat{dat.datnum}:\n'
              f'\tWavenames: {dat.Data.data_keys}\n'
              f'\tComments: {dat.Logs.comments}\n')

    for dat, ax in zip(dats, axs):
        data_keys = dat.Data.data_keys
        if 'fd_0adc' in data_keys:
            ax.plot(dat.Data.x_array, dat.Data.fd_0adc, label='fd_0adc')
        elif 'test' in data_keys:
            x = dat.Data.x_array
            data = dat.Data.test
            if data.shape[0] != x.shape[0]:
                print(f'Dat{dat.datnum}: Making new x_array with len[{data.shape[0]}] instead of [{x.shape[0]}]')
                x = np.linspace(x[0], x[-1], data.shape[0])
            ax.plot(x, data, label='test')
        elif 'test_2d_RAW' in data_keys:
            x = dat.Data.x_array
            y = dat.Data.y_array
            data = dat.Data.test_2d_RAW
            if data.shape[1] != x.shape[0]:
                print(f'Dat{dat.datnum}: Making new x_array with len[{data.shape[1]}] instead of [{x.shape[0]}]')
                x = np.linspace(x[0], x[-1], data.shape[1])
            PF.display_2d(x, y, data, ax)

        PF.ax_setup(ax, f'Dat{dat.datnum}\n{dat.Logs.comments[0:51]}', dat.Logs.x_label, dat.Logs.y_label, fs=6)