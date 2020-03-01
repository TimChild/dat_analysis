from src.Scripts.StandardImports import *




def plot_entropy(dat):
    fig, axs = PF.reuse_plots(3)
    axs[0].plot(dat.Data.x_array, dat.Entropy.entrav)
    PF._optional_plotting_args(axs[0], x_label=dat.Logs.x_label, y_label='Entropy Signal')
    
    dat.display(dat.Entropy.entr, ax=axs[1])
    
    axs[2].scatter(dat.Data.y_array, dat.Entropy.fit_values.dSs)
    PF._optional_plotting_args(axs[2], x_label='Repeats', y_label='Entropy/kB')
    
    fig.suptitle(f'Dat{dat.datnum}')
    
def get_dat(datnum):
    return make_dat_standard(datnum, dattypes=['transition', 'entropy'])

if __name__ == '__main__':
    dat = get_dat(318)
    plot_entropy(dat)

    