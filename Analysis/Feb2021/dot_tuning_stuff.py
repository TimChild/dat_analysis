from src.dat_object.make_dat import get_dat, get_dats
from Analysis.Feb2021.common import calculate_transition_only_fit
from Analysis.Feb2021.common_plotting import transition_trace, transition_fig

from progressbar import progressbar
import plotly.io as pio
from itertools import chain

pio.renderers.default = 'browser'


# datnum_chunks = [list(range(start, start+6)) for start in range(6104, 6165+1, 7)]  # 6 because there are also CSQ scans
# datnum_chunks = [list(range(start, start+6)) for start in range(6167, 6223+1, 7)]  # 6 because there are also CSQ scans
# datnum_chunks = [list(range(start, start+6)) for start in range(6257, 6345+1, 11)]  # 6 because there are also CSQ scans
# datnum_chunks = [list(range(start, start+6)) for start in range(6356, 6389+1, 11)]  # 6 because there are also CSQ scans
datnum_chunks = [list(range(start, start+6)) for start in range(6414, 6424+1, 10)]  # 6 because there are also CSQ scans


if __name__ == '__main__':
    fit_name = 'default'

    all_dats = get_dats(chain(*datnum_chunks))
    for dat in progressbar(all_dats):
        # print(dat.datnum, dat.Logs.comments)
        try:
            calculate_transition_only_fit(dat.datnum, save_name=fit_name, t_func_name='i_sense_digamma',
                                      theta=None, gamma=0, width=None)
        except (TypeError, ValueError):
            print(f'Dat{dat.datnum}: Failed to calculate')

    # param = 'amp'
    param = 'amp/const'
    fig = transition_fig(dats=[get_dat(datnum_chunks[0][0]), get_dat(datnum_chunks[-1][-1])], xlabel='ESC /mV',
                         title_append=' vs ESC for Transition Only scans',
                         param=param)
    for dnums in datnum_chunks:
        all_dats = get_dats(dnums)
        dat = all_dats[0]
        label = f'Dats{all_dats[0].datnum}-{all_dats[-1].datnum}: ' \
                f'ESS={dat.Logs.bds["ESS"]:.1f}mV, ' \
                f'CSS={dat.Logs.bds["CSS"]:.1f}mV'
        all_dats = [dat for dat in all_dats if fit_name in dat.Transition.fit_names]
        fig.add_trace(transition_trace(all_dats, x_func=lambda dat: dat.Logs.fds['ESC'], from_square_entropy=False,
                                       fit_name=fit_name, param=param, label=label, mode='markers+lines'))

    fig.show()
