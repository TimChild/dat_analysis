from __future__ import annotations
import numpy as np
from typing import List, TYPE_CHECKING, Dict, Optional, Tuple
import logging

from dat_analysis.analysis_tools.general_fitting import FitInfo
from dat_analysis.plotting.plotly import OneD, TwoD

from dat_analysis.dat_object.make_dat import get_dat, get_dats
import dat_analysis.useful_functions as U
from dat_analysis.useful_functions import IgorSaveInfo

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF
    import plotly.graph_objects as go

p1d = OneD(dat=None)
p2d = TwoD(dat=None)


class HeatingPower:
    @staticmethod
    def get_fit(dat: DatHDF):
        """This determines which fit to get from dat (e.g. so can just change this to switch to a different fit)"""
        return dat.Transition.get_fit(name='default')


class AllTraceData(HeatingPower):
    temp_dat_dict: Dict[float, List[DatHDF]]  # Temperature: [List[DatHDF]

    def __init__(self, dats: List[DatHDF]):
        self.temp_dat_dict = self.dats_by_fridge_temp(dats)

        self._single_trace_datas = {}

    @staticmethod
    def dats_by_fridge_temp(dats: List[DatHDF]) -> Dict[float, List[DatHDF]]:
        """set self.dats with dict of Tmc: List[dat] from a flat list of dats where currently
        the temperatures are determined as the closest multiple of 50mK"""
        dat_temp_dict = {dat: U.my_round(dat.Logs.temps.mc * 1000, prec=1, base=50) for dat in dats}
        temps = set(dat_temp_dict.values())
        temp_dat_dict = {}
        for temp in temps:
            temp_dat_dict[temp] = []
            for k, v in dat_temp_dict.items():
                if v == temp:
                    temp_dat_dict[temp].append(k)
        return temp_dat_dict

    def temp_datnum_dict(self) -> Dict[float, List[float]]:
        """Makes a dict of just Fridge temp: List[datnums] most likely for printing"""
        return {k: [dat.datnum for dat in dats] for k, dats in self.temp_dat_dict.items()}

    def get_dats_closest_to_temp(self, temp: float) -> SingleTraceData:
        """Return the dats at the closest temperature to requested. If it is more than 10% off then warn user"""
        v = list(self.temp_dat_dict.keys())[U.get_data_index(list(self.temp_dat_dict.keys()), temp)]
        if v not in self._single_trace_datas.keys():
            if (diff := abs(temp - v)) > temp / 10:
                logging.warning(
                    f'Closest temperature to requested is {v:.2f} which is {diff:.2f} from requested {temp:.2f}')
            dats = self.temp_dat_dict[v]
            self._single_trace_datas[v] = SingleTraceData(dats, temperature=v)
        return self._single_trace_datas[v]

    def heating_power_fig(self) -> go.Figure:
        fig = p1d.figure(xlabel='Bias (nA)', ylabel='Theta (mV)', title='Heating Power')
        for temp in self.temp_dat_dict.keys():
            single_trace_data = self.get_dats_closest_to_temp(temp)
            fig.add_trace(single_trace_data.theta_vs_bias_trace())
        return fig

    def save_to_itx(self, filepath='fig3_heater_power.itx'):
        save_infos = []
        for t in self.temp_dat_dict.keys():
            save_infos.append(self.get_dats_closest_to_temp(t)._get_IgorSaveInfo(trace_name=f'{t}mK_theta_vs_bias'))

        U.save_to_igor_itx(file_path=filepath,
                           xs=[s.x for s in save_infos],
                           datas=[s.data for s in save_infos],
                           names=[s.name for s in save_infos],
                           x_labels=[s.x_label for s in save_infos],
                           y_labels=[s.y_label for s in save_infos],
                           )


class SingleTraceData(HeatingPower):
    def __init__(self, dats: List[DatHDF], temperature: float):
        self.dats = self._sort_dats_by_bias(dats)
        self.temperature = temperature

        self._fits = None

    @property
    def thetas(self) -> List[float]:
        """Get thetas """
        return [fit.best_values.theta for fit in self.fits]

    @property
    def biases(self) -> List[float]:
        """Get biases """
        return [dat.Logs.dacs['HO1/10M'] / 10 for dat in self.dats]

    @property
    def fits(self) -> List[FitInfo]:
        """Temporarily store retrieved fits for speed"""
        if self._fits is None:
            self._fits = [self.get_fit(dat) for dat in self.dats]
        return self._fits

    def theta_vs_bias_trace(self) -> go.Scatter:
        # hover_infos = [
        #     hover_info.DefaultHoverInfos.datnum(),
        #     hover_info.DefaultHoverInfos.xlabel('Bias', lambda dat: dat.Logs.dacs['HO1/10M']/10, units='(nA)'),
        #     hover_info.DefaultHoverInfos.ylabel('Theta', lambda dat: self.get_fit(dat).best_values.theta, units='(mV)'),
        # ]
        #
        # hover_infos = hover_info.HoverInfoGroup(hover_infos)
        #
        # trace = p1d.trace(data=self.thetas, x=self.biases, mode='markers+lines', name=f'{self.temperature:.1f}mK',
        #                   hover_template=hover_infos.template, hover_data=hover_infos.customdata(dats))
        trace = p1d.trace(data=self.thetas, x=self.biases, mode='markers+lines', name=f'{self.temperature:.1f}mK')
        return trace

    @staticmethod
    def _sort_dats_by_bias(dats: List[DatHDF]) -> Tuple[DatHDF]:
        ordered = tuple(U.order_list(dats, [dat.Logs.dacs['HO1/10M'] for dat in dats]))
        return ordered

    def save_to_itx(self, file_name: str = 'fig3_single_heater_power.itx',
                    trace_name: str = 'theta_vs_bias',
                    bias_max: Optional[float] = None):
        """
        Save to .itx file

        Args:
            file_name ():
            trace_name ():
            bias_max (): Max bias to include in saved data (i.e. 3 will save only data with bias < 3nA)

        Returns:

        """
        save_info = self._get_IgorSaveInfo(trace_name, bias_max)
        U.save_to_igor_itx(file_name,
                           xs=[save_info.x],
                           datas=[save_info.data],
                           names=[save_info.name],
                           x_labels=save_info.x_label,
                           y_labels=save_info.y_label)

    def _get_IgorSaveInfo(self, trace_name: str = 'theta_vs_bias',
                          bias_max: Optional[float] = None) -> IgorSaveInfo:
        biases = np.array(self.biases)
        thetas = np.array(self.thetas)
        if bias_max is not None:
            cond = np.where(abs(biases) <= bias_max)
            biases = biases[cond]
            thetas = thetas[cond]
        save_info = IgorSaveInfo(
            x=biases,
            data=thetas,
            name=trace_name,
            x_label='Bias (nA)',
            y_label='Theta (mV)'
        )
        return save_info


# class GenerateNrgFit:
#     """Generates and plots or works with NRG fit of dat for Heater Power
#
#     Not going to use this much as I think it probably makes more sense just to stay with i_sense fitting
#     which works well for weakly coupled and is faster
#     """
#     fit_save_name = 'i_sense_centered'
#
#     def __init__(self, dat: DatHDF):
#         self.dat: DatHDF = dat
#         self.fit: Optional[FitInfo] = None
#
#     def generate_nrg_fit(self) -> FitInfo:
#         """Generate an NRG fit using the centers from default Transition (i_sense) fits and with gamma forced to ~0"""
#         dat = self.dat
#         default_fit = dat.Transition.get_fit(name='default')
#         init_fit_pars = default_fit.params
#         init_fit_pars.add('g', 0.1, False)
#         init_fit_pars.add('occ_lin', 0.000, False, -0.001, 0.001)
#
#         centers = [fit.best_values.mid for fit in dat.Transition.get_row_fits(name='default')]
#         avg_data, avg_x = dat.NrgOcc.get_avg_data(centers=centers,
#                                                   name=self.fit_save_name,
#                                                   return_x=True,
#                                                   check_exists=False, overwrite=False)
#
#         fit = dat.NrgOcc.get_fit(name=self.fit_save_name, initial_params=init_fit_pars, data=avg_data, x=avg_x,
#                                  check_exists=False, overwrite=False)
#         self.fit = fit
#         return fit
#
#     def plot_nrg_fit(self) -> go.Figure:
#         if self.fit_save_name not in self.dat.Transition.fit_names:
#             self.generate_nrg_fit()
#
#         dat = self.dat
#         avg_data, avg_x = dat.NrgOcc.get_avg_data(name=self.fit_save_name, return_x=True)
#         fig = p1d.plot(data=avg_data, x=avg_x, trace_name='Avg Data', title=f'Dat{dat.datnum}: NRG Transition Fit',
#                        mode='lines', xlabel=dat.Logs.xlabel, ylabel='Current (nA)')
#         fit = dat.NrgOcc.get_fit(name=self.fit_save_name)
#         fig.add_trace(p1d.trace(data=fit.eval_fit(avg_x), x=avg_x, mode='lines', name='NRG Fit'))
#         return fig
#


if __name__ == '__main__':
    dat = get_dat(7437)
    dats = get_dats((7437, 7796 + 1))

    all_data = AllTraceData(dats)
    all_data.heating_power_fig().show()
    all_data.save_to_itx()

    single_temp_data = all_data.get_dats_closest_to_temp(100)
    single_temp_data.save_to_itx()

    # GenerateNrgFit(dat).plot_nrg_fit().show()
