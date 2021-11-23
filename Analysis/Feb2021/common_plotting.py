"""
Sep 21 -- A few of the plots used in analysis, very far from a complete list, and probably most are too specific to be
useful again.
Moved useful functions from here.
"""

from __future__ import annotations
from typing import List, Callable, Optional, Union, TYPE_CHECKING

import numpy as np

from dat_analysis.analysis_tools.entropy import dat_integrated_sub_lin
from dat_analysis.plotting.plotly.hover_info import HoverInfo

if TYPE_CHECKING:
    pass


def common_dat_hover_infos(datnum=True,
                           heater_bias=False,
                           fit_entropy_name: Optional[str] = None,
                           fit_entropy=False,
                           int_info_name: Optional[str] = None,
                           output_name: Optional[str] = None,
                           integrated_entropy=False,
                           sub_lin: bool = False,
                           sub_lin_width: Optional[Union[float, Callable]] = None,
                           int_info=False,
                           amplitude=False,
                           theta=False,
                           gamma=False,
                           ) -> List[HoverInfo]:
    """
    Returns a list of HoverInfos for the specified parameters. To do more complex things, append specific
    HoverInfos before/after this.

    Examples:
        hover_infos = common_dat_hover_infos(datnum=True, amplitude=True, theta=True)
        hover_group = HoverInfoGroup(hover_infos)

    Args:
        datnum ():
        heater_bias ():
        fit_entropy_name (): Name of saved fit_entropy if wanting fit_entropy
        fit_entropy ():
        int_info_name (): Name of int_info if wanting int_info or integrated_entropy
        output_name (): Name of SE output to integrate (defaults to int_info_name)
        integrated_entropy ():
        sub_lin (): Whether to subtract linear term from integrated_info first
        sub_lin_width (): Width of transition to avoid in determining linear terms
        int_info (): amp/dT/sf from int_info

    Returns:
        List[HoverInfo]:
    """

    hover_infos = []
    if datnum:
        hover_infos.append(HoverInfo(name='Dat', func=lambda dat: dat.datnum, precision='.d', units=''))
    if heater_bias:
        hover_infos.append(HoverInfo(name='Bias', func=lambda dat: dat.AWG.max(0) / 10, precision='.1f', units='nA'))
    if fit_entropy:
        hover_infos.append(HoverInfo(name='Fit Entropy',
                                     func=lambda dat: dat.Entropy.get_fit(name=fit_entropy_name,
                                                                          check_exists=True).best_values.dS,
                                     precision='.2f', units='kB'), )
    if integrated_entropy:
        if output_name is None:
            output_name = int_info_name
        if sub_lin:
            if sub_lin_width is None:
                raise ValueError(f'Must specify sub_lin_width if subtrating linear term from integrated entropy')
            elif not isinstance(sub_lin_width, Callable):
                sub_lin_width = lambda _: sub_lin_width  # make a value into a function so so that can assume function
            data = lambda dat: dat_integrated_sub_lin(dat, signal_width=sub_lin_width(dat), int_info_name=int_info_name,
                                                      output_name=output_name)
            hover_infos.append(HoverInfo(name='Sub lin width', func=sub_lin_width, precision='.1f', units='mV'))
        else:
            data = lambda dat: dat.Entropy.get_integrated_entropy(
                name=int_info_name,
                data=dat.SquareEntropy.get_Outputs(
                    name=output_name).average_entropy_signal)
        hover_infos.append(HoverInfo(name='Integrated Entropy',
                                     func=lambda dat: np.nanmean(data(dat)[-10:]),
                                     precision='.2f', units='kB'))

    if int_info:
        info = lambda dat: dat.Entropy.get_integration_info(name=int_info_name)
        hover_infos.append(HoverInfo(name='SF amp',
                                     func=lambda dat: info(dat).amp,
                                     precision='.3f',
                                     units='nA'))
        hover_infos.append(HoverInfo(name='SF dT',
                                     func=lambda dat: info(dat).dT,
                                     precision='.3f',
                                     units='mV'))
        hover_infos.append(HoverInfo(name='SF',
                                     func=lambda dat: info(dat).sf,
                                     precision='.3f',
                                     units=''))

    return hover_infos


