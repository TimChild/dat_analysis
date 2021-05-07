from dataclasses import dataclass
from typing import Optional, Tuple
import dash_html_components as html

import h5py

from src.DatObject.DatHDF import DatHDF
from src.HDF_Util import DatDataclassTemplate, with_hdf_write


@dataclass
class GammaAnalysisParams(DatDataclassTemplate):
    """All the various things that go into calculating Entropy etc"""
    # To save in HDF
    save_name: str

    # For Entropy calculation (and can also determine transition info from entropy dat)
    entropy_datnum: int
    setpoint_start: Optional[float] = 0.005  # How much data to throw away after each heating setpoint change in secs
    entropy_transition_func_name: str = 'i_sense'  # For centering data only if Transition specific dat supplied,
    # Otherwise this also is used to calculate gamma, amplitude and optionally dT
    entropy_fit_width: Optional[float] = None
    entropy_data_rows: tuple = (None, None)  # Only fit between rows specified (None means beginning or end)

    # Integrated Entropy
    force_dt: Optional[float] = None  # 1.11  #  dT for scaling
    # (to determine from Entropy set dt_from_square_transition = True)
    force_amp: Optional[float] = None  # Otherwise determined from Transition Only, or from Cold part of Entropy
    sf_from_square_transition: bool = False  # Set True to determine dT from Hot - Cold theta

    # For Transition only fitting (determining amplitude and gamma)
    transition_only_datnum: Optional[int] = None  # For determining amplitude and gamma
    transition_func_name: str = 'i_sense_digamma'
    transition_center_func_name: Optional[str] = None  # Which fit func to use for centering data
    # (defaults to same as transition_center_func_name)
    transition_fit_width: Optional[float] = None
    force_theta: Optional[float] = None  # 3.9  # Theta must be forced in order to get an accurate gamma for broadened
    force_gamma: Optional[float] = None  # Gamma must be forced zero to get an accurate theta for weakly coupled
    transition_data_rows: Tuple[Optional[int], Optional[int]] = (
        None, None)  # Only fit between rows specified (None means beginning or end)

    # For CSQ mapping, applies to both Transition and Entropy (since they should always be taken in pairs anyway)
    csq_mapped: bool = False  # Whether to use CSQ mapping
    csq_datnum: Optional[int] = None  # CSQ trace to use for CSQ mapping

    def __str__(self):
        return f'Dat{self.entropy_datnum}:\n' \
               f'\tEntropy Params:\n' \
               f'entropy datnum = {self.entropy_datnum}\n' \
               f'setpoint start = {self.setpoint_start}\n' \
               f'transition func = {self.entropy_transition_func_name}\n' \
               f'force theta = {self.force_theta}\n' \
               f'force gamma = {self.force_gamma}\n' \
               f'fit width = {self.entropy_fit_width}\n' \
               f'data rows = {self.entropy_data_rows}\n' \
               f'\tIntegrated Entropy Params:\n' \
               f'from Entropy directly = {self.sf_from_square_transition}\n' \
               f'forced dT = {self.force_dt}\n' \
               f'forced amp = {self.force_amp}\n' \
               f'\tTransition Params:\n' \
               f'transition datnum = {self.transition_only_datnum}\n' \
               f'fit func = {self.transition_func_name}\n' \
               f'center func = {self.transition_center_func_name}\n' \
               f'fit width = {self.transition_fit_width}\n' \
               f'force theta = {self.force_theta}\n' \
               f'force gamma = {self.force_gamma}\n' \
               f'data rows = {self.transition_data_rows}\n' \
               f'\tCSQ Mapping Params:\n' \
               f'Mapping used: {self.csq_mapped}\n' \
               f'csq datnum: {self.csq_datnum}\n'

    def to_dash_element(self):
        return self.__str__().replace('\t', '&nbsp&nbsp&nbsp').replace('\n', '<br>')
        # lines = self.__str__().replace('\t', '    ').split('\n')
        # dash_lines = [[html.P(l), html.Br()] for l in lines]
        # div = html.Div([c for line in dash_lines for c in line])
        # return div


def save_gamma_analysis_params_to_dat(dat: DatHDF, analysis_params: GammaAnalysisParams,
                                      name: str):
    """Save GammaAnalysisParams to suitable place in DatHDF"""
    @with_hdf_write
    def save_params(d: DatHDF):
        analysis_group = d.hdf.hdf.require_group('Gamma Analysis')
        analysis_params.save_to_hdf(analysis_group, name=name)
    save_params(dat)


def load_gamma_analysis_params(dat: DatHDF, name: str) -> GammaAnalysisParams:
    """Load GammaAnalysisParams from DatHDF"""
    with h5py.File(dat.hdf.hdf_path, 'r') as hdf:
        analysis_group = hdf.get('Gamma Analysis')
        analysis_params = GammaAnalysisParams.from_hdf(analysis_group, name=name)
    return analysis_params