from __future__ import annotations
import lmfit as lm
import os
from typing import TYPE_CHECKING, Optional, Dict, List, Callable, Any, Union, Iterable, Tuple
import copy
import h5py
import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
import logging

from src.hdf_util import with_hdf_write, with_hdf_read, DatDataclassTemplate, params_from_HDF, params_to_HDF, \
    NotFoundInHdfError
from src.dat_object.Attributes.DatAttribute import FittingAttribute, FitPaths
import src.core_util as CU

if TYPE_CHECKING:
    from src.dat_object.dat_hdf import DatHDF
    from src.dat_object.Attributes import AWG
    from src.analysis_tools.general_fitting import FitInfo


logger = logging.getLogger(__name__)

_NOT_SET = object()


class SquareEntropy(FittingAttribute):
    version = '2.1.0'
    group_name = 'Square Entropy'
    description = 'Working with square entropy stuff to generate entropy signal for dat.Entropy (which looks for ' \
                  'dat.SquareEntropy.get_entropy_signal()'

    """
    Version History:
        2.1 -- Added more control over transition fits to v0 part of data. Added fit params to process params
    """

    @property
    def DEFAULT_DATA_NAME(self) -> str:
        return 'i_sense'

    def get_default_params(self, x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None) -> Union[
        List[lm.Parameters], lm.Parameters]:
        if self._which_fit == 'transition':
            return self.dat.Transition.get_default_params(x=x, data=data)
        elif self._which_fit == 'entropy':
            return self.dat.Entropy.get_default_params(x=x, data=data)
        else:
            raise ValueError(f'{self._which_fit} is not recognized')

    def get_default_func(self) -> Callable[[Any], float]:
        if self._which_fit == 'transition':
            return self.dat.Transition.get_default_func()
        elif self._which_fit == 'entropy':
            return self.dat.Entropy.get_default_func()
        else:
            raise ValueError(f'{self._which_fit} is not recognized')

    def default_data_names(self) -> List[str]:
        return ['x', 'i_sense']

    def get_centers(self) -> List[float]:
        return [f.best_values.mid for f in self.get_row_fits(name='cold',
                                                             which_fit='transition', transition_part='cold',
                                                             check_exists=False)]

    def initialize_additional_FittingAttribute_minimum(self):
        pass

    def _get_fit_parent_group_name(self, which: str, row: int = 0, which_fit: str = 'transition') -> str:
        """Get path to parent group of avg or row fit"""
        if which_fit == 'transition':
            if which == 'avg':
                group_name = '/' + '/'.join((self.group_name, 'Transition', 'Avg Fits'))
            elif which == 'row':
                group_name = '/' + '/'.join((self.group_name, 'Transition', 'Row Fits', str(row)))
            else:
                raise ValueError(f'{which} not in ["avg", "row"]')
        elif which_fit == 'entropy':
            if which == 'avg':
                group_name = '/' + '/'.join((self.group_name, 'Entropy', 'Avg Fits'))
            elif which == 'row':
                group_name = '/' + '/'.join((self.group_name, 'Entropy', 'Row Fits', str(row)))
            else:
                raise ValueError(f'{which} not in ["avg", "row"]')
        else:
            raise ValueError(f'{which_fit} is not recognized')
        return group_name

    @with_hdf_write
    def _set_default_fit_groups(self):
        for name in ['Transition', 'Entropy']:
            group = self.hdf.group.require_group(name)
            group.require_group('Avg Fits')
            group.require_group('Row Fits')

    def _get_FitPaths(self):
        """Square Entropy has both Entropy and Transition fits, so need to handle self.fit_paths differently"""
        return None

    @with_hdf_read
    def get_fit_paths(self, which: str = 'transition') -> FitPaths:
        """Alternative to self.fit_paths specific to SquareEntropy which allows for specifying whether looking for
        'transition' or 'entropy' fits """
        avg_fit_group = self.hdf.get(self._get_fit_parent_group_name(which='avg', which_fit=which))
        row_fit_group = self.hdf.get(os.path.split(self._get_fit_parent_group_name('row', 0, which_fit=which))[0])
        if not avg_fit_group or not row_fit_group:
            raise NotFoundInHdfError
        return FitPaths.from_groups(avg_fit_group=avg_fit_group, row_fit_group=row_fit_group)

    @property
    def fit_paths(self):
        """Can be either entropy or transition fits. This is so that other prebuilt code works, but not super
        convenient to use otherwise"""
        if self._which_fit == 'transition':
            if self._fit_paths_transition is None:
                self._fit_paths_transition = self.get_fit_paths(which=self._which_fit)
            return self._fit_paths_transition
        elif self._which_fit == 'entropy':
            if self._fit_paths_entropy is None:
                self._fit_paths_entropy = self.get_fit_paths(which=self._which_fit)
            return self._fit_paths_entropy
        else:
            raise NotImplementedError
        # return self.get_fit_paths(which=self._which_fit)

    def get_fit_names(self, which: str = 'transition') -> List[str]:
        """Easier way to ask for either transition or entropy fit names from other functions since it's hard to set the
        private self._which_fit variable in order to get the right names from self.fit_names"""
        return [k[:-4] for k in self.get_fit_paths(which=which).avg_fits]

    @fit_paths.setter
    def fit_paths(self, value):
        """Doesn't make sense to set it to a single value, so just pass here to prevent errors being raised, but
        really the value is set in self.fit_paths property"""
        logger.debug(f'Ignoring call to set fit_paths for SquareEntropy because SquareEntropy is special and handles'
                     f'this attribute as a property')
        pass

    def __init__(self, dat: DatHDF):
        self._fit_paths_transition = None
        self._fit_paths_entropy = None
        super().__init__(dat)
        self._Outputs: Dict[str, Output] = {}
        self._square_awg = None
        self._which_fit = 'transition'  # Private attribute so that self.get_default_func()/params() can know
        # which type of fitting is happening

    @property
    def square_awg(self) -> AWG:
        """
        Copy of the normal AWG but where AWs are forced to be 4 points only (i.e. groups in possible ramps into the main 4 parts)
        Assumes equal length to each part

        Note: square_awg still has access to HDF. Tried to this from overwriting things, but maybe it's still possible so be careful.
        Returns:
        """
        if not self._square_awg:
            awg = copy.copy(self.dat.AWG)

            # Try to prevent this overwriting real AWG info in HDF by removing set methods
            awg.set_group_attr = None
            awg.set_data = None
            awg.set_data_descriptor = None

            awg._AWs = {k: _force_four_point_AW(aw) for k, aw in awg.AWs.items()}
            self._square_awg = awg
        return self._square_awg

    @property
    def default_Input(self) -> Input:
        return self.get_Inputs()

    @property
    def default_ProcessParams(self) -> ProcessParams:
        return self.get_ProcessParams()

    @property
    def default_Output(self) -> Output:
        return self.get_Outputs(check_exists=False)

    @property
    def x(self):
        """Default x array for Square Entropy (i.e. x per DAC step)"""
        return self.default_Output.x

    @property
    def entropy_signal(self) -> np.ndarray:
        """Default entropy signal for Square Entropy"""
        return self.default_Output.entropy_signal

    @property
    def avg_entropy_signal(self) -> np.ndarray:
        """Default averaged entropy signal for Square Entropy"""
        return self.default_Output.average_entropy_signal

    @property
    def avg_data(self):
        """Quick access for DEFAULT avg_data ONLY"""
        return self.avg_entropy_signal

    @property
    def avg_x(self):
        """Quick access for DEFAULT avg_x ONLY (although this likely be the same all the time)"""
        return self.x

    @property
    def avg_data_std(self):
        """Quick access for DEFAULT avg_data_std ONLY"""
        raise NotImplementedError(f'Not implemented getting std error of entropy signal yet')

    def get_Inputs(self,
                   name: Optional[str] = None,
                   x_array: Optional[np.ndarray] = None,
                   i_sense: Optional[np.ndarray] = None,
                   num_steps: Optional[int] = None,
                   num_cycles: Optional[int] = None,
                   setpoint_lengths: Optional[List[int]] = None,
                   full_wave_masks: Optional[np.ndarray] = None,
                   centers: Optional[np.ndarray] = None,
                   avg_nans: Optional[bool] = None,
                   save_name: Optional[str] = None) -> Input:
        """
        Gathers together necessary inputs for processing square wave info. Anything not specified is gathered by
        defaults.
        If a name is specified and that entry has been saved in the HDF, it will use that as a starting point and
        change anything else that is specified.
        Use save_name to save the Input to HDF, otherwise it WILL NOT be stored automatically

        Returns:
        Args:
            name (): Look for stored Inputus with this name
            x_array (): Original x_array
            i_sense (): Charge sensor data (1D or 2D)
            num_steps (): Number of DAC steps
            num_cycles (): Number of full square wave cycles per DAC step
            setpoint_lengths (): Number of readings per setpoint of Square Wave
            full_wave_masks (): Mask for chunking data into setpoints
            centers (): Center positions to use for averaging data (if left as None then v0 parts of i_sense data will
                be fit with i_sense and that will be used. Good but slow). !! Saved in Outputs.centers_used. !!
            avg_nans (): Whether to include columns which contain NaNs in averaging (generally safer not to but a single
                bad row will mean that almost everything is thrown out)
            save_name (): Name to save into HDF with (i.e. to be accessed with name later)

        Returns:
            (Inputs): The Inputs which go into processing square wave info
        """

        inp: Optional[Input] = None
        if name:
            inp = self._get_saved_Inputs(name)
            if inp:
                for k, v in {'x_array': x_array, 'i_sense': i_sense, 'num_steps': num_steps, 'num_cycles': num_cycles,
                             'setpoint_lengths': setpoint_lengths, 'full_wave_masks': full_wave_masks,
                             'centers': centers,
                             'avg_nans': avg_nans}.items():
                    if v is not None:  # Only overwrite things that have changed
                        setattr(inp, k, v)

        if not inp:  # If not found above then create a new inp
            if any([v is None for v in [num_steps, num_cycles, setpoint_lengths, full_wave_masks]]):
                awg = self.square_awg
                if not num_steps:
                    num_steps = awg.info.num_steps
                if not num_cycles:
                    num_cycles = awg.info.num_cycles
                if not setpoint_lengths:
                    setpoint_lengths = _force_four_point_AW(awg.AWs[0])[1]  # Assume first AW works here
                if not full_wave_masks:
                    full_wave_masks = awg.get_full_wave_masks(0)  # Assume first AW works here

            if x_array is None:
                x_array = self.get_data('x')
            if i_sense is None:
                i_sense = self.get_data('i_sense')
            if centers is None:
                pass  # If pass in None, then centers are calculated from v0 parts of i_sense
            if avg_nans is None:
                avg_nans = False  # Safer to throw away columns with nans in when averaging in general
            inp = Input(x_array=x_array, i_sense=i_sense, num_steps=num_steps, num_cycles=num_cycles,
                        setpoint_lengths=setpoint_lengths, full_wave_masks=full_wave_masks, centers=centers,
                        avg_nans=avg_nans)

        if save_name:
            self.set_group_attr(save_name, inp, group_name='/'.join([self.group_name, 'Inputs']), DataClass=Input)
        return inp

    def _get_saved_Inputs(self, name):
        inp = self.get_group_attr(name, check_exists=True, group_name='/'.join([self.group_name, 'Inputs']),
                                  DataClass=Input)
        return inp

    def get_ProcessParams(self,
                          name: Optional[str] = None,
                          setpoint_start: Optional[int] = None,
                          setpoint_fin: Optional[int] = None,
                          cycle_start: Optional[int] = None,
                          cycle_fin: Optional[int] = None,
                          transition_fit_func: Optional[Callable] = None,
                          transition_fit_params: Optional[lm.Parameters] = None,
                          save_name: Optional[str] = None,
                          ) -> ProcessParams:
        """
        Gathers together necessary ProcessParams info. Similar to get_Inputs.
        If a name is specified and has been saved in HDF, that will be used as a starting point, and anything else
        specified will be changed (NOTE: passing in None will not overwrite things, use 0 or e.g. len(setpoint) to
        refer to beginning or end of array in that case).
        Use save_name to save the ProcessParams to HDF

        Args:
            name ():  Look for stored ProcessParams with this name
            setpoint_start (): Where to start averaging data each setpoint (index position)
            setpoint_fin (): Where to finish averaging data each setpoint (index position)
            cycle_start (): Where to start averaging cycles each DAC step
            cycle_fin (): Where to finish averaging cycles each DAC step
            transition_fit_func (): Optional Function to use for fitting v0 part of data for centering
            transition_fit_params (): Optional Params to use for fitting v0 part of data for centering
            save_name (): Name to save under in HDF

        Returns:
            (ProcessParams): Filled ProcessParams

        """

        def check_setpoints():
            for sp in [setpoint_start, setpoint_fin]:
                if sp is not None:
                    if not isinstance(sp, int):
                        raise TypeError(f'{sp} is not of type {int} or None. This should be a data index')

        check_setpoints()

        pp: Optional[ProcessParams] = None
        if name:
            pp = self._get_saved_ProcessParams(name)

        if not pp:
            # None defaults are what I want here anyway
            pp = ProcessParams(setpoint_start=setpoint_start, setpoint_fin=setpoint_fin, cycle_start=cycle_start,
                               cycle_fin=cycle_fin,
                               transition_fit_func=transition_fit_func, transition_fit_params=transition_fit_params)
        else:
            if setpoint_start:
                pp.setpoint_start = setpoint_start
            if setpoint_fin:
                pp.setpoint_fin = setpoint_fin
            if cycle_start:
                pp.cycle_start = cycle_start
            if cycle_fin:
                pp.cycle_fin = cycle_fin
            if transition_fit_func:
                pp.transition_fit_func = transition_fit_func
            if transition_fit_params:
                pp.transition_fit_params = transition_fit_params

        if save_name:
            self.set_group_attr(save_name, pp, group_name='/'.join([self.group_name, 'ProcessParams']),
                                DataClass=ProcessParams)
        return pp

    def _get_saved_ProcessParams(self, name: str):
        pp: ProcessParams = self.get_group_attr(name, check_exists=True,
                                                group_name='/'.join([self.group_name, 'ProcessParams']),
                                                DataClass=ProcessParams)
        return pp

    def get_row_only_output(self, name: Optional[str] = None, inputs: Optional[Input] = None,
                            process_params: Optional[ProcessParams] = None,
                            check_exists=True, calculate_only=False,
                            overwrite=False) -> Output:
        """For setting Outputs without calculating centers/average first... gets a saved Output like normal (i.e.
        may contain avg if previously saved with it)."""
        if name is not None and name in self.Output_names() and overwrite is False and calculate_only is False:
            return self.get_Outputs(name=name, check_exists=True)
        else:
            if check_exists is True and calculate_only is False:
                raise NotFoundInHdfError(f'{name} not found in Dat{self.dat.datnum}')

        # Otherwise from here calculate and save
        if not inputs:
            inputs = self.get_Inputs()
        if not process_params:
            process_params = self.get_ProcessParams()

        per_row_out = process_per_row_parts(inputs, process_params)
        if not calculate_only and name is not None:
            self._save_Outputs(name, per_row_out)
        return per_row_out

    def get_Outputs(self, name: str = 'default', inputs: Optional[Input] = None,
                    process_params: Optional[ProcessParams] = None,
                    calculate_only: bool = False,
                    overwrite=False, check_exists=_NOT_SET) -> Output:
        """
        Either looks for saved Outputs in HDF file, or generates new Outputs given Inputs and/or ProcessParams.

        If <name> AND Inputs, ProcessParams passed in and <name> already exists, WILL NOT overwrite unless 'overwrite'
        is set True. Otherwise Outputs will be saved under <name>

        Args:
            name (): Name to look for / save under
            inputs (): Input data for calculating Outputs
            process_params (): ProcessParams for calculating Outputs
            calculate_only: Calculate only, do not save anything (may load existing data)
            overwrite (bool): If False, previously calculated is returned if exists, otherwise overwritten
            check_exists (bool): If True, will only load an existing output, will raise NotFoundInHDFError otherwise

        Returns:
            (Outputs): All the various data after processing

        """
        if check_exists is _NOT_SET and inputs is None and process_params is None:  # Probably trying to load saved
            check_exists = True

        if name is None:
            name = 'default'

        if not calculate_only:
            if not overwrite:
                if name in self.Output_names():
                    out = self._get_saved_Outputs(name)
                    return out  # No need to go further if found
            if check_exists is True:
                raise NotFoundInHdfError(f'{name} not found as saved SE.Output of dat{self.dat.datnum}')

        if not inputs:
            inputs = self.get_Inputs()
        if not process_params:
            process_params = self.get_ProcessParams()

        per_row_out = process_per_row_parts(inputs, process_params)

        if inputs.centers is None and not calculate_only:
            all_fits = self.get_row_fits(name=name, initial_params=process_params.transition_fit_params,
                                         fit_func=process_params.transition_fit_func,
                                         data=per_row_out.cycled, x=per_row_out.x,
                                         check_exists=False, overwrite=overwrite,
                                         which_fit='transition', transition_part='cold',
                                         )
            centers = centers_from_fits(all_fits)
        elif inputs.centers is None and calculate_only:
            all_fits = self.get_row_fits(name=name, check_exists=True)
            centers = centers_from_fits(all_fits)
        else:
            centers = inputs.centers

        out = process_avg_parts(partial_output=per_row_out, input_info=inputs, centers=centers)

        if not calculate_only:
            if inputs.centers is None:
                # Calculate average Transition fit because it's fast and then it matches with the row fits
                self.get_fit(x=out.x, data=out.averaged,
                             fit_func=process_params.transition_fit_func,
                             initial_params=process_params.transition_fit_params,
                             which_fit='transition',
                             transition_part='cold',
                             fit_name=name,
                             which='avg',
                             check_exists=False,
                             overwrite=overwrite)
            self._save_Outputs(name, out)
        return out

    def get_row_fits(self, name: Optional[str] = None,
                     initial_params: Optional[lm.Parameters] = None,
                     fit_func: Optional[Callable] = None,
                     data: Optional[np.ndarray] = None,
                     x: Optional[np.ndarray] = None,
                     check_exists=True,
                     overwrite=False,
                     which_fit: str = 'transition',
                     transition_part: Union[str, int] = 'cold') -> List[FitInfo]:
        """Convenience function for calling get_fit for each row"""
        if data is None:
            data = [None] * self.data.shape[0]
        return [self.get_fit(which='row', row=i, fit_name=name,
                             initial_params=initial_params, fit_func=fit_func,
                             data=row, x=x,
                             check_exists=check_exists,
                             overwrite=overwrite,
                             which_fit=which_fit,
                             transition_part=transition_part) for i, row in enumerate(data)]

    def get_fit(self, which: str = 'avg',
                row: int = 0,
                fit_name: Optional[str] = None,
                initial_params: Optional[lm.Parameters] = None,
                fit_func: Optional[Callable] = None,
                output_name: Optional[str] = None,
                data: Optional[np.ndarray] = None,
                x: Optional[np.ndarray] = None,
                calculate_only: bool = False,
                check_exists=True,
                overwrite=False,
                which_fit: str = 'transition',
                transition_part: Union[int, str] = 'cold') -> FitInfo:
        """
        Convenience for calling get_fit of self.dat.Entropy or self.dat.Transition.
        Note: All fits are saved in Entropy/Transition respectively.

        For Transition, this also makes it easy to select which part of the wave to fit to

        Args:
            which (): avg or row (defaults to avg)
            row (): row num to return (defaults to 0)
            fit_name (): name to save/load fit (defaults to default)
            initial_params (): Optional initial params for fitting
            fit_func (): Optional fit func for fitting (defaults same as Entropy/Transition)
            output_name (): Optional name of saved output to use as data for fitting
            data (): Optional data override for fitting
            x (): Optional override of x axis for fitting
            calculate_only (): If True, do not load or save, just calculate
            check_exists (): Whether to raise an error if fit isn't already saved or just to calculate and save fit
            overwrite (): Whether an existing fit should be overwritten
            which_fit (): Which of Transition or Entropy to fit for (defaults to transition)
            transition_part (): If choosing transition fit, which part of transition to fit (defaults to cold)
                Accepts: cold, hot, vp, vm, 0, 1, 2, 3

        Returns:
            (FitInfo): Requested Fit

        """
        def get_entropy_data() -> np.ndarray:
            if which == 'avg':
                d = self.default_Output.average_entropy_signal
            elif which == 'row' and isinstance(row, int):
                d = self.default_Output.entropy_signal[row]
            else:
                raise ValueError(f'which: {which}, row: {row} is not valid')
            return d

        if which_fit.lower() == 'transition':
            self._which_fit = 'transition'
            if check_exists is False or calculate_only:
                if data is None:
                    data = self.get_transition_part(name=output_name, part=transition_part, which=which, row=row,
                                                    existing_only=True)  # Pretty sure I only want existing data here
                    # data = get_transition_data()
                elif data.shape[0] == 4 and data.ndim == 2:
                    data = np.mean(data[get_transition_parts(part=transition_part), :], axis=0)

                if x is None:
                    x = self.get_Outputs(name=output_name, check_exists=True).x

        elif which_fit.lower() == 'entropy':
            self._which_fit = 'entropy'
            if check_exists is False:  # TODO: What to do if calculate_only here...
                if data is None:
                    data = get_entropy_data()

        else:
            raise ValueError(f'{which_fit} not recognized, must be in ["entropy", "transition"]')

        return super().get_fit(which=which, row=row, name=fit_name, initial_params=initial_params, fit_func=fit_func,
                               data=data, x=x,
                               calculate_only=calculate_only, check_exists=check_exists, overwrite=overwrite)

    def get_transition_part(self, name: Optional[str] = None, part: str = 'cold', data: Optional[np.ndarray] = None,
                            which: str = 'avg', row: Optional[int] = None,
                            inputs: Optional[Input] = None, process_params: Optional[ProcessParams] = None,
                            overwrite=False, existing_only=True) -> np.ndarray:
        """
        Convenience method for getting parts of transition data from SquareEntropy measurement
        Or for getting parts of 'data' passed in.

        Args:
            name (): Name of saved Output
            part (): cold, hot, 0, 1, 2, 3, vp, vm
            data (): Optional data to use instead of trying to load an output
            average_data (): Whether to return the average data, or each row of data
            inputs (): Options for get_Output
            process_params (): Options for get_Output
            overwrite (): Options for get_Output
            existing_only (): Options for get_Output

        Returns:

        """
        assert which in ['avg', 'row']
        if name is None:
            name = 'default'

        if data is None:
            out = self.get_Outputs(name=name, inputs=inputs, process_params=process_params,
                                   overwrite=overwrite, check_exists=existing_only)
            if which == 'avg':
                data = out.averaged
            elif which == 'row':
                data = out.cycled[row]

        assert data.shape[-2] == 4  # If not 4, then it isn't square wave transition data

        parts = get_transition_parts(part=part)

        if which == 'avg':
            if data.ndim == 3:
                data = np.mean(data, axis=0)
            data = np.mean(data[parts, :], axis=0)
            return data
        elif which == 'row':  # If row == None, want to return all rows, hence negative axis indexing
            data = np.take(data, parts, axis=-2)
            data = np.mean(data, axis=-2)
            return data
        else:
            raise NotImplementedError

    @with_hdf_read
    def Output_names(self):
        """Get names of saved Outputs in HDF"""
        group = self.hdf.group.get('Outputs')
        return list(group.keys())  # Assume everything in Outputs is an Output

    def _get_saved_Outputs(self, name):
        gn = '/'.join([self.group_name, 'Outputs'])
        if name not in self._Outputs:
            self._Outputs[name] = self.get_group_attr(name, check_exists=True, group_name=gn, DataClass=Output)
        return self._Outputs[name]

    def _save_Outputs(self, name: str, out: Output):
        gn = '/'.join([self.group_name, 'Outputs'])
        self._Outputs[name] = out
        self.set_group_attr(name, out, group_name=gn, DataClass=Output)  # Always should be saving if getting to here

    def initialize_minimum(self):
        super().initialize_minimum()
        # self._set_default_data_descriptors()
        self._make_groups()
        self.initialized = True

    @with_hdf_write
    def _make_groups(self):
        self.hdf.group.require_group('Inputs')
        self.hdf.group.require_group('ProcessParams')
        self.hdf.group.require_group('Outputs')

    def clear_caches(self):
        self._Outputs = {}
        self._square_awg = None


def centers_from_fits(fits: Iterable[FitInfo]) -> np.ndarray:
    return np.array([fit.best_values.mid for fit in fits])


def get_transition_parts(part: str) -> Union[tuple, int]:
    if isinstance(part, str):
        if part == 'cold':
            parts = (0, 2)
        elif part == 'hot':
            parts = (1, 3)
        elif part.lower() == 'vp':
            parts = (1,)
        elif part.lower() == 'vm':
            parts = (3,)
        else:
            raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    elif isinstance(part, int):
        parts = part
    else:
        raise ValueError(f'{part} not recognized. Should be in ["hot", "cold", "vp", "vm"]')
    return parts


@dataclass
class Input(DatDataclassTemplate):
    x_array: np.ndarray
    i_sense: np.ndarray
    num_steps: int
    num_cycles: int
    setpoint_lengths: List[int]
    full_wave_masks: np.ndarray
    centers: Optional[np.ndarray] = None
    avg_nans: bool = False


@dataclass
class ProcessParams(DatDataclassTemplate):
    setpoint_start: Optional[int]  # Index to start averaging for each setpoint
    setpoint_fin: Optional[int]  # Index to stop averaging for each setpoint
    cycle_start: Optional[int]  # Index to start averaging cycles
    cycle_fin: Optional[int]  # Index to stop averaging cycles

    transition_fit_func: Optional[Callable]  # Fit function (not stored in HDF, only set based on fit_name)
    transition_fit_params: Optional[lm.Parameters]  # Params to use for fitting v0 part of data

    def __post_init__(self):
        self.transition_fit_func_name: Optional[str]  # String name of fit_func (e.g. 'i_sense' or 'i_sense_digamma')
        if self.transition_fit_func is not None:
            self.transition_fit_func_name = self.transition_fit_func.__name__
        else:
            self.transition_fit_func_name = None

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return ['transition_fit_func', 'transition_fit_params']

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        import src.dat_object.Attributes.Transition as T
        fit_name = dc_group.get('transition_fit_func_name')
        if fit_name is None or fit_name == 'i_sense':
            fit_func = T.i_sense
        elif fit_name == 'i_sense_digamma':
            fit_func = T.i_sense_digamma
        elif fit_name == 'i_sense_digamma_quad':
            fit_func = T.i_sense_digamma_quad
        else:
            logger.warning(f'{fit_name} not recognized. fit_func returned as T.i_sense')
            fit_func = T.i_sense

        pars_group = dc_group.get('transition_fit_params')
        if pars_group is not None and pars_group.attrs.get('description') == 'Single Parameters of fit':
            fit_params = params_from_HDF(pars_group, initial=True)
        else:
            fit_params = None

        return dict(transition_fit_func=fit_func, transition_fit_params=fit_params)

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        if self.transition_fit_params is not None:
            pars_group = dc_group.require_group('transition_fit_params')
            params_to_HDF(self.transition_fit_params, pars_group)


@dataclass
class Output(DatDataclassTemplate):
    # Data that will be calculated
    x: np.ndarray = field(default=None, repr=False)  # x_array with length of num_steps (for cycled, averaged, entropy)
    chunked: np.ndarray = field(default=None, repr=False)  # Data broken in to chunks based on AWG (just plot
    # raw_data on orig_x_array)
    setpoint_averaged: np.ndarray = field(default=None, repr=False)  # Setpoints averaged only
    setpoint_averaged_x: np.ndarray = field(default=None, repr=False)  # x_array for setpoints averaged only
    cycled: np.ndarray = field(default=None, repr=False)  # setpoint averaged and then cycles averaged data
    averaged: np.ndarray = field(default=None, repr=False)  # setpoint averaged, cycle_avg, then averaged in y

    centers_used: np.ndarray = None
    entropy_signal: np.ndarray = field(default=None, repr=False)  # 2D Entropy signal data
    average_entropy_signal: np.ndarray = field(default=None, repr=False)  # Averaged Entropy signal

    # Store whatever process params were used in here since very relevant to what the output shows.
    # Note: Input is very expensive to store, and does not change much at all, so not being stored in here.
    process_params: ProcessParams = field(default=None)

    @staticmethod
    def ignore_keys_for_hdf() -> Optional[Union[str, List[str]]]:
        return 'process_params'

    def additional_save_to_hdf(self, dc_group: h5py.Group):
        if self.process_params is not None:
            self.process_params.save_to_hdf(dc_group, name='process params')

    @staticmethod
    def additional_load_from_hdf(dc_group: h5py.Group) -> Dict[str, Any]:
        ret = {}
        if 'process params' in dc_group.keys():
            ret['process_params'] = ProcessParams.from_hdf(dc_group, name='process params')
        return ret

    def transition_part(self, which: str = 'cold') -> np.ndarray:
        parts = get_transition_parts(which)
        return np.nanmean(self.averaged[parts, :], axis=0)


def process_per_row_parts(input_info: Input, process_pars: ProcessParams) -> Output:
    """
    Does processing of Input_info using process_pars up to averaging the cycles of data, but does NOT average the data
    because that requires center positions of transition which can require transition fits that should be saved in the dat.
    Args:
        input_info ():
        process_pars ():

    Returns:
        (Output): Partially filled Output
    """
    output = Output()
    inp = input_info
    pp = process_pars
    output.process_params = pp

    # Calculate true x_array (num_steps)
    output.x = np.linspace(inp.x_array[0], inp.x_array[-1], inp.num_steps)

    # Get chunked data (setpoints, ylen, numsteps, numcycles, splen)
    output.chunked = chunk_data(inp.i_sense, full_wave_masks=inp.full_wave_masks, setpoint_lengths=inp.setpoint_lengths,
                                num_steps=inp.num_steps, num_cycles=inp.num_cycles)

    # Average setpoints of data ([ylen], setpoints, numsteps, numcycles)
    output.setpoint_averaged = average_setpoints(output.chunked, start_index=pp.setpoint_start,
                                                 fin_index=pp.setpoint_fin)
    output.setpoint_averaged_x = np.linspace(inp.x_array[0], inp.x_array[-1],
                                             inp.num_steps * inp.num_cycles)

    # Averaged cycles ([ylen], setpoints, numsteps)
    output.cycled = average_cycles(output.setpoint_averaged, start_cycle=pp.cycle_start, fin_cycle=pp.cycle_fin)

    # Per Row Entropy signal
    output.entropy_signal = entropy_signal(np.moveaxis(output.cycled, 1, 0))  # Moving setpoint axis to be first

    return output


def process_avg_parts(partial_output: Output, input_info: Input, centers: np.ndarray) -> Output:
    """
    Finishes off
    Args:
        partial_output ():
        input_info ():
        centers (): The center positions to use for averaging. If None, data will be centered with a default transition fit

    Returns:
        (Output): Filled output (i.e. including averaged data and ent
    """
    inp = input_info
    out = partial_output
    # Center and average 2D data or skip for 1D
    out.x, out.averaged, out.centers_used = average_2D(out.x,
                                                       out.cycled,
                                                       centers=centers,
                                                       avg_nans=inp.avg_nans)

    # region Use this if want to start shifting each heater setpoint of data left or right
    # Align data
    # output.x, output.averaged = align_setpoint_data(xs, output.averaged, nx=None)
    # endregion

    # Avg Entropy signal
    out.average_entropy_signal = entropy_signal(out.averaged)

    return out


def _force_four_point_AW(aw: np.ndarray):
    """
    Takes an single AW and returns an AW with 4 setpoints
    Args:
        aw (np.ndarray):

    Returns:
        np.ndarray: AW with only 4 setpoints but same length as original
    """
    aw = np.asanyarray(aw)
    assert aw.ndim == 2
    full_len = np.sum(aw[1])
    assert full_len % 4 == 0
    new_aw = np.ndarray((2, 4), np.float32)

    # split Setpoints/lens into 4 sections
    for i, aw_chunk in enumerate(np.reshape(aw.swapaxes(0, 1), (4, -1, 2)).swapaxes(1, 2)):
        sp = aw_chunk[0, -1]  # Last value of chunk (assuming each setpoint handles it's own ramp)
        length = np.sum(aw_chunk[1])
        new_aw[0, i] = sp
        new_aw[1, i] = length
    return new_aw


# region Processing functions from I_sense to Entropy
"""All the functions for processing I_sense data into the various steps of square wave heated data"""


def chunk_data(data, full_wave_masks: np.ndarray, setpoint_lengths: List[int], num_steps: int, num_cycles: int) -> List[
    np.ndarray]:
    """
    Breaks up data into chunks which make more sense for square wave heating datasets.
    Args:
        data (np.ndarray): 1D or 2D data (full data to match original x_array).
            Note: will return with y dim regardless of 1D or 2D

    Returns:
        List[np.ndarray]: Data broken up into chunks [setpoints, np.ndarray(ylen, num_steps, num_cycles, sp_len)].

            NOTE: Has to be a list returned and not a ndarray because sp_len may vary per steps

            NOTE: This is the only step where setpoints should come first, once sp_len binned it should be ylen first
    """
    masks = full_wave_masks
    zs = []
    for mask, sp_len in zip(masks, setpoint_lengths):
        sp_len = int(sp_len)
        z = np.atleast_2d(data)  # Always assume 2D data
        zm = z * mask  # Mask data
        zm = zm[~np.isnan(zm)]  # remove blanks
        zm = zm.reshape(z.shape[0], num_steps, num_cycles, sp_len)
        zs.append(zm)
    return zs


def average_setpoints(chunked_data, start_index=None, fin_index=None):
    """ Averages last index of AWG data passed in from index s to f.

    Args:
        chunked_data (List[np.ndarray]): List of datas chunked nicely for AWG data.
            dimensions (num_setpoints_per_cycle, (len(y), num_steps, num_cycles, sp_len))
        start_index (Union(int, None)): Start index to average in each setpoint chunk
        fin_index (Union(int, None)): Final index to average to in each setpoint chunk (can be negative)

    Returns:
        np.ndarray: Array of zs with averaged last dimension. ([ylen], setpoints, num_steps, num_cycles)
        Can be an array here because will always have 1 value per
        averaged chunk of data (i.e. can't have different last dimension any more)
    """

    assert np.all([arr.ndim == 4 for arr in chunked_data])  # Assumes [setpoints, (ylen, num_steps, num_cycles, sp_len)]
    nz = []
    for z in chunked_data:
        z = np.moveaxis(z, -1, 0)  # move sp_len to first axis to make mean nicer
        nz.append(np.mean(z[start_index:fin_index], axis=0))

    # nz = [np.mean(z[:, :, :, start_index:fin_index], axis=3) for z in chunked_data]  # Average the last dimension
    nz = np.moveaxis(np.array(nz), 0, 1)  # So that ylen is first now
    # (ylen, setpoins, num_steps, num_cycles)

    if nz.shape[0] == 1:  # Remove ylen dimension if len == 1
        nz = np.squeeze(nz, axis=0)
    return np.array(nz)


def average_cycles(binned_data, start_cycle=None, fin_cycle=None):
    """
    Average values from cycles from start_cycle to fin_cycle
    Args:
        binned_data (np.ndarray): Binned AWG data with shape ([ylen], setpoints, num_steps, num_cycles)
        start_cycle (Union(int, None)): Cycle to start averaging from
        fin_cycle (Union(int, None)): Cycle to finish averaging on (can be negative to count backwards)

    Returns:
        np.ndarray: Averaged data with shape ([ylen], setpoints, num_steps)

    """
    # [y], setpoints, numsteps, cycles
    data = np.array(binned_data, ndmin=4)  # [y], setpoints, numsteps, cycles
    averaged = np.mean(np.moveaxis(data, -1, 0)[start_cycle:fin_cycle], axis=0)
    if averaged.shape[0] == 1:  # Return 1D or 2D depending on y_len
        averaged = np.squeeze(averaged, axis=0)
    return averaged


def average_2D(x: np.ndarray, data: np.ndarray, centers: Optional[np.ndarray] = None, avg_nans: bool = False):
    """
    Averages data in y direction after centering using fits to v0 parts of square wave. Returns 1D data unchanged
    Args:
        x (np.ndarray): Original x_array for data
        data (np.ndarray): Data after binning and cycle averaging. Shape ([ylen], setpoints, num_steps)
        centers (Optional[np.ndarray]): Optional center positions to use instead of standard automatic transition fits
        avg_nans (bool): Whether to average data which includes NaNs (useful for two part entropy scans)
    Returns:
        Tuple[np.ndarray, np.ndarray]: New x_array, averaged_data (shape (setpoints, num_steps))
    """
    if data.ndim == 3:
        z0s = data[:, (0, 2)]
        z0_avg_per_row = np.mean(z0s, axis=1)
        if centers is None:
            from .Transition import transition_fits
            fits = transition_fits(x, z0_avg_per_row)
            if np.any([fit is None for fit in fits]):  # Not able to do transition fits for some reason
                logger.warning(f'{np.sum([1 if fit is None else 0 for fit in fits])} transition fits failed, blind '
                               f'averaging instead of centered averaging')
                return x, np.mean(data, axis=0)
            fit_infos = [FitInfo.from_fit(fit) for fit in fits]  # Has my functions associated
            centers = [fi.best_values.mid for fi in fit_infos]
        nzs = []
        nxs = []
        for z in np.moveaxis(data, 1, 0):  # For each of v0_0, vP, v0_1, vM
            nz, nx = CU.center_data(x, z, centers, return_x=True)
            nzs.append(nz)
            nxs.append(nx)
        assert (nxs[0] == nxs).all()  # Should all have the same x_array
        ndata = np.array(nzs)
        if avg_nans is True:
            ndata = np.nanmean(ndata, axis=1)  # Average centered data
        else:
            ndata = np.mean(ndata, axis=1)  # Average centered data
        nx = nxs[0]
    else:
        nx = x
        ndata = data
        logger.info(f'Data passed in was {data.ndim - 1}D (not 2D), same values returned')
    return nx, ndata, centers


def entropy_signal(data: np.ndarray) -> np.ndarray:
    """
    Calculates equivalent of second harmonic from data with v0_0, vP, v0_1, vM as first dimension
    Note: Data should be aligned for same x_array before doing this
    Args:
        data (np.ndarray): Data with first dimension corresponding to v0_0, vP, v0_1, vM. Can be any dimensions for rest

    Returns:
        np.ndarray: Entropy signal array with same shape as data minus the first axis

    """
    assert data.shape[0] == 4
    entropy_data = -1 * (np.mean(data[(1, 3),], axis=0) - np.mean(data[(0, 2),], axis=0))
    return entropy_data


def integrate_entropy(data, sf):
    return np.nancumsum(data) * sf


def calculate_dT(bias_lookup, bias):
    return bias_lookup[bias] - bias_lookup[0]


def align_setpoints(xs, data, nx=None):
    """
    In case want to realign data where each setpoint of heating has a different x_array (i.e. taking into account some
    additional shifts)
    Args:
        xs (np.ndarray):  x_array for each heating setpoint in data
        data (np.ndarray):  data with shape (setpoints, num_steps)
        nx (np.ndarray): New x_array to put data on, or will use first of xs by default

    Returns:
        Tuple[np.ndarray, np.ndarray]: new x_array, interpolated data with same shape as original
    """
    assert xs.ndim == 2  # different x_array for each heating setpoint
    assert xs.shape[0] == data.shape[0]
    oxs = xs  # Old xs
    if nx is None:
        nx = xs[0]  # New x
    ndata = []  # New data
    for ox, z in zip(oxs, data):
        interper = interp1d(xs, z, bounds_error=False)
        ndata.append(interper(nx))
    data = np.array(ndata)  # Data which shares the same x axis
    return nx, data


# endregion


def square_wave_time_array(awg: AWG.AWG) -> np.ndarray:
    """Returns time array of single square wave (i.e. time in s for each sample in a full square wave cycle)"""
    num_pts = awg.info.wave_len
    duration = num_pts / awg.measure_freq
    x = np.linspace(0, duration, num_pts)  # In seconds
    return x
