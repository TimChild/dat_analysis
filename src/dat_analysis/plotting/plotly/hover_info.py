from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from dat_analysis.dat_object.dat_hdf import DatHDF


@dataclass
class HoverInfo:
    name: str
    func: Callable
    precision: str = '.2f'
    units: str = 'mV'
    position: Optional[int] = None


class DefaultHoverInfos:
    """
    Common hover infos which are generally useful when plotting multiple dats in one plot
    """
    @staticmethod
    def datnum() -> HoverInfo:
        return HoverInfo(name='Datnum', func=lambda dat: dat.datnum, precision='d', units='')

    @staticmethod
    def heating_bias() -> HoverInfo:
        return HoverInfo(name='Heating Bias', func=lambda dat: dat.AWG.max(0)/10, precision='.2f', units='nA')

    @staticmethod
    def fit_entropy(fit_name: str = 'default') -> HoverInfo:
        return HoverInfo(name='Fit Entropy',
                         func=lambda dat:
                         dat.Entropy.get_fit(name=fit_name, check_exists=True).best_values.dS)

    @classmethod
    def xlabel(cls, dat_or_name: Union[DatHDF, str], xfunc: Callable, units='(mV)'):
        return cls._label(dat_or_name, xfunc, lambda dat: dat.Logs.xlabel, units)

    @classmethod
    def ylabel(cls, dat_or_name: Union[DatHDF, str], yfunc: Callable, units='(mV)'):
        return cls._label(dat_or_name, yfunc, lambda dat: dat.Logs.ylabel, units)

    @staticmethod
    def _label(dat_or_name: Union[DatHDF, str],
               dat_value_func: Callable[[DatHDF], float],
               dat_label_func: Optional[Callable[[DatHDF], str]] = None,
               units='(mV)') -> HoverInfo:
        """
        Create a HoverInfo from general info (i.e. to be used by public methods of DefaultHoverInfos
        Args:
            dat_or_name (): Either a DatHDF to get a label from using dat_label_func or the label as a string
            dat_value_func (): Function which takes dat as the only argument and returns a value
            dat_label_func (): Used only if Dat provided -- Function which takes dat as the only argument and returns a
                string
            units (): Units for the value (defaults to mV)

        Returns:
            HoverInfo: An initialized HoverInfo for the given info
        """
        from dat_analysis.dat_object.dat_hdf import DatHDF
        if isinstance(dat_or_name, DatHDF):
            assert dat_label_func is not None
            name = dat_label_func(dat_or_name)
        elif isinstance(dat_or_name, str):
            name = dat_or_name
        else:
            raise NotImplementedError

        return HoverInfo(name=name, func=dat_value_func, precision='.2f', units=units)


@dataclass
class HoverInfoGroup:
    """
    For working with hover_info in plotly. Use this to get the hovertemplate and customdata

    Examples
        hig = HoverInfoGroup(hover_infos)
        trace = go.Scatter(hovertemplate = hig.template, customdata = hig.customdata(dats))

    """
    hover_infos: List[HoverInfo]

    def __post_init__(self):
        self.funcs, self.template = _additional_data_dict_converter(self.hover_infos)

    def customdata(self, dats: Union[List[DatHDF], DatHDF]) -> Union[List[list], list]:
        """
        Get the customdata (hover_data) for given Dat(s)
        Args:
            dats (): Either a list of Dats for a list of customdata, or single dat for single customdata

        Returns:

        """
        return_single = False
        if not isinstance(dats, list):
            dats = [dats]
            return_single = True

        customdata = [[func(dat) for func in self.funcs] for dat in dats]
        if return_single:
            customdata = customdata[0]
        return customdata


def _additional_data_dict_converter(info: List[HoverInfo], customdata_start: int = 0) -> (list, str):
    """
    Note: Use HoverInfoGroup instance instead of calling this directly.

    Converts a list of HoverInfos into a list of functions and a hover template string
    Args:
        info (List[HoverInfo]): List of HoverInfos containing ['name', 'func', 'precision', 'units', 'position']
            'name' and 'func' are necessary, the others are optional. 'func' should take DatHDF as an argument and return
            a value. 'precision' is the format specifier (e.g. '.2f'), and units is added afterwards
        customdata_start (int): Where to start customdata[i] from. I.e. start at 1 if plot function already adds datnum
            as customdata[0].
    Returns:
        Tuple[list, str]: List of functions which get data from dats, template string to use in hovertemplate
    """
    items = list()
    for d in info:
        name = d.name
        func = d.func
        precision = d.precision
        units = d.units
        position = d.position if d.position is not None else len(items)

        items.insert(position, (func, (name, precision, units)))  # Makes list of (func, (template info))

    funcs = [f for f, _ in items]
    # Make template for each func in order.. (i+custom_data_start) to reserve customdata[0] for datnum
    template = '<br>'.join(
        [f'{name}=%{{customdata[{i + customdata_start}]:{precision}}}{units}' for i, (_, (name, precision, units)) in
         enumerate(items)])
    return funcs, template



