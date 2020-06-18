from src.Configs import Main_Config as cfg
from src import CoreUtil as CU
from src.DatBuilder import Builders
from src.DatBuilder.DatHDF import get_dat_id
from src.DatAttributes import Entropy as E
import abc
import os
from typing import Union, Type, List, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.ExperimentSpecific.BaseClasses import ExperimentSpecificInterface

logger = logging.getLogger(__name__)


def make_dat(datnum, datname, overwrite=False, dattypes=None, ESI_class=None, run_fits=True):
    """ Standard make_dat which will call on experiment specific configs/builders etc

    Args:
        datnum ():
        datname ():
        overwrite ():
        builder ():
        dattypes ():

    Returns:

    """

    ESI_class = ESI_class if ESI_class else cfg.default_ESI  # type: Type[ExperimentSpecificInterface]
    esi = ESI_class(datnum)
    esi.set_dattypes(dattypes)  # Only sets if not None
    hdfdir = esi.get_hdfdir()
    name = get_dat_id(datnum, datname)
    path = os.path.join(hdfdir, name)
    if os.path.isfile(path) and overwrite is False:
        Loader = Builders.get_loader(esi.get_dattypes())
        loader = Loader(path)
        return loader.build_dat()

    # If overwrite/build
    builder = _make_basic_part(esi, datnum, datname, overwrite)
    builder = _make_other_parts(esi, builder, run_fits)
    return builder.build_dat()


def _make_basic_part(esi, datnum, datname, overwrite) -> Builders.NewDatBuilder:
    """Get get the correct builder and initialize Logs, Instruments and Data of builder before returning"""
    dattypes = esi.get_dattypes()
    Builder = Builders.get_builder(dattypes)  # type: Type[Builders.NewDatBuilder]

    hdfdir = esi.get_hdfdir()
    builder = Builder(datnum, datname, hdfdir, overwrite)

    ddir = esi.get_ddir()
    builder.copy_exp_hdf(ddir)
    sweep_logs = esi.get_sweeplogs()

    builder.init_Logs(sweep_logs)
    builder.init_Instruments()

    setup_dict = esi.get_data_setup_dict()
    builder.init_Data(setup_dict)


    dattypes = esi.get_dattypes()
    builder.set_dattypes(dattypes)  # TODO: Fix how this is saved/loaded in HDF
    return builder


def _make_other_parts(esi, builder, run_fits):
    """Add and init Entropy/Transtion/DCbias"""
    dattypes = esi.get_dattypes()
    if isinstance(builder, Builders.TransitionDatBuilder) and 'transition' in dattypes:
        builder.init_Transition()
        if run_fits is True:
            builder.Transition.run_row_fits()
            builder.Transition.set_avg_data()
            builder.Transition.run_avg_fit()

    if isinstance(builder, Builders.EntropyDatBuilder) and 'entropy' in dattypes:
        # TODO: Can't get center data if fits haven't been run yet, need to think about how to init entropy later
        try:
            center_ids = CU.get_data_index(builder.Data.x_array,
                                           [fit.best_values.mid for fit in builder.Transition.all_fits])
        except (TypeError, Exception) as e:
            center_ids = None
            raise e  # TODO: see what gets caught and set except accordingly then remove this
        builder.init_Entropy(center_ids=center_ids)
        if run_fits is True:
            x = builder.Entropy.x
            data = builder.Entropy.data
            mids = [fit.best_values.mid for fit in builder.Transition.all_fits]
            thetas = [fit.best_values.theta for fit in builder.Transition.all_fits]
            params = E.get_param_estimates(x, data)#, mids, thetas)
            builder.Entropy.run_row_fits(params=params)
            if center_ids is not None:
                builder.Entropy.set_avg_data()
                builder.Entropy.run_avg_fit()

    # if isinstance(builder, Builders.DCbiasDatBuilder) and 'dcbias' in dattypes:
    #     pass
    return builder



###############################################

def temp_from_json(jsondict, fridge='ls370'):
    if 'BF Small' in jsondict.keys():
        try:
            temps = temp_from_bfsmall(jsondict['BF Small'])
        except KeyError as e:
            print(jsondict)
            raise e
        return temps
    else:

        logger.info(f'Verbose[][temp_from_json] - Did not find "BF Small" in json')

    return None


def srs_from_json(jsondict, id):
    """

    Args:
        jsondict (dict): srss part of json (i.e. SRS_1: ... , SRS_2: ... etc)
        id (int): number of SRS
    Returns:
        dict: SRS data in a dict with my keys
    """
    if 'SRS_' + str(id) in jsondict.keys():
        srsdict = jsondict['SRS_' + str(id)]
        srsdata = {'gpib': srsdict['gpib_address'],
                   'out': srsdict['amplitude V'],
                   'tc': srsdict['time_const ms'],
                   'freq': srsdict['frequency Hz'],
                   'phase': srsdict['phase deg'],
                   'sens': srsdict['sensitivity V'],
                   'harm': srsdict['harmonic'],
                   'CH1readout': srsdict.get('CH1readout', None)
                   }
    else:
        srsdata = None
    return srsdata


def mag_from_json(jsondict, id, mag_type='ls625'):
    if 'LS625 Magnet Supply' in jsondict.keys():  # FIXME: This probably only works if there is 1 magnet ONLY!
        mag_dict = jsondict['LS625 Magnet Supply']  #  FIXME: Might just be able to pop entry out then look again
        magname = mag_dict.get('variable name', None)  # Will get 'magy', 'magx' etc
        if magname[-1:] == id:  # compare last letter
            mag_data = {'field': mag_dict['field mT'],
                        'rate': mag_dict['rate mT/min']
                        }
        else:
            mag_data = None
    else:
        mag_data = None
    return mag_data


def temp_from_bfsmall(tempdict):
    tempdata = {'mc': tempdict.get('MC K', None),
                'still': tempdict.get('Still K', None),
                'fourk': tempdict.get('4K Plate K', None),
                'mag': tempdict.get('Magnet K', None),
                'fiftyk': tempdict.get('50K Plate K', None)}
    return tempdata