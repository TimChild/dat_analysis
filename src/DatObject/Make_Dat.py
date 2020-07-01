"""This is where ExpSpecific is used to get data into a standard form to be passed to DatBuilders which
make DatHDFs"""

import os
import h5py
import logging


import src.DatObject.DatLoaders
from src import CoreUtil as CU
from src.DatObject import DatBuilder
from src.DatObject.Attributes import Entropy as E

from src.DataStandardize.ExpSpecific.Jun20 import JunESI, JunConfig
default_ESI = JunESI
default_config = JunConfig()

logger = logging.getLogger(__name__)


def make_dat(datnum, datname, overwrite=False, dattypes=None, ESI_class=None, run_fits=True):
    """ Standard make_dat which will call on experiment specific configs/builders etc

    Args:
        datnum ():
        datname ():
        overwrite ():
        dattypes ():

    Returns:

    """

    ESI_class = ESI_class if ESI_class else default_ESI
    esi = ESI_class(datnum)
    esi.set_dattypes(dattypes)  # Only sets if not None
    hdfdir = esi.get_hdfdir()
    name = CU.get_dat_id(datnum, datname)
    path = os.path.join(hdfdir, name+'.h5')
    if os.path.isfile(path) and overwrite is False:
        with h5py.File(path, 'r') as temp_hdf:
            initialized = temp_hdf.attrs.get('initialized', False)
        if initialized:
            Loader = src.DatObject.DatLoaders.get_loader(esi.get_dattypes())
            loader = Loader(file_path=path)
            # loader = Loader(datnum, datname, None, hdfdir)  # Or this should be equivalent
            return loader.build_dat()
        else:
            logger.warning(f'HDF at [{path}] was not initialized properly. Overwriting now.')
            overwrite = True  # Must have failed to fully initialize last time so it's an unfinished HDF anyway

    # If overwrite/build
    builder = _make_basic_part(esi, datnum, datname, overwrite)
    builder = _make_other_parts(esi, builder, run_fits)
    builder.set_initialized()
    builder.hdf.close()

    Loader = src.DatObject.DatLoaders.get_loader(esi.get_dattypes())
    loader = Loader(file_path=path)
    return loader.build_dat()


def _make_basic_part(esi, datnum, datname, overwrite) -> DatBuilder.NewDatBuilder:
    """Get get the correct builder and initialize Logs, Instruments and Data of builder before returning"""
    dattypes = esi.get_dattypes()
    Builder = DatBuilder.get_builder(dattypes)

    hdfdir = esi.get_hdfdir()
    builder = Builder(datnum, datname, hdfdir, overwrite)

    ddir = esi.get_ddir()
    builder.copy_exp_hdf(ddir)
    sweep_logs = esi.get_sweeplogs()

    builder.init_Logs(sweep_logs)

    builder.init_Instruments()

    setup_dict = esi.get_data_setup_dict()
    builder.init_Data(setup_dict)

    builder.init_Other()

    dattypes = esi.get_dattypes()
    builder.set_dattypes(dattypes)

    builder.set_base_attrs_HDF()

    return builder


def _make_other_parts(esi, builder, run_fits):
    """Add and init Entropy/Transtion/DCbias"""
    dattypes = esi.get_dattypes()
    if isinstance(builder, DatBuilder.TransitionDatBuilder) and 'transition' in dattypes:
        builder.init_Transition()
        if run_fits is True:
            builder.Transition.get_from_HDF()
            builder.Transition.run_row_fits()
            builder.Transition.set_avg_data()
            builder.Transition.run_avg_fit()

    if isinstance(builder, DatBuilder.EntropyDatBuilder) and 'entropy' in dattypes:
        # TODO: Can't get center data if fits haven't been run yet, need to think about how to init entropy later
        try:
            center_ids = CU.get_data_index(builder.Data.x_array,
                                           [fit.best_values.mid for fit in builder.Transition.all_fits])

        except TypeError as e:
            center_ids = None
        except Exception as e:
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

    if isinstance(builder, DatBuilder.TransitionDatBuilder) and 'AWG' in dattypes:
        builder.init_AWG(builder.Logs.group, builder.Data.group)

    # if isinstance(builder, Builders.DCbiasDatBuilder) and 'dcbias' in dattypes:
    #     pass
    return builder


