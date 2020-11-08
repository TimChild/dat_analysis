"""This is where ExpSpecific is used to get data into a standard form to be passed to DatBuilders which
make DatHDFs"""
import os
import h5py
import logging
import src.DatObject as DO
import numpy as np
from progressbar import progressbar
from sys import stdout
from src import CoreUtil as CU
from src.DatObject.Attributes import Entropy as E

from src.DataStandardize.ExpSpecific.Sep20 import SepESI, SepExpConfig
from typing import List
default_ESI = SepESI
default_config = SepExpConfig()

logger = logging.getLogger(__name__)


class DatHandler(object):
    """
    Holds onto references to open dats (so that I don't try open the same datHDF more than once). Will return
    same dat instance if already open.
    Can also see what dats are open, remove individual dats from DatHandler, or clear all dats from DatHandler
    """
    open_dats = {}

    @staticmethod
    def _get_dat_id(datnum, datname, ESI_class = None):
        esi = ESI_class(datnum) if ESI_class is not None else default_ESI(datnum)
        path_id = esi.Config.dir_name
        return f'{path_id}:dat{datnum}[{datname}]'

    @classmethod
    def get_dat(cls, datnum, datname=None, overwrite=False, dattypes=None, run_fits=True,
                ESI_class = None) -> DO.DatHDF.DatHDF:
        datname = datname if datname else 'base'
        dat_id = cls._get_dat_id(datnum, datname, ESI_class)
        if dat_id not in cls.open_dats or overwrite is True:
            new_dat = make_dat(datnum, datname, overwrite=overwrite, dattypes=dattypes, ESI_class=ESI_class, run_fits=run_fits)
            cls.open_dats[dat_id] = new_dat
        return cls.open_dats[dat_id]

    @classmethod
    def get_dats(cls, datnums, datname=None, overwrite=False, dattypes=None, run_fits=True,
                 ESI_class = None, progress=True) -> List[DO.DatHDF.DatHDF]:
        if not ((hasattr(datnums, '__iter__') and type(datnums) != tuple) or (type(datnums) == tuple and len(datnums) == 2)):            raise ValueError(f'Datnums [{datnums}] cannot be interpreted as an iterable or the values of a range')
        if type(datnums) == tuple:
            datnums = range(datnums[0], datnums[1])
        if progress is True:
            dats = list()
            for num in progressbar(datnums):
                stdout.flush()
                dats.append(cls.get_dat(num, datname=datname, overwrite=overwrite, dattypes=dattypes,
                            run_fits=run_fits, ESI_class=ESI_class))
        else:
            dats = [cls.get_dat(num, datname=datname, overwrite=overwrite, dattypes=dattypes,
                            run_fits=run_fits, ESI_class=ESI_class) for num in datnums]
        return dats

    @classmethod
    def list_open_dats(cls):
        return cls.open_dats

    @classmethod
    def remove_dat(cls, datnum, datname, ESI_class=None, verbose=True):
        dat_id = cls._get_dat_id(datnum, datname, ESI_class)
        if dat_id in cls.open_dats:
            dat = cls.open_dats[dat_id]
            dat.hdf.close()
            del cls.open_dats[dat_id]
            logger.info(f'Removed [{dat_id}] from dat_handler', verbose)
        else:
            logger.info(f'Nothing to be removed', verbose)

    @classmethod
    def clear_dats(cls):
        for dat in cls.open_dats.values():
            dat.hdf.close()
            del dat
        cls.open_dats = {}


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
            Loader = DO.DatLoaders.get_loader(esi.dat_types())
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

    Loader = DO.DatLoaders.get_loader(esi.dat_types())
    loader = Loader(file_path=path)
    return loader.build_dat()


def _make_basic_part(esi, datnum, datname, overwrite) -> DO.DatBuilder.NewDatBuilder:
    """Get get the correct builder and initialize Logs, Instruments and Data of builder before returning"""
    data_exists = esi._check_data_exists()
    if not data_exists:
        raise FileNotFoundError(f'No experiment data found for dat{datnum} in:\r{esi.get_ddir()}\r')

    dattypes = esi.dat_types()
    Builder = DO.DatBuilder.get_builder(dattypes)

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

    dattypes = esi.dat_types()
    builder.set_dattypes(dattypes)

    builder.set_base_attrs_HDF()

    return builder


def _make_other_parts(esi, builder, run_fits):
    """Add and init Entropy/Transtion/DCbias"""
    dattypes = esi.dat_types()
    if isinstance(builder, DO.DatBuilder.TransitionDatBuilder) and 'transition' in dattypes:
        builder.init_Transition()
        if run_fits is True:
            builder.Transition.get_from_HDF()
            builder.Transition.run_row_fits()
            builder.Transition.set_avg_data()
            builder.Transition.run_avg_fit()

    if isinstance(builder, DO.DatBuilder.TransitionDatBuilder) and 'AWG' in dattypes:
        builder.init_AWG(builder.Logs.group, builder.Data.group)

    if isinstance(builder, DO.DatBuilder.SquareDatBuilder) and 'square entropy' in dattypes:
        builder.init_SquareEntropy()
        if run_fits is True:
            builder.SquareEntropy.get_from_HDF()
            builder.SquareEntropy.process()

    if isinstance(builder, DO.DatBuilder.EntropyDatBuilder) and 'entropy' in dattypes:
        # TODO: Can't get center data if fits haven't been run yet, need to think about how to init entropy later
        builder.Data.get_from_HDF()
        try:
            centers = np.array([f.best_values.mid for f in builder.Transition.all_fits])
        except TypeError as e:
            centers = None
        except Exception as e:
            raise e  # TODO: see what gets caught and set except accordingly then remove this
        builder.init_Entropy(centers=centers)
        if run_fits is True:
            builder.Entropy.get_from_HDF()
            x = builder.Entropy.x
            data = builder.Entropy.data
            mids = [fit.best_values.mid for fit in builder.Transition.all_fits]
            thetas = [fit.best_values.theta for fit in builder.Transition.all_fits]
            params = E.get_param_estimates(x, data)#, mids, thetas)
            builder.Entropy.run_row_fits(params=params)
            if centers is not None:
                builder.Entropy.set_avg_data()
                builder.Entropy.run_avg_fit()

    return builder


