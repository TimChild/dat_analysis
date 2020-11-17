"""This is where ExpSpecific is used to get data into a standard form to be passed to DatBuilders which
make DatHDFs"""
from __future__ import annotations
import os
import logging
from src.DatObject.DatHDF import DatHDF
from src import CoreUtil as CU
from src.DataStandardize.ExpSpecific.Sep20 import SepExp2HDF
from singleton_decorator import singleton
import src.HDF_Util as HDU
from src.DatObject.DatHDF import DatHDFBuilder
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.DataStandardize.BaseClasses import Exp2HDF


default_Exp2HDF = SepExp2HDF


logger = logging.getLogger(__name__)


@singleton
class DatHandler(object):
    """
    Holds onto references to open dats (so that I don't try open the same datHDF more than once). Will return
    same dat instance if already open.
    Can also see what dats are open, remove individual dats from DatHandler, or clear all dats from DatHandler
    """
    open_dats = {}

    @classmethod
    def get_dat(cls, datnum, datname='base', overwrite=False,  init_level='min', exp2hdf=None) -> DatHDF:
        exp2hdf = exp2hdf(datnum) if exp2hdf else default_Exp2HDF(datnum)
        full_id = f'{exp2hdf.ExpConfig.dir_name}:{CU.get_dat_id(datnum, datname)}'  # For temp local storage
        path = exp2hdf.get_datHDF_path()
        cls._ensure_dir(path)
        if overwrite:
            cls._delete_hdf(path)
            del cls.open_dats[full_id]

        if full_id not in cls.open_dats:  # Need to open or create DatHDF
            if os.path.isfile(path):
                cls.open_dats[full_id] = cls._open_hdf(path)
            else:
                cls._check_exp_data_exists(exp2hdf)
                builder = DatHDFBuilder(exp2hdf, init_level)
                cls.open_dats[full_id] = builder.build_dat()
        return cls.open_dats[full_id]

    @staticmethod
    def _ensure_dir(path):
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            logger.warning(f'{dir_path} was not found, being created now')
            os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def _delete_hdf(path: str):
        """Should delete hdf if it exists"""
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _open_hdf(path: str):
        hdf_container = HDU.HDFContainer.from_path(path)
        dat = DatHDF(hdf_container)
        return dat

    @staticmethod
    def _check_exp_data_exists(exp2hdf: Exp2HDF):
        exp_path = exp2hdf.get_exp_dat_path()
        if os.path.isfile(exp_path):
            return True
        else:
            raise FileNotFoundError(f'No experiment data found for dat{exp2hdf.datnum} at {os.path.abspath(exp_path)}')

    # @classmethod
    # def get_dat(cls, datnum, datname=None, overwrite=False, dattypes=None, init_level='min',
    #             Exp2HDF = None) -> DO.DatHDF.DatHDF:
    #     datname = datname if datname else 'base'
    #     dat_id = cls._get_dat_id(datnum, datname, Exp2HDF)
    #     if dat_id not in cls.open_dats or overwrite is True:
    #         new_dat = make_dat(datnum, datname, overwrite=overwrite, dattypes=dattypes, Exp2HDF=Exp2HDF, init_level=init_level)
    #         cls.open_dats[dat_id] = new_dat
    #     return cls.open_dats[dat_id]

    # @classmethod
    # def get_dats(cls, datnums, datname=None, overwrite=False, dattypes=None, init_level='min',
    #              Exp2HDF = None, progress=True) -> List[DO.DatHDF.DatHDF]:
    #     # TODO: Multithread this
    #     if not ((hasattr(datnums, '__iter__') and type(datnums) != tuple) or (type(datnums) == tuple and len(datnums) == 2)):
    #         raise ValueError(f'Datnums [{datnums}] cannot be interpreted as an iterable or the values of a range')
    #     if type(datnums) == tuple:
    #         datnums = range(datnums[0], datnums[1])
    #     if progress is True:
    #         dats = list()
    #         for num in progressbar(datnums):
    #             stdout.flush()
    #             dats.append(cls.get_dat(num, datname=datname, overwrite=overwrite, dattypes=dattypes,
    #                        init_level=init_level, Exp2HDF=Exp2HDF))
    #     else:
    #         dats = [cls.get_dat(num, datname=datname, overwrite=overwrite, dattypes=dattypes,
    #                         init_level=init_level, Exp2HDF=Exp2HDF) for num in datnums]
    #     return dats

    @classmethod
    def list_open_dats(cls):
        return cls.open_dats

    # @classmethod
    # def remove_dat(cls, datnum, datname, Exp2HDF=None, verbose=True):
    #     dat_id = cls._get_dat_id(datnum, datname, Exp2HDF)
    #     if dat_id in cls.open_dats:
    #         dat = cls.open_dats[dat_id]
    #         dat.old_hdf.close()
    #         del cls.open_dats[dat_id]
    #         logger.info(f'Removed [{dat_id}] from dat_handler', verbose)
    #     else:
    #         logger.info(f'Nothing to be removed', verbose)
    #
    # @classmethod
    # def clear_dats(cls):
    #     for dat in cls.open_dats.values():
    #         dat.old_hdf.close()
    #         del dat
    #     cls.open_dats = {}


def make_dat(datnum, datname, overwrite=False, dat_types = None, Exp2HDF=None, init_level='min'):
    """
    This creates a DatHDF

    Args:
        datnum ():
        datname ():
        overwrite ():
        dat_types ():
        Exp2HDF ():
        init_level ():

    Returns:

    """


# def make_dat(datnum, datname, overwrite=False, dattypes=None, Exp2HDF=None, init_level='min'):
#     """ Standard make_dat which will call on experiment specific configs/builders etc
#
#     Args:
#         datnum (): Datnum
#         datname (): Name to save with (if want to make multiple versions of a single DatHDF)
#         overwrite (): Whether to delete existing DatHDF and do a clean initialization
#         dattypes (): Override the dattypes for the dat (this is the easiest thing to forget when running experiment)
#
#     Returns:
#
#     """
#
#     Exp2HDF = Exp2HDF if Exp2HDF else default_Exp2HDF
#     exp2hdf = Exp2HDF(datnum)
#     exp2hdf.dat_types = dattypes  # Only sets if not None
#     hdfdir = exp2hdf.get_hdfdir()
#     name = exp2hdf.get_name(datname)
#     path = os.path.join(hdfdir, name+'.h5')
#     if os.path.isfile(path) and overwrite is False:
#         with h5py.File(path, 'r') as temp_hdf:
#             initialized = temp_hdf.attrs.get('initialized', False)
#         if initialized:
#             Loader = DO.DatLoaders.get_loader(exp2hdf.dat_types())
#             loader = Loader(file_path=path)
#             # loader = Loader(datnum, datname, None, hdfdir)  # Or this should be equivalent
#             return loader.build_dat()
#         else:
#             logger.warning(f'HDF at [{path}] was not initialized properly. Overwriting now.')
#             overwrite = True  # Must have failed to fully initialize last time so it's an unfinished HDF anyway
#
#     # If overwrite/build
#     builder = _make_basic_part(exp2hdf, datnum, datname, overwrite)
#     builder = _make_other_parts(exp2hdf, builder, run_fits)
#     builder.set_initialized()
#     builder.hdf.close()
#
#     Loader = DO.DatLoaders.get_loader(exp2hdf.dat_types())
#     loader = Loader(file_path=path)
#     return loader.build_dat()
#
#
# def _make_basic_part(exp2hdf, datnum, datname, overwrite) -> DO.DatBuilder.NewDatBuilder:
#     """Get get the correct builder and initialize Logs, Instruments and Data of builder before returning"""
#     data_exists = exp2hdf._check_data_exists()
#     if not data_exists:
#         raise FileNotFoundError(f'No experiment data found for dat{datnum} in:\r{exp2hdf.get_ddir()}\r')
#
#     dattypes = exp2hdf.dat_types()
#     Builder = DO.DatBuilder.get_builder(dattypes)
#
#     hdfdir = exp2hdf.get_hdfdir()
#     builder = Builder(datnum, datname, hdfdir, overwrite)
#
#     ddir = exp2hdf.get_ddir()
#     builder.copy_exp_hdf(ddir)
#
#     sweep_logs = exp2hdf.get_sweeplogs()
#
#     builder.init_Logs(sweep_logs)
#
#     builder.init_Instruments()
#
#     setup_dict = exp2hdf.get_data_setup_dict()
#     builder.init_Data(setup_dict)
#
#     builder.init_Other()
#
#     dattypes = exp2hdf.dat_types()
#     builder.set_dattypes(dattypes)
#
#     builder.set_base_attrs_HDF()
#
#     return builder
#
#
# def _make_other_parts(exp2hdf, builder, run_fits):
#     """Add and init Entropy/Transtion/DCbias"""
#     dattypes = exp2hdf.dat_types()
#     if isinstance(builder, DO.DatBuilder.TransitionDatBuilder) and 'transition' in dattypes:
#         builder.init_Transition()
#         if run_fits is True:
#             builder.Transition.get_from_HDF()
#             builder.Transition.run_row_fits()
#             builder.Transition.set_avg_data()
#             builder.Transition.run_avg_fit()
#
#     if isinstance(builder, DO.DatBuilder.TransitionDatBuilder) and 'AWG' in dattypes:
#         builder.init_AWG(builder.Logs.group, builder.Data.group)
#
#     if isinstance(builder, DO.DatBuilder.SquareDatBuilder) and 'square entropy' in dattypes:
#         builder.init_SquareEntropy()
#         if run_fits is True:
#             builder.SquareEntropy.get_from_HDF()
#             builder.SquareEntropy.process()
#
#     if isinstance(builder, DO.DatBuilder.EntropyDatBuilder) and 'entropy' in dattypes:
#         # TODO: Can't get center data if fits haven't been run yet, need to think about how to init entropy later
#         builder.Data.get_from_HDF()
#         try:
#             centers = np.array([f.best_values.mid for f in builder.Transition.all_fits])
#         except TypeError as e:
#             centers = None
#         except Exception as e:
#             raise e  # TODO: see what gets caught and set except accordingly then remove this
#         builder.init_Entropy(centers=centers)
#         if run_fits is True:
#             builder.Entropy.get_from_HDF()
#             x = builder.Entropy.x
#             data = builder.Entropy.data
#             mids = [fit.best_values.mid for fit in builder.Transition.all_fits]
#             thetas = [fit.best_values.theta for fit in builder.Transition.all_fits]
#             params = E.get_param_estimates(x, data)#, mids, thetas)
#             builder.Entropy.run_row_fits(params=params)
#             if centers is not None:
#                 builder.Entropy.set_avg_data()
#                 builder.Entropy.run_avg_fit()
#
#     return builder


