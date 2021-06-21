from __future__ import annotations
import inspect
import os
from src.data_standardize.base_classes import Directories, get_expected_sub_dir_paths, ExpConfigBase
from src.data_standardize.ExpSpecific import Sep20
from src.dat_object.make_dat import DatHandler
import shutil

from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from src.dat_object.dat_hdf import  DatHDF


def stack_inspector():
    """Prints out current stack with index values"""
    stack = inspect.stack()
    for i, frame in enumerate(stack):
        for j, val in enumerate(frame):
            print(f'[{i}][{j}] = {val},', end='\t')
        print('')


def get_testing_Exp2HDF(dat_dir, output_dir):
    Testing_SysConfig = get_testing_SysConfig(dat_dir, output_dir)
    Testing_ExpConfig = get_testing_ExpConfig()

    class Testing_Exp2HDF(Sep20.SepExp2HDF):
        ExpConfig = Testing_ExpConfig()
        SysConfig = Testing_SysConfig()

        def _get_update_batch_path(self):
            raise NotImplementedError

        def synchronize_data(self):
            raise NotImplementedError

    return Testing_Exp2HDF


def get_testing_ExpConfig() -> Type[ExpConfigBase]:
    class Testing_ExpConfig(Sep20.SepExpConfig):
        pass
    return Testing_ExpConfig


def get_testing_SysConfig(dat_dir, output_dir):
    def get_testing_dirs(dat_dir, output_dir) -> Directories:
        hdfdir, _ = get_expected_sub_dir_paths(output_dir)
        ddir = dat_dir
        dirs = Directories(hdfdir, ddir)
        return dirs

    class Testing_SysConfig(Sep20.SepSysConfig):
        main_folder_path = NotImplementedError
        dir_name = NotImplementedError
        Directories = get_testing_dirs(dat_dir, output_dir)

        def get_directories(self):
            raise NotImplementedError

        def synchronize_data_batch_file(self):
            raise NotImplementedError

    return Testing_SysConfig


def init_testing_dat(datnum, output_directory, allow_non_matching_directory=False, overwrite=True) -> DatHDF:
    """
    Initialized Dat for testing purposes
    Args:
        datnum (int): Identifier of dat to load
        dat_hdf_directory (str): Place to initialize dat to

    Returns:
        DatHDF: Returns dat instance which has HDF initialized in output_directory/Dat_HDFs/...
    """
    dat_dir = os.path.abspath('fixtures/dats/2020Sep/')
    if not allow_non_matching_directory:
        output_directory = os.path.normpath(output_directory)
        assert 'Outputs' in output_directory.split(os.sep)

    Exp2hdf = get_testing_Exp2HDF(dat_dir, output_directory)
    dat = DatHandler().get_dat(datnum, overwrite=True, exp2hdf=Exp2hdf)
    return dat


def clear_outputs(output_dir):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
