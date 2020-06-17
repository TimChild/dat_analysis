from typing import Union, List, Set

import src.DatBuilder
from src.DatBuilder import Util, Builders, DatHDF
from src import CoreUtil as CU
from src.Configs import Main_Config as cfg
from src.DFcode import DatDF as DF, SetupDF as SF

import abc

class ExpToDatBase(abc.ABC):
    """Base class for standard functions going from Experiment to Dat"""

    def __init__(self, datnum, datname='base', dattypes=None, setupdf=None, config=None, run_fits=True):
        """ Basic info to go from exp data to Dat

        Args:
            datnum (int): Datnum to look for
            datname (str): Name to save with
            dattypes (Union[str, List[str], Set[str]]): What info the dat contains, should try to store this info in sweeplog comments and get from there automatically
            setupdf (SF.SetupDF): Setup DataFrame with multipliers/offsets for any recorded data by datnum
            config (module): Exp specific config to get exp specific params from
            run_fits (bool): Whether to run fits on initialization
        """
        self.datnum = datnum
        self.datname = datname
        self.dattypes = dattypes
        self.setupdf = setupdf
        self.config = config
        self.run_fits = run_fits





def make_dat_from_exp(datnum, datname: str = 'base', dattypes: Union[str, List[str], Set[str]] = None, setupdf: SF.SetupDF = None, config=None,
                      run_fits=True) -> DatHDF.DatHDF:
    """
    Loads or creates dat object and interacts with Main_Config (and through that the Experiment specific configs)

    @param datnum: dat[datnum].h5
    @type datnum: int
    @param datname: name for storing in datdf and files
    @type datname: str
    @param dattypes: what types of info dat contains, e.g. 'transition', 'entropy', 'dcbias'
    @type dattypes: Union[str, List[str], Set[str]]
    @param datdf: datdf to load dat from or overwrite to.
    @type datdf: DF.DatDF
    @param setupdf: setup df to use to get corrected data when loading dat in
    @type setupdf: SetupDF
    @param config: config file to use when loading dat in, will default to cfg.current_config. Otherwise pass the whole module in
    @type config: module
    @return: dat object
    @rtype: Dat
    """

    old_config = cfg.current_config
    if config is None:
        config = cfg.current_config
    else:
        cfg.set_all_for_config(config, folder_containing_experiment=None)

    datdf = datdf if datdf else DF.DatDF()
    setupdf = setupdf if setupdf else SF.SetupDF()

    if setupdf.config_name != datdf.config_name or setupdf.config_name != config.__name__.split('.')[-1]:
        raise AssertionError(f'C.make_dat_standard: setupdf, datdf and config have different config names'
                             f'[{setupdf.config_name, datdf.config_name, config.__name__.split(".")[-1]},'
                             f' probably you dont want to continue!')

    dat_builder = Builders.EntropyDatBuilder(datnum, datname, dfname=datdf.name)

    json_str = dat_builder.hdf['Exp_metadata'].attrs['sweep_logs']
    sweeplogs_json = Util.metadata_to_JSON(json_str, config, datnum)

    dat_builder.init_Logs(sweeplogs_json)
    dat_builder.init_Instruments()

    dattypes = Util.get_dattypes(dattypes, getattr(dat_builder.Logs, 'comments', None), config.dat_types_list)
    dat_builder.set_dattypes(dattypes)  # Update in Dat

    setup_dict = Util.get_data_setup_dict(dat_builder, dattypes, setupdf, config)
    dat_builder.init_Data(setup_dict=setup_dict)

    if 'transition' in dattypes:
        dat_builder.init_Transition()
        if run_fits is True:
            dat_builder.Transition.run_row_fits()
            dat_builder.Transition.set_avg_data()
            dat_builder.Transition.run_avg_fit()

    if 'entropy' in dattypes:
        try:
            center_ids = CU.get_data_index(dat_builder.Data.x_array,
                                           [fit.best_values.mid for fit in dat_builder.Transition.all_fits])
        except Exception as e:
            center_ids = None
            raise e  # TODO: see what gets caught and set except accordingly then remove this

        dat_builder.init_Entropy(center_ids=center_ids)
        if run_fits is True:
            dat_builder.Entropy.run_row_fits()
            if center_ids is not None:
                dat_builder.Entropy.set_avg_data()
                dat_builder.Entropy.run_avg_fit()

    if 'dcbias' in dattypes:
        dat_builder.init_DCbias()
        # TODO: Finish this

    dat = dat_builder.build_dat()

    cfg.set_all_for_config(old_config, folder_containing_experiment=None)
    return dat