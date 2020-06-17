import abc
import os
from datetime import datetime
import h5py
import logging
from src import CoreUtil as CU
from src.Configs import Main_Config as cfg
from src.DatBuilder.Util import match_name_in_group
from src.DatCode import Data, Logs, Instruments
from src.HDF import Util as HU

logger = logging.getLogger(__name__)


BASE_ATTRS = ['datnum', 'datname', 'dat_id', 'dattypes', 'config_name', 'date_initialized']


def get_dat_id(datnum, datname):
    """Returns unique dat_id within one experiment.
    (i.e. specific to whichever DF the dat is a member of)"""
    name = f'Dat{datnum}'
    if datname != 'base':
        name += f'[{datname}]'
    return name


class DatHDF(object):
    """Overall Dat object which contains general information about dat, more detailed info should be put
    into DatAttribute classes. Everything in this overall class should be useful for 99% of dats
    """
    version = '1.0'
    """
    Version history
        1.0 -- HDF based save files
    """

    def __init__(self, datnum: int, datname, dat_hdf, Data=None, Logs=None, Instruments=None, Entropy=None, Transition=None, DCbias=None):
        """Constructor for dat"""
        self.version = DatHDF.version
        self.config_name = cfg.current_config.__name__.split('.')[-1]
        self.dattype = None
        self.datnum = datnum
        self.datname = datname
        self.hdf = dat_hdf

        self.date_initialized = datetime.now().date()

        self.Data = Data
        self.Logs = Logs
        self.Instruments = Instruments
        self.Entropy = Entropy
        self.Transition = Transition
        self.DCbias = DCbias

    def __del__(self):
        self.hdf.close()  # Close HDF when object is destroyed


class NewDatLoader(abc.ABC):
    def __init__(self, datnum=None, datname=None, file_path=None):
        if file_path is not None:
            assert all([datnum is None, datname is None])
            self.hdf = h5py.File(file_path, 'r+')
        else:
            assert datnum is not None
            datname = datname if datname else 'base'
            self.hdf = h5py.File(HU.get_dat_hdf_path(get_dat_id(datnum, datname)))

        # Base attrs
        self.datnum = None
        self.datname = None
        self.dat_id = None
        self.dattypes = None
        self.config_name = None
        self.date_initialized = None

        self.get_Base_attrs()
        self.Data = Data.NewData(self.hdf)
        self.Logs = Logs.NewLogs(self.hdf)
        self.Instruments = None  # TODO: Replace with Instruments
        # self.Instruments = Instruments.NewInstruments(self.hdf)

    def get_Base_attrs(self):
        for key in BASE_ATTRS:
            setattr(self, key, self.hdf.attrs.get(key, None))

    @abc.abstractmethod
    def build_dat(self) -> DatHDF:
        """Override to add checks for Entropy/Transition etc"""
        return DatHDF(self.datnum, self.datname, self.hdf, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)


class NewDatBuilder(abc.ABC):
    """Base DatHDF builder class. Only contains the core DatCode Logs, Data, Instruments. Any others should be
    added in a subclass of this"""
    def __init__(self, datnum, datname, load_overwrite='load'):
        # Init with basic info at least - enough to Identify DatHDF
        assert load_overwrite in ['load', 'overwrite']
        overwrite = True if load_overwrite == 'overwrite' else False
        # Base attrs for Dat
        self.datnum = datnum
        self.datname = datname
        self.config_name = cfg.current_config.__name__.split('.')[-1]
        self.date_initialized = datetime.now().date()
        self.dat_id = get_dat_id(datnum, datname)
        self.dattypes = None

        self.hdf_path = HU.get_dat_hdf_path(self.dat_id, path=None, overwrite=overwrite)  # Location of My HDF which will store everything to do with dat
        self.hdf = h5py.File(self.hdf_path, 'r+')  # Open file in Read/Write mode

        self.copy_exp_hdf()  # Will copy Experiment HDF if not already existing

        # Init General Dat attributes to None
        self.Data = None  # type: Data.NewData
        self.Logs = None  # type: Logs.NewLogs
        self.Instruments = None  # type: Instruments.NewInstruments

        # Basic Inits which are sufficient if data exists in HDF already. Otherwise need to be built elsewhere
        self.init_Data()
        self.init_Logs()
        self.init_Instruments()

    def copy_exp_hdf(self):
        """Copy experiment HDF data into my HDF file if not done already"""
        if 'Exp_measured_data' not in self.hdf.keys() or 'Exp_metadata' not in self.hdf.keys():  # Only if first time
            hdfpath = CU.get_full_path(os.path.join(cfg.ddir, f'dat{self.datnum:d}.h5'))
            if os.path.isfile(CU.get_full_path(hdfpath)):  # Only if original HDF exists
                e_data = self.hdf.create_group('Exp_measured_data')
                with h5py.File(CU.get_full_path(hdfpath), 'r') as hdf:
                    for key in hdf.keys():
                        if isinstance(hdf[key], h5py.Dataset) and key not in self.hdf['Exp_measured_data'].keys():  # Only once
                            ds = hdf[key]
                            e_data[key] = ds[:]  # Make full copy of data to my HDF
                        elif isinstance(hdf[key], h5py.Group) and key not in self.hdf.keys():
                            hdf.copy(hdf[key], self.hdf, 'Exp_metadata')  # Make full copy of group to my HDF
                self.hdf.flush()  # writes changes to my HDF to file
            else:
                raise FileNotFoundError(f'Did not find HDF at {hdfpath}')

    def init_Base(self):
        """ For storing Base info in HDF attrs
        Note: dattypes won't be set here!"""
        hdf = self.hdf
        for attr, val in zip(BASE_ATTRS, [self.datnum, self.datname, self.dat_id, self.dattypes, self.config_name, self.date_initialized]):
            hdf.attrs[attr] = val

    @abc.abstractmethod
    def set_dattypes(self, value=None):
        """Reminder to set dattypes attr in HDF at some point"""
        self.dattypes = value if value else self.dattypes
        self.hdf['dattypes'] = self.dattypes

    def init_Data(self, setup_dict=None):
        """
        @param setup_dict: dict formatted as {<standard_name>:[<exp_name(s), multiplier(s), offset(s)*], ...}  *optional
            there MUST be a multiplier and offset for every possible exp_name
        @type setup_dict: Dict[list]
        @return: Sets attributes in Data
        @rtype: None
        """
        self.Data = self.Data if self.Data else Data.NewData(self.hdf)  # Will init Data from Dat HDF if already exists, otherwise will be blank init
        if setup_dict is not None:  # For initializing data into Dat HDF (Exp_data should already be located in 'Exp_measured_data' inside Dat HDF
            for item in setup_dict.items():  # Use Data.get_setup_dict to create
                standard_name = item[0]  # The standard name used in rest of this analysis
                info = item[1]  # The possible names, multipliers, offsets to look for in exp data  (from setupDF)
                exp_names = CU.ensure_list(info[0])  # All possible names in exp
                exp_names = ['Exp_'+exp_name for exp_name in exp_names]  # Stored with prefix in Dat HDF
                exp_name, index = match_name_in_group(exp_names, self.hdf['Exp_measured_data'])  # First name which matches a dataset in exp
                multiplier = info[1][index]  # Get the correction multiplier
                offset = info[2][index] if len(info) == 3 else 0  # Get the correction offset or default to zero
                if multiplier == 1 and offset == 0:  # Just link to exp data
                    self.Data.link_data(standard_name, exp_name)  # Hard link to data (so not duplicated in HDF file)
                else:  # duplicate and alter dataset before saving in HDF
                    data = self.Data.get_dataset(exp_name)[:]  # Get copy of exp Data
                    data = data*multiplier+offset  # Adjust as necessary
                    self.Data.set_data(standard_name, data)  # Store as new data in HDF
            self.hdf.flush()
        self.Data.get_from_HDF()  # Set up Data attrs (doesn't do much for Data)

    def init_Logs(self, json=None):
        self.Logs = self.Logs if self.Logs else Logs.NewLogs(self.hdf)
        if json is not None:
            group = self.Logs.group

            # Simple attrs
            Logs._init_logs_set_simple_attrs(group, json)

            # Instr attrs  # TODO: maybe want these part of subclass
            Logs._init_logs_set_srss(group, json)
            Logs._init_logs_set_babydac(group, json)
            Logs._init_logs_set_fastdac(group, json)

            # TODO: add mags
            # for i in range(1, cfg.current_config.instrument_num['mags']+1+1):
            #     if f'Mag...':
            #         pass

            self.hdf.flush()
        self.Logs.get_from_HDF()  # To put the values stored in Dat HDF into Logs attrs

    def init_Instruments(self):
        assert self.Logs is not None
        # TODO: copy links from relevant groups in logs to Instruments
        self.Instruments = self.Instruments if self.Instruments else Instruments.NewInstruments(self.hdf)
        self.Instruments.get_from_HDF()
        pass

    @abc.abstractmethod
    def build_dat(self) -> DatHDF:
        """Override if passing more info to NewDat (like any other DatCode"""
        return DatHDF(self.datnum, self.datname, self.hdf, Data=self.Data, Logs=self.Logs, Instruments=self.Instruments)

############## FIGURE OUT WHAT TO DO WITH/WHERE TO PUT

# predicted frequencies in power spectrum from dac step size
# dx = np.mean(np.diff(x))
# dac_step = 20000/2**16  # 20000mV full range with 16bit dac
# step_freq = meas_freq/(dac_step/dx)
#
# step_freqs = np.arange(1, meas_freq/2/step_freq)*step_freq
#
# fig, ax = plt.subplots(1)
# PF.Plots.power_spectrum(deviation, 2538 / 2, 1, ax, label='Average_filtered')
#
# # step_freqs = np.arange(1, meas_freq / 2 / 60) * 60
#
# for f in step_freqs:
#     ax.axvline(f, color='orange', linestyle=':')