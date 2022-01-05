# Getting started with dat_analysis

1. Download Git Repo
   1. Clone repo from [GitHub](https://github.com/TimChild/dat_analysis) (easist to use [GitHub Desktop](https://desktop.github.com/))
   2. Make a new branch for yourself
2. Make a new config file in `src/dat_analysis/data_standardize/exp_specific/`
   1. Name should specify CD date and Computer (e.g. `Nov21_TimPC.py`)
      1. Probably easiest to copy and paste a similar (recent) one and modify from there
      2. Main things it should contain are:
         1. Subclass of ExpConfigBase (For Experiment specific things)
         2. Subclass of SysConfigBase (For System specific things)
         3. Subclass of Exp2HDF (What is interacted with in program (i.e. nothing is ever asked of ExpConfig or SysConfig directly (I think) so overriding here should be sufficient to change behaviour)
3. Set new Exp2HDF as default in `src/dat_analysis/dat_object/make_dat.py` by changing `default_Exp2HDF` to point to the new subclass of Exp2HDF:
   ``` 
   from dat_analysis.data_standardize.exp_specific.Nov21_TimPC import Nov21_TimPC_Exp2HDF
   default_Exp2HDF = Nov21_TimPC_Exp2HDF
   ```
4. Make a new Anaconda environment and install dat_analysis
   1. Install [Anaconda](https://www.anaconda.com/products/individual)
   2. Create a new conda environment using `conda create -n <name> python=3.10` in a conda terminal
   3. Activate the environment using `conda activate <name>`
   4. Install `dat_analysis` using `pip install <path to dat_analysis repo>`
      1. To install in an editable mode add the `-e` flag
5. Basic usage:
   1. Using the conda environment: 
       ```
      from dat_analysis.dat_object.make_dat import get_dat, get_dats
   
      dat = get_dat(100)  # Gets dat100.h5
      dats = get_dats(range(100, 110, 2))  # Gets every other dat from 100 to 110
      ```

## What you can do with a dat:

A ‘dat’ as returned from DatHander.get_dat(datnum) is the Python interface to a DatHDF in the directory specified in the Config classes. The DatHDF is where everything to do with the dat is saved. All information is looked for there first before being generated and saved. Note that for increased performance, most things are cached when read from the HDF file, so just changing the HDF directly will not necessarily update in a python runtime unless caches are cleared or the runtime is restarted.
The ‘dat’ itself has some normal attributes like ‘datnum’ and ‘date_initialized’ but by itself is very minimalistic. Additional functions and information are generally stored in “DatAttributes” which are properties of the ‘dat’. (e.g. `dat.Data` gets you to the “Data” “DatAttribute”.) 

In general, the DatAttributes are only generated when asked for, and if you ask for something which can’t be generated an error will be raised which hopefully will give some clue as to what was missing in order to create it (e.g. missing ‘i_sense’ data). 
Once created, they are automatically saved in the HDF file, so if you ask for the same thing again later, it will load from the HDF file rather than creating it again. 
In general, it should always be safe to delete a whole DatAttribute (a ‘group’ in the HDF) from the HDF file (either through python or just in the HDF viewer) if you want to start that part again from scratch without losing everything else. 

Currently the DatAttributes which you can use are:

### Data
Should be able to load anything from experiment file, and also makes it easy to save new data.  (Note: to avoid duplicating data for minor changes unnecessarily I use DataDescriptor's. Mostly you don't have to worry about it, but if for example you want the to change some raw data to be 10x larger so it's in units of nA for example, then it is more efficient to change only the data descriptor to say that when loading the data should be multiplied by 10 rather than saving a whole copy of the data with every value just 10x larger. 

### Transition
This makes it easier to do fitting and save multiple different fits. Use dat.Transition.get_fit(...) to generate new fits or load old ones (if they don't exist they'll be created). Or dat.Transition.avg_fit and dat.Transition.row_fits will return fits using default fitting parameters. By default it will look for ‘i_sense’ data (which is currently looked for in the experiment file under the names ‘cscurrent’ or ‘cscurrent_2d’ based on the `ExpConfigBase.get_default_dat_info()`… (I.e. if ‘cscurrent’ is saved in the Experiment file, ‘i_sense’ will exist in the DatHDF). 

### Entropy
Very similar to Transition but for entropy data. 
Note: This will by default use the centers determined by SquareEntropy or dat.Transition.get_centers() (which itself comes from dat.Transition.row_fits) in order to average 2D entropy data into a 1D array for dat.Entropy.avg_fit. This is done because the center positions from fits to transition data are usually more accurate than the center position from fits to entropy data.

### Logs
Quick and nicer access to all the things stored in sweeplogs of original experiment file. 
Note: ‘dat.Logs.sweeplogs’ will return the full sweeplogs dict (after substitutions etc specified in config files). So if something is missing, you can look there to see if the thing is being stored under a different name than expected (i.e. someone may have changed how IGOR stores the data). In that case, either add a temporary fix to ConfigFiles (can be done for specific datnums) or add a new field following the pattern of all the rest (e.g. add to Logs.initialize_minimum() and add a property right under Logs.__init__() and then re-run Logs.initialize_minimum()) 

### AWG (Arbitrary Wave Generator of FastDAC)
This contains information about what the FastDAC AWG was doing during the scan, also has some helpful functions for making masks etc. For SquareWave AWGs specifically, see dat.SquareEntropy.square_awg (which mostly copies info from AWG, but forces the AWG to be a 4 point wave (i.e. bunches all of the ramping parts into the main setpoints)). 

### Square Entropy
This handles the conversion of charge sensor data into an entropy signal and stores all the relevant information along the way (i.e. setpoint averaged, cycled etc). It’s mostly comprised of three parts: Input, ProcessParams, Output. You can generate any of them by using SquareEntropy.get_<thing>(…) and any params not specified will be generated automatically. Input is the data that is required to calculate square entropy stuff, process params are the things that affect how processing happens (i.e. where to start and finish averaging). Outputs contain all the different steps of processing and are saved to the HDF by default. The Inputs/ProcessParams are not saved to HDF by default because they are very quick to generate, but can be saved by specifying ‘save_name’ and can be loaded by specifying ‘name’ (see docs for more info). 
Note: No Entropy fitting is done here, that should be done from dat.Entropy (which will automatically use square entropy data if available). For more control, pass in the necessary arguments to dat.Entropy.get_fit(…) (e.g. pass in the x, and entropy signal you want to use). 

### Figures
(this is pretty new and subject to change when I get working on the Dash app some more). The idea of this is to save figures in the DatHDF as well, that way figures can be passed between python runtimes easily, and also viewed from HDF more easily. 
