# Example folder for Analyzed Data
I.e. the .h5 files that are saved from `dat_analysis`

Note: It is safe to delete files in here (although you'll lose any saved anlaysis of course)

If deleted (or not present in the first place), when loading a new experiment `.h5` file with `dat_analysis`, a standardized version will be created here.

If there already is a corresponding analyzed `.h5` file here, and `overwrite=False`, then the experiment `.h5` will not be touched, and the file here will be loaded directly.


By "standardized", I mean that the format of the `.h5` file created by `dat_analysis` is quite different to the `.h5` files generated by IgorAcq, and should always be the same even if the experiment `.h5` files have changed format due to changes in IgorAcq.