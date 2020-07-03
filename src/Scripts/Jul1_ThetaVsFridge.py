from src.Scripts.StandardImports import *

import src.DatObject.DatBuilder as DB
from src.DataStandardize.ExpSpecific.Jun20 import JunESI
from src.DataStandardize.ExpSpecific.Jan20 import JanESI
import src.HDF_Util as HDU

if __name__ == '__main__':

    CU.sub_poly_from_data()

    datnums = set(range(117, 213+1))
    datnums = datnums - set(list(range(160, 186+1)))
    for datnum in datnums:
        print(f'\nStarting Dat{datnum}:\n')
        dat = get_dat(datnum, overwrite=False, run_fits=True)

        # Fixes if necessary
        Jun20.Fixes.add_full_sweeplogs(dat)
        Jun20.Fixes.log_temps(dat)

    dats = [get_dat(num) for num in datnums]


