from src.Scripts.StandardImports import *

from src.DatObject.Make_Dat import make_dat
from src.DataStandardize.ExpSpecific import Jan20

# old_dat = make_dat(2713, 'base', overwrite=True, ESI_class=Jan20.JanESI, run_fits=False)

dat = make_dat(95, 'base')
