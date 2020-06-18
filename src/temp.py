from src.DatBuilder.Exp_to_standard import make_dat
from src.ExperimentSpecific.Jan20.Jan20ESI import JanESI


if __name__ == '__main__':
    dat = make_dat(2711, 'base', overwrite=True, dattypes=None,
                   ESI_class=JanESI, run_fits=True)

