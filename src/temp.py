from src.DatBuilder.Exp_to_standard import make_dat
from src.ExperimentSpecific.Jan20.Jan20ESI import JanESI
from src import CoreUtil as CU
import logging


CU.set_default_logging()

if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    dat = make_dat(2711, 'base', overwrite=False, dattypes=None,
                   ESI_class=JanESI, run_fits=True)
    # dat.Transition.run_row_fits()
    # dat.Transition.run_avg_fit()
    #
    # centers = [fit.best_values.mid for fit in dat.Transition.all_fits]
    # dat.Entropy.recalculate_entr(centers)
    # dat.Entropy.run_row_fits()
    # dat.Entropy.run_avg_fit()

