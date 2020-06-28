from src.DatObject.Make_Dat import make_dat
from src.DataStandardize.ExpSpecific.Jan20 import JanESI
from src import CoreUtil as CU

CU.set_default_logging()

import abc


class SettersMixin(object):
    a = None  # type: int
    b = None  # type: int

    def _set_a(self):
        print(self.a)

    def _set_b(self):
        print(self.b)

    @staticmethod
    def _set_c():
        print('c')


class A(abc.ABC, SettersMixin):

    def __init__(self):
        self.a = 1
        self.b = 2
        self._set_a()


class SettersMixinOverride(object):
    @staticmethod
    def _set_b():
        print('overridden with mixing')


class B(SettersMixinOverride, A):
    @staticmethod
    def _set_a(**kwargs):
        print('overridden')





if __name__ == '__main__':

    b = B()


    # logging.basicConfig(level=logging.DEBUG)
    dat = make_dat(2711, 'base', overwrite=True, dattypes=None,
                   ESI_class=JanESI, run_fits=True)
    # dat.Transition.run_row_fits()
    # dat.Transition.run_avg_fit()
    #
    # centers = [fit.best_values.mid for fit in dat.Transition.all_fits]
    # dat.Entropy.recalculate_entr(centers)
    # dat.Entropy.run_row_fits()
    # dat.Entropy.run_avg_fit()

    pass
