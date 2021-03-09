from concurrent.futures import ProcessPoolExecutor
from src.DatObject.Make_Dat import get_dat

pool = ProcessPoolExecutor()


def do_multiprocessed(datnum: int, **kwargs):
    """Only pass in datnum because that is an int and pickleable (plus any other args/kwargs which are pickleable (and not big)"""
    dat = get_dat(datnum)
    fit = dat.Transition.get_fit(which='avg', name='test',
                                 check_exists=False)
    return fit


if __name__ == '__main__':
    """I think it's important to call this in main because otherwise each process tries to also start processes"""
    fits = list(pool.map(do_multiprocessed, range(1604, 1635,
                                                  2)))  # Fit all even dats between 1604 and 1635 multiprocessed (fits are collected, but could also be accessed through dat.... after this has run)
