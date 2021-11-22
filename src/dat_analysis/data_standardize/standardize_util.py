import threading
import time
import logging

logger = logging.getLogger(__name__)


def wait_for(datnum, ESI_class=None):
    def _wait_fn(num):
        esi = ESI_class(num)
        while True:
            found = esi._check_data_exists(suppress_output=True)
            if found:
                print(f'Dat{num} is ready!!')
                break
            else:
                time.sleep(10)

    if ESI_class is None:
        from dat_analysis.dat_object.make_dat import default_Exp2HDF
        ESI_class = default_Exp2HDF

    x = threading.Thread(target=_wait_fn, args=(datnum,))
    x.start()
    print(f'A thread is waiting on dat{datnum} to appear')
    return x

