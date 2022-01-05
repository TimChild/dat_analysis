import threading
import time
import logging
from deprecation import deprecated

logger = logging.getLogger(__name__)


@deprecated(details="2022/01--Don't think this function is used, plan to remove")
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
        from ..dat_object.make_dat import default_Exp2HDF
        ESI_class = default_Exp2HDF

    x = threading.Thread(target=_wait_fn, args=(datnum,))
    x.start()
    print(f'A thread is waiting on dat{datnum} to appear')
    return x

