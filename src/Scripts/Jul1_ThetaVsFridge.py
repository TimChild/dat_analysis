from src.Scripts.StandardImports import *


if __name__ == '__main__':
    datnums = list(range(100, 105+1))

    for datnum in datnums:
        dat = get_dat(datnum, overwrite=False, run_fits=True)
