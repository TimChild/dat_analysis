from src.Scripts.StandardImports import *


if __name__ == '__main__':
    datnums = list(range(158, 159+1))

    for datnum in datnums:
        print(f'\nStarting Dat{datnum}:\n')
        dat = get_dat(datnum, overwrite=False, run_fits=True)
