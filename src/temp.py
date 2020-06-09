from src.Scripts.StandardImports import *
# from src.Sandbox import *
import src.DatCode.Transition as T
import src.DatCode.Entropy as E
import h5py
h5py.enable_ipython_completer()


def printer(name):
    print(name)
    return None


class TestClass(object):
    def __init__(self):
        self.f = h5py.File('Z:/temp/test.hdf5', 'r')

    def __del__(self):
        self.f.close()  # Close HDF when object is cleaned up


if __name__ == '__main__':
    data = np.linspace(0, 10, 100)
    t = TestClass()
    # t.f['Group1']['bla'] = np.array([1, 2, 3])



