from typing import Union

import src.DatBuilder
from src.DatCode import Entropy as E, Transition as T, DCbias as DC
from src.DatBuilder import DatHDF


class EntropyDatLoader(DatHDF.NewDatLoader):
    """For loading dats which may have any of Entropy, Transition, DCbias"""
    def __init__(self, datnum=None, datname=None, file_path=None):
        super().__init__(datnum, datname, file_path)
        if 'entropy' in self.dattypes:
            self.Entropy = E.NewEntropy(self.hdf)
        if 'transition' in self.dattypes:
            self.Transition = T.NewTransitions(self.hdf)
        if 'dcbias' in self.dattypes:
            self.DCbias = DC.NewDCBias(self.hdf)

    def build_dat(self) -> src.DatBuilder.DatHDF.DatHDF:
        return src.DatBuilder.DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments,
                                            self.Entropy, self.Transition, self.DCbias)


class EntropyDatBuilder(DatHDF.NewDatBuilder):
    """For building dats which may have any of Entropy, Transition, DCbias"""
    def __init__(self, datnum, datname, dfname='default'):
        super().__init__(datnum, datname, dfname)
        self.Transition: Union[T.NewTransitions, None] = None
        self.Entropy: Union[E.NewEntropy, None] = None
        self.DCbias = None

    def set_dattypes(self, value=None):
        """Just need to remember to call this to set dattypes in HDF"""
        super().set_dattypes(value)

    def init_Entropy(self, center_ids):
        """If center_ids is passed as None, then Entropy.data (entr) is not initialized"""
        self.Entropy = self.Entropy if self.Entropy else E.NewEntropy(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        entx = self.Data.get_dataset('entx')
        enty = self.Data.get_dataset('enty')
        E.init_entropy_data(self.Entropy.group, x, y, entx, enty, center_ids=center_ids)

    def init_Transition(self):
        self.Transition = self.Transition if self.Transition else T.NewTransitions(self.hdf)
        x = self.Data.get_dataset('x_array')
        y = self.Data.get_dataset('y_array')
        i_sense = self.Data.get_dataset('i_sense')
        T.init_transition_data(self.Transition.group, x, y, i_sense)

    def init_DCbias(self):
        pass  # TODO: Finish this one

    def build_dat(self):
        return src.DatBuilder.DatHDF.DatHDF(self.datnum, self.datname, self.hdf, self.Data, self.Logs,
                                            self.Instruments,
                                            self.Entropy, self.Transition, self.DCbias)


