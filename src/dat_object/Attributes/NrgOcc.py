from typing import List, Callable, Any, Optional, Union

import lmfit as lm
import numpy as np
import logging

from src.dat_object.Attributes.DatAttribute import FittingAttribute

logger = logging.getLogger(__name__)


class NrgOcc(FittingAttribute):
    version = '1.0.0'
    group_name = 'Nrg Occ'
    description = 'Fitting to Occupation (i_sense) NRG data'
    DEFAULT_DATA_NAME = 'i_sense'
    FIT_METHOD = 'powell'

    def get_default_params(self, x: Optional[np.ndarray] = None, data: Optional[np.ndarray] = None) -> Union[
        List[lm.Parameters], lm.Parameters]:
        # TODO: Params might be a list.. need to do that
        params = self.dat.Transition.get_default_params(x=x, data=data)  # These estimates should be roughly the same
        params.add_many(
            ('g', 1, True, 0, 300),
            ('lin_occ', 0, True, -0.1, 0.1)
        )
        return params

    def get_default_func(self) -> Callable[[Any], float]:
        from src.analysis_tools.nrg import NRG_func_generator
        return NRG_func_generator('i_sense')

    def default_data_names(self) -> List[str]:
        return ['x', 'i_sense']

    def clear_caches(self):
        super().clear_caches()

    def get_centers(self) -> List[float]:
        logger.info(f'Dat{self.dat.datnum}: Starting NRG Centter Fits')
        return [fit.best_values.mid for fit in self.row_fits]

    def initialize_additional_FittingAttribute_minimum(self):
        pass


