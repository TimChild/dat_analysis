import numpy as np
from typing import List, Callable, Union, Optional, Any
import lmfit as lm
import logging

from . import dat_attribute as DA
from ...analysis_tools.transition import default_transition_params, i_sense, get_param_estimates

logger = logging.getLogger(__name__)


class Transition(DA.FittingAttribute):
    version = '2.0.0'
    group_name = 'Transition'
    description = 'Fitting to charge transition (measured by charge sensor qpc). Expects data with name "i_sense"'
    DEFAULT_DATA_NAME = 'i_sense'

    def default_data_names(self) -> List[str]:
        return ['x', 'i_sense']

    def clear_caches(self):
        super().clear_caches()

    def get_centers(self) -> List[float]:
        logger.info(f'Dat{self.dat.datnum}: Starting Transition Center Fits')
        return [fit.best_values.mid for fit in self.row_fits]

    def get_default_params(self, x: Optional[np.ndarray] = None,
                           data: Optional[np.ndarray] = None) -> Union[List[lm.Parameters], lm.Parameters]:
        if x is not None and data is not None:
            params = get_param_estimates(x, data)
            if len(params) == 1:
                params = params[0]
            return params
        else:
            return default_transition_params()

    def get_default_func(self) -> Callable[[Any], float]:
        return i_sense

    def initialize_additional_FittingAttribute_minimum(self):
        pass

