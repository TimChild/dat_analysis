from unittest import TestCase
import pandas as pd
from src.DFcode import DFutil
from unittest.mock import patch
from io import StringIO



class Test(TestCase):
    def test_protect_data_from_reindex(self):
        pd.DataFrame.set_index = DFutil.protect_data_from_reindex(pd.DataFrame.set_index)
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'col3': [5, 6]})
        df.set_index(['col1', 'col2'], inplace=True)
        with patch('sys.stdout', new_callable=StringIO) as mock:
            df.set_index(['col3'], inplace=True)
        print(mock.getvalue())
        self.assertEqual('WARNING', mock.getvalue()[:7])


