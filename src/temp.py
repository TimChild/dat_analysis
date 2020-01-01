import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np

def replacementfn(inp):
    print(f'Im the replacement and I got the input: {inp}')
    return 'fake answer'


    

