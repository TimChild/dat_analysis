import pickle
import inspect
from unittest.mock import patch
from src import temp
import pandas as pd
import numpy as np
import src.DFcode.DFutil as DU




class testclass(object):
    def __init__(self):
        self.a = 10
        self.b = 20
        self.c = 50

    def add(self):
        print(f'a + b = {self.a+self.b}')

    def multiply(self):
        print(f'a*b = {self.a*self.b}')

    def addc(self):
        print(f'a+c = {self.a+self.c}')


if __name__ == '__main__':
    with open('testpkl.pkl', 'rb') as f:
        two = pickle.load(f)
