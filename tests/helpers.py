import src.Configs.Main_Config as cfg
import functools
import inspect
import os



def stack_inspector():
    """Prints out current stack with index values"""
    stack = inspect.stack()
    for i, frame in enumerate(stack):
        for j, val in enumerate(frame):
            print(f'[{i}][{j}] = {val},', end='\t')
        print('')

