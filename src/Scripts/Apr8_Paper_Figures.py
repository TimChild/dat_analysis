from src.Scripts.StandardImports import *
from typing import Union
import functools
import inspect



def switch_config_decorator_maker(config, folder_containing_experiment=None):
    """
    Decorator Maker - makes a decorator which switches to given config and back again at the end

    @param config: config file to switch to temporarily
    @type config: module
    @return: decorator which will switch to given config temporarily
    @rtype: function
    """

    def switch_config_decorator(func):
        """
        Decorator - Switches config before a function call and returns it back to starting state afterwards

        @param func: Function to wrap
        @type func: function
        @return: Wrapped Function
        @rtype: function
        """

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            if config != cfg.current_config:  # If config does need changing
                old_config = cfg.current_config  # save old config module (probably current experiment one)
                old_folder_containing_experiment = cfg.current_folder_containing_experiment
                cfg.set_all_for_config(config, folder_containing_experiment)
                result = func(*args, **kwargs)
                cfg.set_all_for_config(old_config, old_folder_containing_experiment)  # Return back to original state
            else:  # Otherwise just run func like normal
                result = func(*args, **kwargs)
            return result

        return func_wrapper

    return switch_config_decorator


def wrapped_call(decorator, func):
    result = decorator(func)()
    return result


def test_decorator(func):
    def wrapper(*args, **kwargs):
        print('before')
        result = func(*args, **kwargs)
        print('after')
        return result
    return wrapper


from src.Configs import Jan20Config
jan20_ent_datnums = [1492,1495,1498,1501,1504,1507,1510,1513,1516,1519,1522,1525,1528,1533,1536,1539,1542,1545,1548,1551,1554,1557,1560,1563,1566]
jan20_dcdat = [1529]  # Theta vs DCbias

from src.Configs import Mar19Config
mar19_config_switcher = switch_config_decorator_maker(Mar19Config)
mar19_ent_datnums = list(range(2689, 2711+1, 2))  # Not in order
mar19_trans_datnums = list(range(2688, 2710+1, 2))  # Not in order
mar19_dcdats = list(range(3385, 3410+1, 2))  # 0.8mV steps. Repeat scan of theta at DC bias steps
mar19_dcdats2 = list(range(3386, 3410+1, 2))  # 3.1mV steps. Repeat scan of theta at DC bias steps
mar19_datdf = wrapped_call(mar19_config_switcher, DF.DatDF)  # type: DF.DatDF
mar19_make_dat = mar19_config_switcher(make_dat_standard)

from src.Configs import Sep19Config
sep19_config_switcher = switch_config_decorator_maker(Sep19Config)
sep19_ent_datnums = [2227, 2228, 2229]  # For more see OneNote (Integrated Entropy (Nik v2) - Jan 2020/General Notes/Datasets for Integrated Entropy Paper)
sep19_dcdat = [1947]  # For more see OneNote ""
sep19_datdf = wrapped_call(sep19_config_switcher, DF.DatDF) # type: DF.DatDF
sep19_make_dat = sep19_config_switcher(make_dat_standard)



class test():
    def __init__(self):
        print(f'init with cfg.ddir = {cfg.ddir}')

    def print_test(self):
        print(cfg.ddir)



if __name__ == '__main__':
    t = test()
    # sdat = sep19_make_dat(1947)
    mdat = mar19_make_dat(3385, dattypes='transition', dfoption='overwrite')
    # dat = make_dat_standard(1529, dfoption='load')
    mar19_datdf.update_dat(mdat)
    mar19_datdf.save()

# TODO: Make sure that dats will be saved to correct place regardless of current state of config. Not sure how much I rely on things like ddir outside of Core
# TODO: Test loading some dats from each different experiment, will require lots of changes in specific config files probably