from src.Scripts.StandardImports import *


from src.Configs import Jan20Config
jan20_ent_datnums = [1492,1495,1498,1501,1504,1507,1510,1513,1516,1519,1522,1525,1528,1533,1536,1539,1542,1545,1548,1551,1554,1557,1560,1563,1566]
jan20_dcdat = [1529]  # Theta vs DCbias

from src.Configs import Mar19Config
mar19_ent_datnums = list(range(2689, 2711+1, 2))  # Not in order
mar19_trans_datnums = list(range(2688, 2710+1, 2))  # Not in order
mar19_dcdats = list(range(3385, 3410+1, 2))  # 0.8mV steps. Repeat scan of theta at DC bias steps
mar19_dcdats2 = list(range(3386, 3410+1, 2))  # 3.1mV steps. Repeat scan of theta at DC bias steps
mar19_datdf = DF.DatDF()

from src.Configs import Sep19Config
sep19_ent_datnums = [2227, 2228, 2229]  # For more see OneNote (Integrated Entropy (Nik v2) - Jan 2020/General Notes/Datasets for Integrated Entropy Paper)
sep19_dcdat = [1947]  # For more see OneNote ""


if __name__ == '__main__':
    dat = make_dat_standard()

# TODO: Make wrapper that optionally takes config as an argument and temporarily changes the main config whilst executing wrapped function
# TODO: Make sure that dats will be saved to correct place regardless of current state of config. Not sure how much I rely on things like ddir outside of Core
# TODO: Test loading some dats from each different experiment, will require lots of changes in specific config files probably