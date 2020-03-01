from src.Sandbox import *
datdf = DF.DatDF()



dat = make_dat_standard(420, dfoption='load')
np.nancumsum(dat.Entropy.entrav)