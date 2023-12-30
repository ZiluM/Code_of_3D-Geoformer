import sys
sys.path.append("/glade/work/zilumeng/SSNLIM")
from slim import *
import pickle
import matplotlib.pyplot as plt
import cftime
import numpy as np
import pandas as pd
import xarray as xr
from EOF import EOF


with open("./LIM/eof.pkl","rb") as f:
    sst_svd = pickle.load(f)

pcs_num = 40
pcs = sst_svd.pcs[:]