from myconfig import mypara
import numpy as np
from copy import deepcopy
import xarray as xr
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction import func_pre
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import torch
from Geoformer import Geoformer
import logging
import sys

sys.path.append("/glade/work/zilumeng/SSNLIM")
from EOF import EOF


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = get_logger()

for lead_month in range(20):
    logger.info("lead_time:"+str(lead_month)+'===================='*3)
    lead_month_ls = []
    for model_idx in range(23):
        logging.info("model_index:"+str(model_idx)+'===================='*1)
        # path1 = f"./data/noise1/pred_3x4p_1_{model_idx}.npy"
        path0 = f"./data/noise1/noise_3x4p_1_{model_idx}.npy"
        noise = np.load(path0)[:, lead_month]
        lead_month_ls.append(noise)
        logger.info("noise.shape:" + str(noise.shape))
        del noise
    lead_month_ls = np.concatenate(lead_month_ls, axis=0)
    logger.info("lead_month_ls.shape:" + str(lead_month_ls.shape))
    eof = EOF(lead_month_ls)
    eof.solve(method="dask_svd",chunks=(1000),dim_min =400)
    logger.info("EOF done!")
    pc = eof.get_pc()
    pt = eof.get_pt()
    logger.info("pc shape:{}".format(pc.shape))
    logger.info("pt shape:{}".format(pt.shape))
    logger.info("var_perc:{}".format(eof.get_varperc()))
    # print("var_perc:",eof.get_varperc())
    eof.save(f"./data/noise1/eof/3x4p_1_leadmonth{lead_month}.pkl")
    # eof.save("./LIM/eof.pkl") 

    logger.info("EOF saved!" + str(lead_month))
