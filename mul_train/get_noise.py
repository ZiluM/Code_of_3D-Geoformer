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


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

class make_dataset2():
    """
    online reading dataset
    """
    def __init__(self, mypara,model_index = 0):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_pretr)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.all_group = mypara.all_group
        temp = data_in["temperatureNor"][
            :,
            :,
            mypara.lev_range[0] : mypara.lev_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if mypara.needtauxy:
            print("loading tauxy...")
            taux = data_in["tauxNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            self.field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.field_data = temp
            del temp
        self.max_idx = self.field_data.shape[0] * (self.field_data.shape[1] - self.output_length - self.input_length)
        self.model_index = model_index

    def __iter__(self):
        # st_min = self.input_length - 1
        # ed_max = self.field_data.shape[1] - self.output_length
        # for model_num in np.arange(self.field_data.shape[0]):
        model_num = self.model_index
        for indx in range(0,self.field_data.shape[1]-self.output_length-self.input_length):
            dataX = self.field_data[model_num, indx : indx + self.input_length]
            dataY = self.field_data[model_num, indx + self.input_length : indx + self.input_length + self.output_length]
            # print(model_num, indx)
            yield dataX[np.newaxis,...], dataY[np.newaxis,...]


    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "temp lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }


log = get_logger()



# i_file = "/glade/work/zilumeng/3D_trans/model/Geoformer_beforeTrans.pkl"
i_file = "/glade/work/zilumeng/3D_trans/model/3x4p_1.pkl"
model_name = i_file.split("/")[-1].split(".")[0]
# print(i_file)
log.info("loading model from {}".format(i_file))
mymodel = Geoformer(mypara).to(mypara.device)
mymodel.load_state_dict(torch.load(i_file))
mymodel.eval()

log.info("loading data from {}".format(mypara.adr_pretr))
dataset = make_dataset2(mypara,model_index=0)
model_num = dataset.field_data.shape[0]

for model_index in range(model_num):
    dataset = make_dataset2(mypara,model_index=model_index)

    # DataLoader(
    #         dataCS, batch_size=mypara.batch_size_eval, shuffle=False
    #     )
    resls = []
    predls = []
    log.info(f"predicting...{model_index}")
    idx = 0
    for dataX, dataY in dataset:
        log.info(f" =====================predicting...{idx} ========================")
        with torch.no_grad():
            # print(i,j)
            # dataX, dataY = dataset.__iter__()
            dataX = torch.from_numpy(dataX).float().to(mypara.device)
            log.info("shape of dataX: {}".format(dataX.shape))
            # dataY = torch.from_numpy(dataY).float().to(mypara.device)
            # print(dataX.shape, dataY.shape)
            pred = mymodel(dataX,
                    predictand=None,
                    train=False,)
            pred = pred.cpu().detach().numpy()
            log.info("shape of pred: {}".format(pred.shape))
            noise = dataY - pred
            resls.append(noise)
            predls.append(pred)
            log.info("shape of noise: {}".format(noise.shape))
            del dataX, dataY,pred
        idx += 1
    resls = np.concatenate(resls,axis=0)
    predls = np.concatenate(predls,axis=0)
    log.info(f"resls_shape: {resls.shape}, predls_shape: {predls.shape}")
    log.info(f" =====================saving...{idx} ========================")
    np.save(f"./data/noise1/noise_{model_name}_{model_index}.npy",resls)
    np.save(f"./data/noise1/pred_{model_name}_{model_index}.npy",predls)
    log.info(f" =====================saved...{idx} ========================")

    # idx += 1
    # if idx % 1000 == 0:
        # log.info(f" =====================saving...{idx} ========================")
        # resls = np.concatenate(resls,axis=0)
        # np.save("./data/noise_cmip6_{}.npy".format(idx),resls)
        # resls = []