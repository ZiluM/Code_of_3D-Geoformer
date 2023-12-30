import sys, os

import numpy as np
import logging
import xarray as xr
sys.path.append("/glade/work/zilumeng/SSNLIM")
sys.path.append("/glade/work/zilumeng/3D_trans/Code")
from myconfig import mypara
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

class make_dataset2():
    """
    online reading dataset
    """
    def __init__(self, mypara):
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

    def __iter__(self):
        # st_min = self.input_length - 1
        # ed_max = self.field_data.shape[1] - self.output_length
        for model_num in np.arange(self.field_data.shape[0]):
            for indx in range(0,self.field_data.shape[1]-self.output_length-self.input_length):
                dataX = self.field_data[model_num, indx : indx + self.input_length]
                dataY = self.field_data[model_num, indx + self.input_length : indx + self.input_length + self.output_length]
                print(model_num, indx)
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
    
loger = get_logger()
loger.info("loading data...")
data = make_dataset2(mypara).field_data


model_num = data.shape[0]
month = data.shape[1]
loger.info("data shape:{}".format(data.shape))

# data = data.reshape((model_num*month, *data.shape[2:]))
# data = data
loger.info("data shape:{}".format(data.shape))
data_ls = []
for i in range(model_num):
    model_data = data[i]
    data_ls.append(model_data)
data = np.concatenate(data_ls,axis=0)
loger.info("data shape:{}".format(data.shape))
# loger.info("data shape:{}".format(data.shape))
loger.info("Start EOF...")
eof = EOF(data)
eof.solve(method="dask_svd",chunks=(1000),dim_min =30)

loger.info("EOF done!")
pc = eof.get_pc()
pt = eof.get_pt()
loger.info("pc shape:{}".format(pc.shape))
loger.info("pt shape:{}".format(pt.shape))
loger.info("var_perc:{}".format(eof.get_varperc()))
# print("var_perc:",eof.get_varperc())
eof.save("./LIM/eof.pkl")
# eof.save("./LIM/eof.pkl") 

loger.info("EOF saved!")


# data = 