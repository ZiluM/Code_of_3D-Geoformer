from Geoformer import Geoformer
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from func_for_prediction import make_dataset_test
from myconfig import mypara
from perlin_numpy import generate_perlin_noise_3d
# Load the model

# param = 



mypara.batch_size = 1


lead_max = mypara.output_length
adr_datain = (
    # "./data/GODAS_group_up150_temp_tauxy_8021_kb.nc"
    "./data/GODAS_group_up150_temp_tauxy_1980_2021_kb.nc"
)
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_1980_2021_kb.nc"
data_ori = xr.open_dataset(adr_oridata)


temp_ori_region = data_ori["temperatureNor"][
    :,
    mypara.lev_range[0] : mypara.lev_range[1],
    mypara.lat_range[0] : mypara.lat_range[1],
    mypara.lon_range[0] : mypara.lon_range[1],
].values

dataCS = make_dataset_test(
    address=adr_datain,
    needtauxy=mypara.needtauxy,
    lev_range=mypara.lev_range,
    lon_range=mypara.lon_range,
    lat_range=mypara.lat_range,
)
dataloader_test = DataLoader(
    dataCS, batch_size=mypara.batch_size, shuffle=False
)

adr_model = "model/Geoformer_beforeTrans.pkl"
mypara.output_length = 1
mymodel = Geoformer(mypara).to(mypara.device)
mymodel.load_state_dict(torch.load(adr_model))
mymodel.eval()


test_group = len(dataCS)

if mypara.needtauxy:
    n_lev = mypara.lev_range[1] - mypara.lev_range[0] + 2
    sst_lev = 2
else:
    n_lev = mypara.lev_range[1] - mypara.lev_range[0]
    sst_lev = 0


ii = 0
iii = 0
free_run_max = 720*3
test_num = 30
# for i in range(test_num):
#     first_var = dataloader_test.__iter__().__next__()
#     print(first_var[0,0,0])
dataloader_test = iter(dataloader_test)
while iii < 10:
    first_var = next(dataloader_test)
    iii += 1
    print(first_var[0,0,0])

free_run_ls = []
with torch.no_grad():
    while ii < free_run_max:
        print(first_var.shape)
        out_var = mymodel(
            first_var.float().to(mypara.device),
            predictand=None,
            train=False,
        )
        res = out_var[:,[0]]
        res = res.float().to(mypara.device)
        # torch.normal(0,1,size=res.shape).to(mypara.device)
        first_var = first_var.float().to(mypara.device)
        noise = torch.from_numpy(generate_perlin_noise_3d(res.shape[2:], (1, 3, 3))).to(mypara.device) 
        print(noise.shape)
        print(noise[:,:10])
        first_var = torch.concatenate([first_var[:,:-1],res + noise],dim=1)
        # print(first_var.shape)
        ii = ii + 1
        free_run_ls.append(out_var.cpu().detach().numpy()[np.newaxis,...])
        print(ii)
free_run_ls = np.concatenate(free_run_ls,axis=0)
np.save("./data/free_run_ls3_noise.npy",free_run_ls)
print(free_run_ls.shape)
print("======save======")
    # for input_var in dataloader_test:
    #     print(input_var.shape)
    #     out_var = mymodel(
    #         input_var.float().to(mypara.device),
    #         predictand=None,
    #         train=False,
    #     )
    #     print(out_var.shape)
    #     ii += out_var.shape[0]
    #     if torch.cuda.is_available():
    #         var_pred[iii:ii] = out_var.cpu().detach().numpy()
    #     else:
    #         var_pred[iii:ii] = out_var.detach().numpy()
    #     iii = ii