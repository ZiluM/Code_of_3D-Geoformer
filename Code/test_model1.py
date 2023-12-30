from myconfig import mypara
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction import func_pre

mpl.use("Agg")
# plt.rc("font", family="Arial")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


# --------------------------------------------------------
files = file_name("./model")
file_num = len(files)
lead_max = mypara.output_length
adr_datain = (
    # "./data/GODAS_group_up150_temp_tauxy_8021_kb.nc"
    "./data/GODAS_group_up150_temp_tauxy_1980_2021_kb.nc"
)
# adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_kb.nc"
adr_oridata = "./data/GODAS_up150m_temp_nino_tauxy_1980_2021_kb.nc"
# ---------------------------------------------------------
num = 0
# for i_file in files[: file_num + 1]:
i_file = "/glade/work/zilumeng/3D_trans/model/3x4p_1.pkl"
print(i_file)
model_name = "3x4p_1"
# num += 1
# fig1 = plt.figure(figsize=(5, 2.5), dpi=300)
# ax1 = fig1.add_subplot(1, 2, 1)
# ax2 = fig1.add_subplot(1, 2, 2)
(cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true,) = func_pre(
    mypara=mypara,
    adr_model=i_file,
    adr_datain=adr_datain,
    adr_oridata=adr_oridata,
    needtauxy=mypara.needtauxy,
)
# print(cut_var_pred.shape)
# print(cut_var_true.shape)
# cut_var_pred = 
np.save(f"./data/noise1/pred_{model_name}_true.npy", cut_var_pred)
np.save(f"./data/noise1/true_{model_name}_true.npy", cut_var_true)
# np.save("./noise/cut_var_true.npy", cut_var_true)

# np.save("./data/cut_nino_pred.npy", cut_nino_pred)
# np.save("./data/cut_nino_true.npy", cut_nino_true)
    # ---------------------------------------------------------
    # cut_var_pred