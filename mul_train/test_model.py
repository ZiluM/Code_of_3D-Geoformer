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
import logging
import sys

mpl.use("Agg")
plt.rc("font", family="Arial")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

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


def file_name(file_dir):
    logger.info(f"file_dir: {file_dir}")
    L = []
    # files = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
                # files.append(file)
    return L


# --------------------------------------------------------
files= file_name("./model")
files1 = [file.split("/")[-1] for file in files ]
logger.info(f"files: {files}")
print(files)
print(files1)
logger.info(f"files1: {files1}")
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
for i_file in files[: file_num + 1]:
    print("*************" * 8)
    logger.info(f"i_file: {i_file}")
    # print(i_file)
    num += 1
    fig1 = plt.figure(figsize=(5, 2.5), dpi=300)
    ax1 = fig1.add_subplot(1, 2, 1)
    ax2 = fig1.add_subplot(1, 2, 2)
    if files1[num-1][0] == "G":
        mypara.patch_size = (3,4)
    else: 
        p1 = files1[num-1][0]
        p2 = files1[num-1][2]
        mypara.patch_size = (int(p1), int(p2))
    mypara.H0 = int((mypara.lat_range[1] - mypara.lat_range[0]) / mypara.patch_size[0])
    mypara.W0 = int((mypara.lon_range[1] - mypara.lon_range[0]) / mypara.patch_size[1])
    mypara.emb_spatial_size = mypara.H0 * mypara.W0
    logger.info(f"mypara.patch_size: {mypara.patch_size}")
    logger.info("begin to predict")
    
    (cut_var_pred, cut_var_true, cut_nino_pred, cut_nino_true,) = func_pre(
        mypara=mypara,
        adr_model=i_file,
        adr_datain=adr_datain,
        adr_oridata=adr_oridata,
        needtauxy=mypara.needtauxy,
    )
    logger.info(f"cut_nino_pred.shape: {cut_nino_pred.shape}")
    logger.info(f"cut_var_pred.shape: {cut_var_pred.shape}")
    # -----------
    cut_nino_true_jx = deepcopy(cut_nino_true[(24 - lead_max + 1) :])
    cut_nino_pred_jx = deepcopy(cut_nino_pred[:, (24 - lead_max + 1) :])
    assert np.mod(cut_nino_true_jx.shape[0], 12) == 0
    corr = np.zeros([lead_max])
    mse = np.zeros([lead_max])
    mae = np.zeros([lead_max])
    bb = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        aa = runmean(cut_nino_pred_jx[l], 3)
        corr[l] = np.corrcoef(aa, bb)[0, 1]
        mse[l] = mean_squared_error(aa, bb)
        mae[l] = mean_absolute_error(aa, bb)
    del aa, bb
    # -------------figure---------------
    ax1.plot(corr, color="C0", linestyle="-", linewidth=1, label="Corr")
    ax1.plot(mse ** 0.5, color="C2", linestyle="-", linewidth=1, label="RMSE")
    ax1.plot(mae, color="C3", linestyle="-", linewidth=1, label="MAE")
    ax1.plot(np.ones(lead_max) * 0.5, color="k", linestyle="--", linewidth=1)
    ax1.set_xlim(0, lead_max - 1)
    ax1.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax1.xaxis.set_minor_locator(MultipleLocator(1))
    ax1.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=9)
    ax1.set_xlabel("Prediction lead (months)", fontsize=9)

    ax1.set_ylim(0, 1)
    ax1.set_yticks(np.arange(0, 1.01, 0.1))
    ax1.set_yticklabels(np.around(np.arange(0, 1.01, 0.1), 1), fontsize=9)
    ax1.grid(linestyle=":")
    # ---------skill contourf
    # 1983.1~2021.12
    long_eval_yr = 2021 - 1983 + 1
    cut_nino_true_jx = runmean(cut_nino_true_jx, 3)
    for l in range(lead_max):
        cut_nino_pred_jx[l] = runmean(cut_nino_pred_jx[l], 3)  # [lead_max,len]
    pre_nino_tg = np.zeros([long_eval_yr, 12, lead_max])
    for l in range(lead_max):
        for i in range(long_eval_yr):
            pre_nino_tg[i, :, l] = cut_nino_pred_jx[l, 12 * i : 12 * (i + 1)]
    real_nino = np.zeros([long_eval_yr, 12])
    for i in range(long_eval_yr):
        real_nino[i, :] = cut_nino_true_jx[12 * i : 12 * (i + 1)]
    tem1 = deepcopy(pre_nino_tg)
    pre_nino_st = np.zeros(pre_nino_tg.shape)
    for y in range(long_eval_yr):
        for t in range(12):
            terget = t + 1
            for l in range(lead_max):
                lead = l + 1
                start_mon = terget - lead
                if -12 < start_mon <= 0:
                    start_mon += 12
                elif start_mon <= -12:
                    start_mon += 24
                pre_nino_st[y, start_mon - 1, l] = tem1[y, t, l]
    del y, t, l, start_mon, terget, lead, tem1
    tem1 = deepcopy(pre_nino_st)
    tem2 = deepcopy(real_nino)
    nino_skill = cal_ninoskill2(tem1, tem2)
    # ---------------figure
    ax2.contourf(
        nino_skill, levels=np.arange(0, 1.01, 0.1), extend="both", cmap="RdBu_r"
    )
    ct1 = ax2.contour(nino_skill, [0.5, 0.6, 0.7, 0.8, 0.9], colors="k", linewidths=1)
    ax2.clabel(
        ct1,
        fontsize=8,
        colors="k",
        fmt="%.1f",
    )
    ax2.set_xlim(0, lead_max - 1)
    ax2.set_xticks(np.array([1, 5, 10, 15, 20]) - 1)
    ax2.xaxis.set_minor_locator(MultipleLocator(1))
    ax2.set_xticklabels(np.array([1, 5, 10, 15, 20]), fontsize=9)
    ax2.set_xlabel("Prediction lead (months)", fontsize=9)
    ax2.set_yticks(np.arange(0, 12, 1))
    y_ticklabel = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    ax2.set_yticklabels(y_ticklabel, fontsize=9)
    ax2.set_ylabel("Month", fontsize=9)
    del tem1, tem2
    legend = ax1.legend(
        loc="lower left",
        ncol=3,
        fontsize=5,
    )

    _ = ax1.text(x=0.02, y=1.02, s="(a)", fontsize=9)
    _ = ax2.text(x=0.02, y=11.24, s="(b)", fontsize=9)

    plt.tight_layout()
    logger.info(f"save figure: {files1[num-1].split('.')[0]}test_skill.png")
    plt.savefig(f"./model/{files1[num-1].split('.')[0]}test_skill.png")
    plt.show()
    print("*************" * 8)
