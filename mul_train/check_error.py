# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


# %%
# model_idx = 0

for model_idx in range(23):
    print(model_idx,'===================='*3)
    path1 = f"./data/noise1/pred_3x4p_1_{model_idx}.npy"
    path2 = f"./data/noise1/noise_3x4p_1_{model_idx}.npy"

    pred = np.load(path1)
    noise = np.load(path2)

    # %%
    orig = pred + noise

    # %%
    # Nino34_slice = None,None,slice(2),slice(15,36),slice(49,75)
    Nino34 = orig[:,:,2,15:36,49:75].mean(axis=(-1,-2))
    Nino34.shape

    # %%
    # orig.shape
    Nino34_pred = pred[:,:,2,15:36,49:75].mean(axis=(-1,-2))
    Nino34.shape

    # %%
    import sacpy as scp
    corr_ls = []
    var_ls = []
    field_var = []
    noise_var = []
    for i in range(Nino34.shape[1]):
        corr = scp.LinReg(Nino34[:,i],Nino34_pred[:,i]).corr
        # print(corr)
        corr_ls.append(corr)
        var_ls.append(Nino34_pred[:,i].var())
        field_var.append(pred[:,i].var())
        noise_var.append(noise[:,i].var())


    # %%
    months = np.arange(1,21)
    # plt.plot(months,corr_ls)
    fig = plt.figure()
    plt.subplot(221)
    plt.plot(months,var_ls,label=model_idx)
    # plt.axhline(0.5,color="red",linestyle="--")
    plt.xlabel("Month")
    # plt.ylabel("Correlation")
    plt.ylabel("Variance")
    plt.title("Nino34 prediction variance, mean={:.3f}".format(np.array(var_ls).mean()))
    plt.subplot(222)
    plt.plot(months,corr_ls,label=model_idx)
    plt.axhline(0.5,color="red",linestyle="--")
    plt.xlabel("Month")
    plt.ylabel("Correlation")
    plt.title("Nino34 prediction correlation")
    plt.subplot(223)
    plt.plot(months,field_var,label=model_idx)
    mean_var = np.array(field_var).mean()
    plt.xlabel("Month")
    plt.ylabel("Variance")
    plt.title("Field variance (mean={:.3f})".format(mean_var))
    plt.subplot(224)
    plt.plot(months,noise_var,label=model_idx)
    plt.xlabel("Month")
    plt.ylabel("Variance")
    plt.title("Noise variance, mean={:.3f}".format(np.array(noise_var).mean()))
    plt.tight_layout()

    plt.legend()
    plt.savefig(f"./data/noise1/pic/3x4p_1_{model_idx}.png")
    print(f"save to ./data/noise1/pic/3x4p_1_{model_idx}.png")
    plt.show()


# %%



