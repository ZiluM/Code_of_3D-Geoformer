import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import sys
sys.path.append("/glade/work/zilumeng/3D_trans/Da/")
from utils import make_dataset_ens
import cartopy.crs as ccrs
import sacpy.Map
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import pickle as pkl






def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def Nino34_cal(xa,config):
    sst = xa.loc[dict(lev=2,lat=slice(-5,5),lon=slice(190,240))].mean(dim=['lat','lon'])
    Nino34 = sst.compute()
    return Nino34

def Nino34_plot(Nino34s,config):
    fig = plt.figure(figsize=(10,5))
    ax = plt.subplot(1,1,1)
    colors = ['lightblue','lightcoral'][::-1]
    colors1 = ['blue','red'][::-1]
    titles = ['Xp','Xa']
    for i in range(2):
        Nino34 = Nino34s[i]
        tm = Nino34['time']
        # ax.plot(tm,Nino34)
        for j in range(config['Nens']):
            ax.plot(tm,Nino34.loc[{'ens':j}],color=colors[i],linewidth=0.7,alpha=0.3)
        ax.plot(tm,Nino34.mean(dim='ens'),color=colors1[i],linewidth=1.3,label=titles[i]+'_mean',zorder= 10)
        # ax.set_title('Nino34_' + titles[i])
    ax.plot(tm,config['real_nino34'],color='black',linewidth=1.3,label='real',zorder= 10)
    corr_xa = np.corrcoef(Nino34s[1].mean(dim='ens').compute(),config['real_nino34'])[0,1]
    corr_xp = np.corrcoef(Nino34s[0].mean(dim='ens').compute(),config['real_nino34'])[0,1]
    save_path = config['post_path'] + 'Nino34.png'
    ax.legend()
    ax.set_title('Nino34, xa_corr: {:.2f}, xp_corr: {:.2f}'.format(corr_xa,corr_xp))
    fig.savefig(save_path,dpi=300)
    fig.show()
    config['plot_data']['Xa_Nino34'] = Nino34s
    config['plot_data']['real_Nino34'] = config['real_nino34']

class REAL_DATA:
    def __init__(self,config):
        # obs_path = config['obs_path']
        mypara = config['my_para']
        needtauxy = mypara.needtauxy
        lon_range = mypara.lon_range
        lat_range = mypara.lat_range
        lev_range = mypara.lev_range
        address = config['true_path']
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = lev_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["temperatureNor"][
            :,
            lev_range[0] : lev_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, None], tauy[:, None], temp), axis=1
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
        start_time = config['start_time']
        times = pd.date_range("1980-01-01", "2021-12-31", freq="MS")
        start_index = np.abs(times - start_time).argmin()
        self.dataX = self.dataX[start_index:config['DA_length']+start_index]
        # data = sd.load()
        data = data_in
        stdtemp = data["stdtemp"][mypara.lev_range[0] : mypara.lev_range[1]].values
        stdtemp = np.nanmean(stdtemp, axis=(1, 2))
        stdtaux = data["stdtaux"].values
        stdtaux = np.nanmean(stdtaux, axis=(0, 1))
        stdtauy = data["stdtauy"].values
        stdtauy = np.nanmean(stdtauy, axis=(0, 1))
        stds = np.concatenate((stdtaux[None], stdtauy[None], stdtemp), axis=0)
        config['obs_std'] = stds
        lon_nino_loc = mypara.lon_nino_relative
        lat_nino_loc = mypara.lat_nino_relative
        Nino34 = self.dataX[:,2,lat_nino_loc[0]:lat_nino_loc[1],lon_nino_loc[0]:lon_nino_loc[1]].mean(axis=(1,2))
        config['real_nino34'] = Nino34 * stds[2]

def specific_loc_indx(lats,lons,lat,lon):
    """
    lats: (Ny,)
    lons: (Nx,)
    lat: float
    lon: float
    return: i: index of lon
            j: index of lat
    """
    lat_abs = np.abs(lats-lat)
    lon_abs = np.abs(lons-lon)
    return np.argmin(lon_abs) , np.argmin(lat_abs)

def Obs_plot(config,xa,xp):
    obs_path = config['save_path'] + config['job_name'] + '/' + 'obs' +  '.npy'
    obs = np.load(obs_path) * config['obs_std'][2]
    obs_locs = config['obs_locs']
    # config['obs_locs'] = obs_locs
    lons = xa['lon'].values
    lats = xa['lat'].values
    for i in range(config['Nobs']):
        config['logger'].info(f'Plotting Obs_{i}')
        fig = plt.figure(figsize=(7,4))
        ax = plt.subplot(1,1,1)
        ilon,ilat = specific_loc_indx(lats,lons,obs_locs[i][1],obs_locs[i][0])

        xa_obs = xa.sel(lev=2).isel(lat=ilat,lon=ilon).chunk({'time':-1}).compute()
        xp_obs = xp.sel(lev=2).isel(lat=ilat,lon=ilon).chunk({'time':-1}).compute()
        config['logger'].info(f'Obs_{i} location: lon: {lons[ilon]}, lat: {lats[ilat]}')
        config['logger'].info(f'xa_obs.shape: {xa_obs.shape}, xp_obs.shape: {xp_obs.shape}')
        for j in range(config['Nens']):
            config['logger'].info(f'Plotting Obs_{i} ensemble {j}')
            ax.plot(xa_obs['time'],xa_obs.loc[{'ens':j}],color='lightcoral',alpha=0.5)
            ax.plot(xp_obs['time'],xp_obs.loc[{'ens':j}],color='lightblue',alpha=0.5)
        ax.plot(xa_obs['time'],xa_obs.mean(dim='ens'),color='red',label='xa',zorder=10)
        ax.plot(xp_obs['time'],xp_obs.mean(dim='ens'),color='blue',label='xp',zorder=10)
        mean_t = config['obs_mean_length']
        obs_time = xa_obs['time'][int(mean_t/2)::mean_t]
        ax.plot(obs_time,obs.T[i],color='black',label='obs',zorder=10)
        ax.legend()
        ax.set_title(f'Obs_{i}, lon: {lons[ilon]}, lat: {lats[ilat]}')
        save_path = config['post_path'] + f'Obs_{i}.png'
        fig.savefig(save_path,dpi=300)
        fig.show()


def cal_var(xa,xp,config):
    xa_var = xa.var(dim='ens').compute()
    xp_var = xp.var(dim='ens').compute()
    xp_var_time = xp_var.mean(['lon', 'lat','lev']).compute()
    xa_var_time = xa_var.mean(['lon', 'lat','lev']).compute()
    fig = plt.figure(figsize=(7,4))
    ax = plt.subplot(1,1,1)
    ax.plot(xa_var_time['time'],xa_var_time,label='xa_var')
    ax.plot(xp_var_time['time'],xp_var_time,label='xp_var')
    ax.legend()
    ax.set_title('xa_var_time')
    plt.savefig(config['post_path'] + 'xa&p_var_time.png',dpi=300)
    plt.show()


def field_corr(field1, field2):
    """
    field1: time, nspace
    field2: time, nspace
    """
    field1a = field1 - field1.mean(axis=0)
    field2a = field2 - field2.mean(axis=0)
    covar = np.einsum("ij...,ij...->j...", field1a, field2a) / (field1a.shape[0] - 1)  # covar:nspace
    corr = covar / np.std(field1a, axis=0) / np.std(field2a, axis=0)  # corr: nspace
    return corr.real


def init_map(fig,idx):
    ax = fig.add_subplot(3,3,idx,projection=ccrs.PlateCarree(central_longitude=180))
    # ax.coastlines()
    # ax.set_aspect("auto")
    return ax

def plot_corr(xa,real_data,config):
    xam = xa.mean(dim='ens').compute().to_numpy()
    corrs = field_corr(xam,real_data)
    lon = xa['lon'].values
    lat = xa['lat'].values
    obs_locs = np.array(config['obs_locs']).T # 2, Nobs
    levs = 9
    fig = plt.figure(figsize=(15, 10))
    # fig.sub
    fig.suptitle(f'Correlation', fontsize=12)
    titles = [
    'taux','tauy'
    ]
    for i in [5.,  20.,  40.,  60.,  90., 120., 150.]:
        titles.append("temperature level:"+str(i)+'m')
    for j in range(9):
        # ax = fig.add_subplot(3,3, j + 1)
        ax = init_map(fig,j+1)
        m = ax.scontourf(lon,lat,corrs[j],levels=np.arange(-0.1,1+0.01,0.1),cmap="RdYlBu_r", transform=ccrs.PlateCarree(),extend='min')
        if j == 2:
            ax.scatter(obs_locs[0],obs_locs[1],color='black',marker='*',s=50,zorder=11,label='obs',transform=ccrs.PlateCarree())
            ax.legend()
        ax.init_map(stepx=60,stepy=5)
        # ax.sig_plot(lon,lat,1 - corrs1[i, j],thrshd=0.5)
        ax.set_title(titles[j] + "  mean:{:.2f}".format(np.nanmean(corrs[j])),fontsize=9)
    ax_bar = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(m, cax=ax_bar)
    fig.savefig(config['post_path'] + 'Correlation.png',dpi=300)
    config['plot_data']['Correlation'] = corrs

def plot_MSE(xa,real,config):
    xam = xa.mean(dim='ens').compute().to_numpy()
    mse = np.nanmean((xam - real)**2,axis=0)
    lon = xa['lon'].values
    lat = xa['lat'].values
    levs = 9
    fig = plt.figure(figsize=(15, 10))
    # fig.sub
    fig.suptitle(f'MSE', fontsize=12)
    titles = [
    'taux','tauy'
    ]
    obs_locs = np.array(config['obs_locs']).T # 2, Nobs
    for i in [5.,  20.,  40.,  60.,  90., 120., 150.]:
        titles.append("temperature level:"+str(i)+'m')
    for j in range(9):
        # ax = fig.add_subplot(3,3, j + 1)
        ax = init_map(fig,j+1)
        m = ax.scontourf(lon,lat,mse[j],cmap="Reds", transform=ccrs.PlateCarree(),extend='max')
        if j == 2:
            ax.scatter(obs_locs[0],obs_locs[1],color='black',marker='*',s=50,zorder=11,label='obs',transform=ccrs.PlateCarree())
            ax.legend()
        ax.init_map(stepx=60,stepy=5)
        # ax.sig_plot(lon,lat,1 - corrs1[i, j],thrshd=0.5)
        ax.set_title(titles[j] + "  mean:{:.2f}".format(np.nanmean(mse[j])),fontsize=9)
    ax_bar = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(m, cax=ax_bar)
    fig.savefig(config['post_path'] + 'MSE.png',dpi=300)
    config['plot_data']['MSE'] = mse




def plot_vertical_corr(xa,real, config):
    xam = xa.mean(dim='ens').compute()
    real = xr.DataArray(real,coords=xam.coords)
    xam_eq = xam.sel(lev=slice(2,None),lat=slice(-5,5)).mean(dim='lat')
    real_eq = real.sel(lev=slice(2,None),lat=slice(-5,5)).mean(dim='lat')
    corrs = field_corr(xam_eq,real_eq)
    lon = xa['lon'].values
    levs = np.array([  5.,  20.,  40.,  60.,  90., 120., 150.])
    height = levs
    fig = plt.figure(figsize=(8, 3.5))

    ax = fig.add_subplot(111)
    m = ax.contourf(lon,height,corrs, levels=np.arange(-0.1, 1+0.01, 0.05), cmap="RdBu_r",extend='min')
    ax.invert_yaxis()
    # ax.set_yscale("exp")
    ax.set_yticks([5, 20, 40, 60, 90, 120, 150])
    ax.set_yticklabels([5, 20, 40, 60, 90, 120, 150], fontsize=9)
    # ax.set_title("lead = " + str(i + 1) + ' month', fontsize=9)
    ax.set_title("Equator Correlation", fontsize=9)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.set_ylabel("Height(m)", fontsize=9)
    ax.set_xlabel("Longitude", fontsize=9)
    plt.colorbar(m)
    plt.savefig(config['post_path'] + 'Equator_Correlation.png',dpi=300)
    config['plot_data']['Equator_Correlation'] = corrs

def plot_vertical_MSE(xa,real, config):
    xam = xa.mean(dim='ens').compute()
    real = xr.DataArray(real,coords=xam.coords)
    xam_eq = xam.sel(lev=slice(2,None),lat=slice(-5,5)).mean(dim='lat')
    real_eq = real.sel(lev=slice(2,None),lat=slice(-5,5)).mean(dim='lat')
    # corrs = field_corr(xam_eq,real_eq)
    mse = np.nanmean((xam_eq - real_eq)**2,axis=0)
    lon = xa['lon'].values
    levs = np.array([  5.,  20.,  40.,  60.,  90., 120., 150.])
    height = levs
    fig = plt.figure(figsize=(8, 3.5))

    ax = fig.add_subplot(111)
    m = ax.contourf(lon,height,mse, cmap="Reds",extend='max')
    ax.invert_yaxis()
    # ax.set_yscale("exp")
    ax.set_yticks([5, 20, 40, 60, 90, 120, 150])
    ax.set_yticklabels([5, 20, 40, 60, 90, 120, 150], fontsize=9)
    # ax.set_title("lead = " + str(i + 1) + ' month', fontsize=9)
    ax.set_title("Equator MSE", fontsize=9)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.set_ylabel("Height(m)", fontsize=9)
    ax.set_xlabel("Longitude", fontsize=9)
    plt.colorbar(m)
    plt.savefig(config['post_path'] + 'Equator_MSE.png',dpi=300)
    config['plot_data']['Equator_MSE'] = mse



def save_plot_data(config):
    plot_data = config['plot_data']
    save_path = config['post_path'] + 'plot_data.npy'
    np.save(save_path,plot_data)
    config['logger'].info(f'plot data saved to {save_path}')



