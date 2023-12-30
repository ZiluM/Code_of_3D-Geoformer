import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import yaml
import putils


yml_path = sys.argv[1]
logger = putils.get_logger()
with open(yml_path, 'r') as stream:
    config = yaml.safe_load(stream)
config['logger'] = logger

config['plot_data'] = {}


sys.path.append(config['model_config'])
from myconfig1 import mypara

config['my_para'] = mypara

logger.info('config: ' + str(config))

config['post_path'] = config['post_path'] + 'post_' + config['job_name'] + '/'
if not os.path.exists(config['post_path']):
    os.makedirs(config['post_path'])

# load real data ============================

logger.info('Start loading real data ===========================================')
real_data = putils.REAL_DATA(config).dataX
stds = config['obs_std']
logger.info('real_data.shape: ' + str(real_data.shape))
logger.info('stds.shape: ' + str(stds.shape))
real_data = real_data * config['obs_std'][:,None,None]
logger.info('End loading real data ===========================================')




# sys.path.append(config['model_config'])
logger.info('Start Postprocessing xp ===========================================')
xp_data = (xr.open_mfdataset(config['save_path'] + config['job_name'] + '/xp*.nc')['xp']).chunk({"time":240}) * stds[:,None,None] 
logger.info('End Postprocessing xp ===========================================')
logger.info('Start Postprocessing xa ===========================================')
xa_data = (xr.open_mfdataset(config['save_path'] + config['job_name'] + '/xa*.nc')['xa']).chunk({"time":240}) * stds[:,None,None]
logger.info('End Postprocessing xa ===========================================')

putils.plot_corr(xa_data,real_data,config)
putils.plot_vertical_corr(xa_data,real_data,config)
putils.plot_MSE(xa_data,real_data,config)
putils.plot_vertical_MSE(xa_data,real_data,config)



logger.info('Start Postprocessing Variance')
putils.cal_var(xa_data,xp_data,config)
logger.info('End Postprocessing Variance')

logger.info('Start Postprocessing Nino34')
xa_nino34 = putils.Nino34_cal(xa_data,config)
xp_nino34 = putils.Nino34_cal(xp_data,config)

Nino34s = [xp_nino34,xa_nino34]

putils.Nino34_plot(Nino34s,config)
logger.info('End Postprocessing Nino34')

logger.info('Start Postprocessing Obs')
putils.Obs_plot(config,xa_data,xp_data)
logger.info('End Postprocessing Obs')


putils.save_plot_data(config)


