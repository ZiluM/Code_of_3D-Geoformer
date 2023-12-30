import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# noise_path = "/glade/work/zilumeng/3D_trans/data/noise/"
# # for i in range(0,44001,1000):
# name_ls = list(range(1000,44001,1000))
# name_ls.append(44804)
# np.random.seed(1)
# idxs = np.random.random_integers(0,len(name_ls)-1,size=2,)
# print(idxs)
# idxs = [name_ls[idxs[i]] for i in range(2)]
# times = pd.date_range(start="1850-01",end="2014-12",freq="MS")[12:-20]
# months = times.month
# print(months.shape,months)
# months = np.concatenate([months]*23,axis=0)
# print(months.shape,months)
# for idx in idxs:
#     print(idx)
#     name = "noise_cmip6_{}.npy".format(idx)
#     path = noise_path + name
#     noise = np.load(path)
#     slice0 = slice(idx-1000,idx)
#     months1 = months[slice0]
#     # print(months1.shape,months1)
#     label = (months1 == 1)
#     noise1 = noise[label]
#     print(noise1.shape,noise1[:10,0,0])

class load_noise():
    """
    load noise for each month and each lead time
    """
    def __init__(self,config):
        self.config = config
        self.noise_path = config['noise_path']
        self.months = np.arange(1,13)
        name_ls = list(range(1000,44001,1000))
        name_ls.append(44804)
        self.name_ls = name_ls
        times = pd.date_range(start="1850-01",end="2014-12",freq="MS")[12:-20]
        months = times.month
        months = np.concatenate([months]*23,axis=0)
        self.months = months
    
    def load(self, month,max_lead,number):
        """
        month: current month
        load noise for each month and each lead time
        """
        month = month + 1 if month < 12 else 1 # next month for noise save
        noise_ls = []
        name_ls = self.name_ls
        idxs = np.random.randint(0,len(name_ls)-1,size=2,)
        print(idxs)
        idxs = [name_ls[idxs[i]] for i in range(2)]
        number1 = number // 2
        number2 = number - number1
        numbers = [number1,number2]
        noise_ls = []
        for idx,num in zip(idxs,numbers): 
            name = "noise_cmip6_{}.npy".format(idx)
            path = self.noise_path + name
            noise = np.load(path)
            print(noise.shape)
            slice0 = slice(idx-1000,idx)
            months1 = self.months[slice0]
            label = (months1 == month)
            noise1 = noise[label]
            idx_noise = np.arange(0,noise1.shape[0])
            rd_idx_noise = np.random.choice(idx_noise,num,replace=False)
            noise1 = noise1[rd_idx_noise,:max_lead]
            noise_ls.append(noise1)
        noise_ls = np.concatenate(noise_ls,axis=0)
        return noise_ls

if __name__ == '__main__':
    config = {'noise_path':"/glade/work/zilumeng/3D_trans/data/noise/"}
    no = load_noise(config)
    noise = no.load(1,13,100)
    print(noise.shape)




# print(months.shape,months)
# print(months1.shape)
# print("loading...")
# # for i in name_ls:
# #     print(i)
# #     name = "noise_cmip6_{}.npy".format(i)
# #     path = noise_path + name
# #     noise = np.load(path)
# #     noise_ls.append(noise)
# data_idx = np.arange(0,44804)
# data_idx = data_idx.reshape(23,1,1948)
# data_idx = np.repeat(data_idx,20,axis=1)
# print(data_idx.shape)
# print(data_idx)
# months = np.repeat(np.arange(1,13)[np.newaxis,:],noise.shape[0]/12//12,axis=0).flatten()
# months = np.concatenate([])

# for lead_time in range(20):
#     data = noise[:,lead_time]
#     print(data.shape,'data')
#     for mon in range(1,13):
#         label = (months1 == mon)
#         data1 = data[label]
#         print(data1.shape,'data1')
#         np.save("./data/noise1/noise_cmip6_l{}_m{}.npy".format(lead_time,mon),data1)
    