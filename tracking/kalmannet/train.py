# Imports
import numpy as np
import sys
from pathlib import Path
import tensorflow as tf
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import yaml 

# Set path
wd = Path(__file__).parent.resolve()
sys.path.append(str(wd))

from Linear_sysmdl import SystemModel
from KalmanNet import KalmanNetNN
from pipeline import Pipeline_KF

#%%
class dotdict(dict): # remove if in utils
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

#%% Train function
# wandb.agent(sweep_id, train, count=15)

def train(config=None):
    # if config.wandb:
    
    # Init
    wandb.init(project="hyperparameter-sweeps")
    
    # Other configs
    dt  = 1e-2
    # Data
    linear_sysmdl_config = {
        "r": 0.05,            # observation noise variance
        "q": 0.1,            # process noise variance
        "v_max": 10,
        "p_rng": [0,100],
        "dt": 1e-2
        }

    linear_sysmdl_config = dotdict(linear_sysmdl_config)

    nonlinear_sysmdl_config = {
        "res": 100,
        "v_max": .75*1/dt,
        "w_max": 0.05*np.pi,
        "noise_var": 0.25,
        "dt": 1e-2
        }

    nonlinear_sysmdl_config = dotdict(nonlinear_sysmdl_config)

    # Configurations to generate data
    data_config = {
        "n_train": 2564,    # number of train samples
        "n_test": 513,      # number of test samples
        "n_val": 513,       # number of validation samples
        "n_t": 100,          # number of timesteps
        "dt": dt,  
        "method": "both",    # 'nonlinear', 'linear' or 'both'
        "linear_config": linear_sysmdl_config,
        "nonlinear_config": nonlinear_sysmdl_config,
        "save": True
        }
    data_config = dotdict(data_config)
    
    # Set default
    configs = {
        "learning_rate": 0.005,
           # "batch_size": 64,
          "N_neur_L1": 8,
          # "N_neur_L2": 16,
          "hidden_dim_factor": 2,    # hidden dimensions
          # "dropout": 0,
          "L2_lambda": 1e-4
        }
    
    config = wandb.config
    
    config.data_config = data_config
    config.n_layers = 1
    config.epochs = 30
    config.dropout = 0.2
    config.N_neur_L2 = 16
    config.batch_size = 64
    # config.learning_rate = 0.005
    # config.L2_lambda = 1e-4
    config.wandb = True
    #L2, bs, lr, L2_lambda
    if wandb.run:
        wandb.config.update({k: v for k, v in configs.items() if k not in dict(wandb.config.items())})
        # configs = dict(wandb.config.items())
    
    # Data Generation
    print('Data loading...')
    tempList = []
    name_list = ['train_input_both', 'train_target_both', 'cv_input_both', 'cv_target_both', 'test_input_both', 'test_target_both']
    for j in range(len(name_list)):
        name=wd / 'data' / (name_list[j]+'.npy')
        tempList.append(tf.convert_to_tensor(np.load(name)))
    [train_input, train_target, val_input, val_target, test_input, test_target] = tempList
    
    # Training       
    print("Start training...")
    strToday = datetime.today().strftime("%m_%d_%y")
    strNow = datetime.now().strftime("%H:%M:%S")
    
    print("Current Time =", strToday," ", strNow)
    sys_model = SystemModel(T = data_config.n_t, dt = data_config.dt)
    model = KalmanNetNN()
    model.Build(config , sys_model) ##wandb.config 
    model_name = './checkpoints/'+'test'
    
    pipeline = Pipeline_KF("KNet_"+ strToday,save_model=True)#, name=('_L1_'+str(N_neur_L1)+'_L2_'+str(N_neur_L2)+ '_HN_'+str(hidden_dim_factor)))
    pipeline.setssModel(sys_model)
    pipeline.setModel(model)
    pipeline.setTrainingParams(config,wd=Path(__file__).parent.resolve())
    pipeline.NNtrain(train_input, train_target, val_input, val_target)

#%% Train
if __name__ == '__main__': 
    # train()
    # Sweep
    sweep_config = yaml.load(open(Path(wd/'sweep_config.yml')), Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep_config,project="hyperparameter-sweeps")
    
    wandb.agent(sweep_id, function=train)
