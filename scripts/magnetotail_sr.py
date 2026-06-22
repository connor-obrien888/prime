import pysr
import sympy
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from pysr import PySRRegressor, TensorBoardLoggerSpec
from sklearn.preprocessing import RobustScaler
import omegaconf
import argparse

def main(config, runname):
    cfg = omegaconf.OmegaConf.load(
        config
    )

    dataset, targets, in_tf, in_train, in_val, in_test, tar_train, tar_val, tar_tf, tar_test = load_dataset(
        datapath = cfg.data.datapath,
        in_store = cfg.data.in_store,
        tar_store = cfg.data.tar_store,
        in_storekey = cfg.data.in_storekey,
        sample_rate = cfg.data.sample_rate, #Sampling rate of input data in seconds
        window = cfg.data.window,
        stride = cfg.data.stride,
        train_frac = cfg.data.train_frac, #These percentages control the dataset split
        val_frac = cfg.data.val_frac, #The full dataset is split train_frac, val_frac, test_frac in sequence
        test_frac = cfg.data.test_frac, #TO DO: randomly assign data in chunks
        tar_scaler = RobustScaler(), #The type of scaler to be used on target data
        in_scaler = RobustScaler(), #The type of scaling to be used on input data
        in_keys = cfg.data.in_keys,
        tar_keys = cfg.data.tar_keys,
        train_start = pd.to_datetime(cfg.data.data_limits[0]),
        train_stop = pd.to_datetime(cfg.data.data_limits[1]),
        mode = cfg.data.tar_sc,
    )
    X = in_train
    y = tar_train

    logger_spec = TensorBoardLoggerSpec(
        log_dir=os.path.join(cfg.opt.log_dir, runname),
        log_interval=cfg.opt.log_interval,
    )

    model = PySRRegressor(
        niterations=cfg.opt.niterations,
        populations=cfg.model.populations,  # Use more populations
        population_size=cfg.model.population_size,
        binary_operators=cfg.model.binary_operators,
        unary_operators=cfg.model.unary_operators,
        logger_spec=logger_spec,
        cluster_manager=cfg.opt.cluster_manager,
        parallelism=cfg.opt.parallelism,
        turbo=cfg.opt.turbo,
        maxsize=cfg.model.maxsize,
        maxdepth=cfg.model.maxdepth,
    )

    model.fit(X, y)

    best_idx = model.equations_.query(
        f"loss < {2 * model.equations_.loss.min()}"
    ).score.idxmax()
    print(model.sympy(best_idx))

class dataset():
    def __init__(self, tar_full, in_full, tar_scaler, in_scaler, loc_scaler, tar_train, in_train, tar_test, in_test, inds_train, inds_test, ds_mask, in_keys, tar_keys):
        self.tar_full = tar_full
        self.in_full = in_full
        self.tar_scaler = tar_scaler
        self.in_scaler = in_scaler
        self.loc_scaler = loc_scaler
        self.tar_train = tar_train
        self.in_train = in_train
        self.tar_test = tar_test
        self.in_test = in_test
        self.inds_train = inds_train
        self.inds_test = inds_test
        self.ds_mask = ds_mask
        self.in_keys = in_keys
        self.tar_keys = tar_keys

    def get_train(self):
        return self.in_train, self.tar_train

    def get_test(self):
        return self.in_test, self.tar_test

    def get_full(self):
        return self.in_full, self.tar_full

    def get_inds(self):
        return self.inds_train, self.inds_test

    def get_mask(self):
        return self.ds_mask

    def get_data(self):
        in_data = self.in_scaler.inverse_transform(self.in_full)
        tar_data = self.tar_scaler.inverse_transform(self.tar_full)
        return in_data, tar_data

def load_dataset(datapath = './data/',
                 in_store = 'wind_data_full.h5',
                 tar_store = 'geotail.csv',
                 in_storekey = 'wind_full',
                 sample_rate = 100, #Sampling rate of input data in seconds
                 window = 100,
                 stride = 0,
                 train_frac = 0.6, #These percentages control the dataset split
                 val_frac = 0.2, #The full dataset is split train_frac, val_frac, test_frac in sequence
                 test_frac = 0.2, #TO DO: randomly assign data in chunks
                 tar_scaler = RobustScaler(), #The type of scaler to be used on target data
                 in_scaler = RobustScaler(), #The type of scaling to be used on input data
                 in_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'SME', 'SMR', 'R_xgse', 'R_ygse', 'R_zgse', 'Rx_int', 'Ry_int', 'Rz_int', 'tilt_int', 'SYM_H'], #Data keys to include in input dataset
                 tar_keys = ['N','temp','Bx','By','Bz'], #Targets from Geotail dataset to match
                 train_start = pd.to_datetime('2005-01-01 00:00:00+0000'),
                 train_stop = pd.to_datetime('2010-05-01 00:00:00+0000'),
                 mode = 'mms'):
    '''
    Helper function that loads a dataset from a file.
    '''
    in_raw = pd.read_hdf(datapath + in_store, key = in_storekey, mode = 'r') #Prepare input data
    # in_raw['Epoch'] = pd.to_datetime(in_raw['Epoch'], utc = True)

    if (mode == 'mms'):
        tar_raw = pd.read_csv(datapath + tar_store) #Prepare target data
        tar_raw['datetime'] = pd.to_datetime(tar_raw['time'], utc = True) #REMOVE format = 'mixed' WHEN PORTING TO SCC
    else:
        tar_raw = pd.read_csv(datapath + tar_store, index_col=0) #Prepare target data
        tar_raw['datetime'] = pd.to_datetime(tar_raw['datetime'], utc = True) #REMOVE format = 'mixed' WHEN PORTING TO SCC
    tar_raw = tar_raw.sort_values(by=['datetime'])

    dataset = in_raw #Start with the input data from Wind, then add in geotail data binned to Wind data cadence
    dataset = dataset[(dataset['Epoch'] > (train_start-pd.Timedelta(str((window+stride)*sample_rate)+'s'))) & (dataset['Epoch'] < (train_stop+pd.Timedelta(str((window+stride)*sample_rate)+'s')))] #Cut dataset just to training interval (for speed)
    if sample_rate != 100: #If a cadence other than the Wind data cadence is specified
        dataset = dataset.resample(str(sample_rate)+'s', on='Epoch').mean()
    # print(dataset)
    # dataset['Epoch'] = dataset.index
    # dataset['Epoch'] = pd.to_datetime(dataset['Epoch'], utc = True)
    # dataset = dataset.reset_index(drop=True)
    tar_slice = tar_raw[(tar_raw['datetime'] > train_start)&(tar_raw['datetime'] < train_stop)]
    tar_slice = tar_slice.reset_index(drop=True)

    #Assign entries in tar_slice to bins defined by the Wind epochs
    tar_slice.loc[:, 'bindex'] = pd.cut(tar_slice['datetime'], dataset['Epoch'], right = False)
    tar_group = tar_slice.groupby('bindex', as_index=False).agg('mean') #Average in each bin
    for key in tar_group.columns:
        if (key=='datetime') | (key=='bindex'):
            continue #Do not include 'datetime' or 'bindex' in dataset
        else:
            dataset.loc[:, key] = tar_group.loc[:, key] #Assign the binned geotail data to each timestamp in dataset
    targets = dataset.loc[:, tar_keys].dropna() #Targets only exist for parts of the dataset, thus dropna() is used to isolate the existing data
    tar_arr = targets.to_numpy() #Get the array of target features before other descriptive features are added
    in_arr = np.zeros((len(targets), window, len(in_keys)))
    for i, idx in enumerate(targets.index): #i is numpy index, idx is pandas index
        in_arr[i, :, :] = dataset.loc[(idx-window-stride):(idx-stride-1), in_keys]
        targets.loc[idx, 'input_start'] = dataset.loc[(idx-window-stride), 'Epoch']
        targets.loc[idx, 'input_stop'] = dataset.loc[(idx-stride-1), 'Epoch']
        #THIS SECTION HANDLES POSITION AND DIPOLE TILT. IF ENCODER IS USED THIS SHOULD BE CHANGED
        if (mode == 'geotail'):
            in_arr[i, :, -5:] = dataset.loc[idx, ['Rx_int', 'Ry_int', 'Rz_int', 'tilt_int', 'SYM_H']]
        if (mode == 'mms'):
            in_arr[i, :, -3:] = dataset.loc[idx, ['r_gsm_x','r_gsm_y','r_gsm_z']]
        else:
            in_arr[i, :, -3:] = dataset.loc[idx, ['r_gsm_x','r_gsm_y','r_gsm_z']] 
    #Assign data to datasets
    #Train
    tar_train = tar_arr[:int(train_frac*len(tar_arr))]
    in_train = in_arr[:int(train_frac*len(in_arr))]
    targets.loc[targets.index[:int(train_frac*len(targets))], 'dataset_flag'] = 0 # 0 is train, 1 is validation, 2 is test

    #Validation
    tar_val = tar_arr[int(train_frac*len(tar_arr)):int((train_frac+val_frac)*len(tar_arr))]
    in_val = in_arr[int(train_frac*len(tar_arr)):int((train_frac+val_frac)*len(tar_arr))]
    targets.loc[targets.index[int(train_frac*len(tar_arr)):int((train_frac+val_frac)*len(tar_arr))], 'dataset_flag'] = 1 # 0 is train, 1 is validation, 2 is test

    #Test
    tar_test = tar_arr[int((train_frac+val_frac)*len(tar_arr)):int((train_frac+val_frac+test_frac)*len(tar_arr))]
    in_test = in_arr[int((train_frac+val_frac)*len(tar_arr)):int((train_frac+val_frac+test_frac)*len(tar_arr))]
    targets.loc[targets.index[int((train_frac+val_frac)*len(tar_arr)):int((train_frac+val_frac+test_frac)*len(tar_arr))], 'dataset_flag'] = 2 # 0 is train, 1 is validation, 2 is test

    #Rescale the data
    #Rescale target data
    tar_tf = tar_scaler.fit(dataset.loc[targets[targets['dataset_flag']==0].index, tar_keys].to_numpy())
    tar_train = tar_tf.transform(tar_train)
    tar_val = tar_tf.transform(tar_val)
    tar_test = tar_tf.transform(tar_test)

    #Rescale input data
    in_tf = in_scaler.fit(dataset.loc[targets[targets['dataset_flag']==0].index, in_keys].to_numpy())
    in_train = np.asarray([in_tf.transform(frame) for frame in in_train]) #This syntax applies the transformer in 
    in_val = np.asarray([in_tf.transform(frame) for frame in in_val])
    in_test = np.asarray([in_tf.transform(frame) for frame in in_test])
    in_arr = np.asarray([in_tf.transform(frame) for frame in in_arr])
    return dataset, targets, in_tf, in_train, in_val, in_test, tar_train, tar_val, tar_tf, tar_test

def closest_argmin(A, B):
    '''
    Helper function that returns indices of elements in array B closest to each element in array A.
    '''
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

def input_window(input_data, inds, in_keys, window, stride, flag = 'percent'):
    '''
    Helper function that splits time series input data into windows

    Parameters
    ----------
    input_data : float, array-like
        Dataframe of rescaled input data.
    inds : float, array-like
        Array of indices in input_data corresponding to the start times of each target.
    in_keys : list
        List of keys for input data.
    window : int
        Window size in 100s entries.
    stride : int
        Stride between end of window and time of prediction in 100s entries.
    flag : str, optional
        What type of interpolation flag to return with the windowed array. 'percent'
        yields percent of data in window is interpolated. 'tdelt' is longest stretch
        of interpolated data in input window (in indices). Default 'percent'.

    Returns
    -------
    in_arr : float, array-like
        Input array of windows of input_data.
    inter_flags : float, array-like
        Array of percentages of each window that are interpolated data.

    '''
    in_arr = np.zeros(( len(inds) , window, len(in_keys) )) #input array staging, Keras dataset order [samples, time window, features]
    inter_flags = np.zeros(len(inds)) #Get a place to put fraction of input data that is interpolated
    for i in np.arange(len(inds)):
        in_arr[i, :, :] =  input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), in_keys] #Get the timeseries that is 'window' long, 'stride' away from the MMS target time
        if (flag == 'percent'): #Percent of input window that is interpolated
            inter_flags[i] = np.sum(input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), 'flag'])/window #calculate percentage of data that is interpolated
        if (flag == 'tdelt'): #Longest number of consecutive interpolated datapoints in window
            tdelt_max = 0
            tdelt = 0
            for n in input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), 'flag']:
                if (n == 1):
                    tdelt += 1
                    if (tdelt > tdelt_max):
                        tdelt_max = tdelt
                else:
                    tdelt = 0
            inter_flags[i] = tdelt_max     
    return in_arr, inter_flags

def ds_constructor(target_data, input_data, inds, in_keys, tar_keys, window = 140, stride = 1, inter_thresh = 0.5, tar_scaler = RobustScaler(), in_scaler = RobustScaler(), loc_scaler = RobustScaler(), return_mask = False, flag = 'percent'):
    '''
    Helper function that constructs keras datasets from target and input Dataframes.

    Parameters
    ----------
    target_data : float, array-like
        Dataframe of unscaled target data.
    input_data : float, array-like
        Dataframe of rescaled input data.
    inds : float, array-like
        Array of indices in input_data corresponding to the start times of each target.
    in_keys : list
        List of input keys to use in input_data.
    tar_keys : list
        List of target keys to use in target_data.
    window : int, optional
        Window size in 100s entries. Default 140
    stride : int, optional
        Stride between end of window and time of prediction in 100s entries. Default 1
    inter_thresh : float, optional
        Fraction of interpolated data that is acceptable to include in input window. 
        Default 0.5
    tar_scaler : Scaler, optional
        Instance of a Scaler class from module sklearn.preprocessing._data.
        Default skl.preprocessing.RobustScaler().
    in_scaler : Scaler, optional
        Instance of a Scaler class from module sklearn.preprocessing._data.
        Default skl.preprocessing.RobustScaler().
    return_mask : bool, optional
        Do we want to return a mask of what data is below inter_thresh? Useful for reconstructing datasets.
        Default False.

    Returns
    -------
    tar_ds : float, array-like
        Target dataset rescaled target_data.
    in_ds : float, array-like
        Input dataset of rescaled windows of input_data.
    tar_tf : Scaler
        Scaler to invert scaled target data.
    in_tf : Scaler
        Scaler to invert scaled input data.
    '''
    #Copy the DataFrames for safety
    target_data_cp = target_data.copy()
    input_data_cp = input_data.copy()

    #Append 'Rx_int', 'Ry_int', 'Rz_int', 'tilt_int' to the target keys for scaling, in order to add them to in_arr
    loc_df = target_data_cp.loc[:, ['Rx_int', 'Ry_int', 'Rz_int', 'tilt_int']] #Get the location and dipole tilt data
    #Add dummy location and tilt data to the input data, it is overwritten with separate location data later
    input_data_cp['Rx_int'] = np.zeros(len(input_data_cp))
    input_data_cp['Ry_int'] = np.zeros(len(input_data_cp))
    input_data_cp['Rz_int'] = np.zeros(len(input_data_cp))
    input_data_cp['tilt_int'] = np.zeros(len(input_data_cp))
    
    #Rescale here
    tar_tf = tar_scaler.fit(target_data_cp.loc[:,tar_keys].to_numpy()) #This rescales data to inter-quartile range to reduce outlier sensitivity
    in_tf = in_scaler.fit(input_data_cp.loc[:,in_keys].to_numpy())
    loc_tf = loc_scaler.fit(loc_df.to_numpy())
    target_data_cp.loc[:,tar_keys] = tar_tf.transform(target_data_cp.loc[:,tar_keys].to_numpy()) #Look, its pretty annoying that this is the syntax for this operation, but here we are
    input_data_cp.loc[:,in_keys] = in_tf.transform(input_data_cp.loc[:,in_keys].to_numpy())
    loc_df = loc_tf.transform(loc_df.to_numpy())
    
    #Target array
    tar_arr = target_data_cp.loc[:, tar_keys].to_numpy() #Target array is a subset of the MMS data (just parameters)
    
    #Input Array
    in_arr, inter_flags = input_window(input_data_cp, inds, in_keys, window, stride, flag = flag)
    in_arr[:,:,11:15] = np.reshape(np.repeat(loc_df, window, axis = 0), (len(in_arr), window, 4)) #Add the location and dipole tilt data to the input array (overwriting the dummy data)
    
    mask = (inter_flags < inter_thresh) #Mask out data thats above the inter-thresh
    tar_ds = tar_arr[mask, :] #Only keep points that have a good fraction of real, non-interpolated data
    in_ds = in_arr[mask, :, :]
    
    if return_mask:
        return tar_ds, in_ds, tar_tf, in_tf, loc_tf, mask
    else:
        return tar_ds, in_ds, tar_tf, in_tf, loc_tf

def chunker(A, n, f, return_inds = False):
    '''
    Helper function that splits array into chunks and assigns them to two datasets.

    Parameters
    ----------
    A : float, array-like
        Array to be split along axis 0.
    n : int
        Length of each chunk to be split
    f : float
        Fraction of data to end up in the smaller (test) array
    return_inds : bool, optional
        Do we want to return the locations of the train data? Useful for reconstructing datasets.
        Default False.
    Returns
    -------
    tar_ds : float, array-like
        Target dataset rescaled target_data.
    in_ds : float, array-like
        Input dataset of rescaled windows of input_data.
    '''
    k = int(1/f)
    A_tmp = np.copy(np.array_split(A, len(A)//(n-1), axis=0))
    A_test = np.concatenate(A_tmp[::k])
    index = np.ones(len(A_tmp), dtype = bool)
    index[::k] = False
    A_train = np.concatenate(A_tmp[index])
    if return_inds:
        return A_train, A_test, index
    else:
        return A_train, A_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Single symbolic regression run.")
    parser.add_argument(
        "--config",
        type=str,
        default="/glade/u/home/cobrien/prime/prime_lib/configs/mt_sr_config.yaml",
        help="Path to config file defining training run.",
    )
    parser.add_argument(
        "--runname",
        type=str,
        default="srtest",
        help="Name of this model run.",
    )
    args = parser.parse_args()
    main(args.config, args.runname)