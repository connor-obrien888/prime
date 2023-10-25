import numpy as np
import os
import tensorflow as tf
from tensorflow import keras as ks
import keras_tuner as kt
from keras.utils.generic_utils import get_custom_objects
import ks_custom as ksc
import matplotlib.pyplot as plt
import pandas as pd

datapath = '../data/' #Point at data folder
os.environ['SPEDAS_DATA_DIR'] = datapath #Point SPEDAS at this folder

#Fix Random Seed
seed = 888 #Lucky!
tf.random.set_seed(seed) #Seed the TF generator
np.random.seed(seed) #Seed the NP generator

#Register the swish function in a layer object
get_custom_objects().update({'swish': ks.layers.Activation(ks.activations.swish)})

#Assign the job to whatever GPU we were given
def get_n_cores():
    """The NSLOTS variable, If NSLOTS is not defined throw an exception."""
    nslots = os.getenv("NSLOTS")
    if nslots is not None:
        return int(nslots)
    raise ValueError("Environment variable NSLOTS is not defined.")

#More SCC specific boilerplate to ensure we're using as many GPUs as we're given in the optimal mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

NUM_GPUS = len(tf.config.experimental.list_physical_devices("GPU"))
print("Num GPUs Available: ", NUM_GPUS)
if NUM_GPUS > 0:
    print(os.getenv("CUDA_VISIBLE_DEVICES"))

tf.config.set_soft_device_placement(True)
tf.keras.backend.set_floatx("float32")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(get_n_cores())

#Split up and scale the datasets
in_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
tar_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne'] #Targets from MMS dataset to match
window = 60 #Input window length (100s units)
stride = 15 #Stride away from target (100s units)
inter_frac = 5/60 #How many interpolated points in a row to tolerate
flag = 'tdelt' #Whether to cut inputs as a percent of total or length of maximum interpolated stretch of data
ds = ksc.load_dataset(datapath + 'mms_data.h5', datapath + 'wind_data.h5', 'sw', in_keys, tar_keys, split_frac=0.2, window=window, stride=stride, inter_frac=int(inter_frac*window), flag=flag, tar_storekey = 'old_targets', conesize = np.pi/2, vx_cut=False)

#Quick model
name = 'prime_sw_005' #Name of the training run to be saved (if train) or loaded (if not train)
model = ks.Sequential([ks.layers.GRU(units=352),
                       ks.layers.Dense(units=192, activation='elu'),
                       ks.layers.Dense(units=48, activation='elu'),
                       ks.layers.Dense(units=48, activation='elu'),
                       ks.layers.LayerNormalization(),
                       ks.layers.Dropout(0.2),
                       ks.layers.Dense(ds.tar_full.shape[1]*2,activation='linear')
                      ])
model.compile(optimizer=tf.optimizers.Adamax(learning_rate=1e-4), loss=ksc.crps_loss, metrics = ksc.mse_metric)

#Train the model
history = model.fit(ds.in_train, ds.tar_train, epochs=20, validation_split = 0.25)
hist = pd.DataFrame(history.history) #Get the training and validation losses
plt.plot(hist['loss'],color='r') #Training loss
plt.plot(hist['val_loss'],color='b') #Validation loss
plt.plot(hist['mse_metric'],color='r',linestyle='--') #Training loss
plt.plot(hist['val_mse_metric'],color='b',linestyle='--') #Validation loss
plt.ylabel('Loss')
plt.xlabel('Epoch (Training iteration)')
plt.legend(['Training','Validation', 'Training MSE', 'Validation MSE'])
#Add text with minimum val_loss and val_mse_metric to the center right of the plot
plt.text(0.5, 0.5, 'Min Val Loss: ' + str(np.round(np.min(hist['val_loss']),4)) + '\nMin Val MSE: ' + str(np.round(np.min(hist['val_mse_metric']),4)), horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.savefig('opt_model_loss.pdf', bbox_inches='tight')

#Save the model
model.save(name + '.h5')

#Get model outputs for the test dataset
mms_data = pd.read_hdf(datapath + 'mms_data.h5', key = 'old_targets', mode = 'a')
mms_data = ksc.nightside_cut(mms_data, conesize = np.pi/2) #Cut out the nightside data since classification is unreliable there
mms_data.loc[:, ['R_xgse', 'R_ygse', 'R_zgse']] /= 6378 #Scale to Earth Radii
mms_sw = mms_data.drop(mms_data[mms_data['region']!=2].index) #Solar Wind Data
mms_cut = mms_sw[ds.ds_mask]
mms_train = mms_cut.iloc[ds.inds_train]
mms_test = mms_cut.iloc[ds.inds_test]
predict_test_arr = model.predict(ds.in_test)
predict_test = pd.DataFrame(columns = ['Epoch', 'B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ne', 'Ne_sig', 'interp_frac'])
predict_test['Epoch'] = mms_test['Epoch']
predict_test['interp_frac'] = -1 #find a good way to grab the interpolated fration at this step
predict_test[['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne']] = ds.tar_scaler.inverse_transform(predict_test_arr[:, ::2])
predict_test[['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig', 'Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig', 'Ne_sig']] = np.abs(ds.tar_scaler.inverse_transform(predict_test_arr[:, ::2] + predict_test_arr[:, 1::2]) - ds.tar_scaler.inverse_transform(predict_test_arr[:, ::2]))
predict_test.to_hdf(datapath + 'outputs.h5', key = 'prime_predict_test')

#Get predictions for SW dataset
wind_data = pd.read_hdf(datapath + 'wind_data.h5', key = 'inputs') #Load the Wind data
wind_sw_ind = ksc.closest_argmin(mms_sw['Epoch'].to_numpy(), wind_data['Epoch'].to_numpy()) #Get the indices of the input dataset closest to each target time in the solar wind
predict_arr = model.predict(ds.in_full)
predict = pd.DataFrame(columns = ['Epoch', 'B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ne', 'Ne_sig', 'interp_frac'])
predict['Epoch'] = wind_data['Epoch'].iloc[wind_sw_ind][ds.ds_mask]
predict['interp_frac'] = -1 #find a good way to grab the interpolated fration at this step
predict[['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne']] = ds.tar_scaler.inverse_transform(predict_arr[:, ::2])
predict[['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig', 'Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig', 'Ne_sig']] = np.abs(ds.tar_scaler.inverse_transform(predict_arr[:, ::2] + predict_arr[:, 1::2]) - ds.tar_scaler.inverse_transform(predict_arr[:, ::2]))
predict.to_hdf(datapath + 'outputs.h5', key = 'prime_predict')

#Get predictions for full wind dataset
wind_data_rs = pd.read_hdf(datapath + 'wind_data.h5', key = 'wind_combined') #Load the Wind data
wind_data_rs.index = np.arange(len(wind_data_rs)) #make the indes linear
wind_data_rs.loc[:,in_keys] = ds.in_scaler.transform(wind_data.loc[:,in_keys]) #Rescale the data
inds = np.arange(window+stride, len(wind_data_rs)) #Indices to start each time series at
wind_data_rs_arr, flags = ksc.input_window(wind_data_rs, inds, in_keys, window, stride) #Break the array into input timeseries
predict_full_arr = model.predict(wind_data_rs_arr)
predict_full = pd.DataFrame(columns = ['Epoch', 'B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ne', 'Ne_sig', 'interp_frac'])
predict_full['Epoch'] = wind_data_rs['Epoch'][(window+stride):]
predict_full['interp_frac'] = flags
predict_full[['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne']] = ds.tar_scaler.inverse_transform(predict_full_arr[:, ::2])
predict_full[['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig', 'Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig', 'Ne_sig']] = np.abs(ds.tar_scaler.inverse_transform(predict_full_arr[:, ::2] + predict_full_arr[:, 1::2]) - ds.tar_scaler.inverse_transform(predict_full_arr[:, ::2]))
predict_full.to_hdf(datapath + 'outputs.h5', key = 'prime_predict_full')

#Create predictions at the bow shock for the time interval considered in this study
wind_data_rs = pd.read_hdf(datapath + 'wind_data.h5', key = 'wind_combined') #Load the Wind data
wind_data_rs['target_Px_gse'] = 6378*13.25 #Point the target at a fixed point (average BS nose location)
wind_data_rs['target_Py_gse'] = 0
wind_data_rs['target_Pz_gse'] = 0
wind_data_rs.index = np.arange(len(wind_data_rs)) #make the indes linear
wind_data_rs.loc[:,in_keys] = ds.in_scaler.transform(wind_data.loc[:,in_keys]) #Rescale the data
inds = np.arange(window+stride, len(wind_data_rs)) #Indices to start each time series at
wind_data_rs_arr, flags = ksc.input_window(wind_data_rs, inds, in_keys, window, stride) #Break the array into input timeseries
predict_bs_arr = model.predict(wind_data_rs_arr)
predict_bs = pd.DataFrame(columns = ['Epoch', 'B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ne', 'Ne_sig', 'interp_frac'])
predict_bs['Epoch'] = wind_data_rs['Epoch'][(window+stride):]
predict_bs['interp_frac'] = flags
predict_bs[['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne']] = ds.tar_scaler.inverse_transform(predict_bs_arr[:, ::2])
predict_bs[['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig', 'Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig', 'Ne_sig']] = np.abs(ds.tar_scaler.inverse_transform(predict_bs_arr[:, ::2] + predict_bs_arr[:, 1::2]) - ds.tar_scaler.inverse_transform(predict_bs_arr[:, ::2]))
predict_bs.to_hdf(datapath + 'outputs.h5', key = 'prime_predict_bs')