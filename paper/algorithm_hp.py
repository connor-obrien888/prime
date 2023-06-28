import numpy as np
import os
import tensorflow as tf
from tensorflow import keras as ks
import keras_tuner as kt
from keras.utils.generic_utils import get_custom_objects
import ks_custom as ksc

datapath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/data/' #Point at my SCC data folder
modelpath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/modelstore/' #Point at my SCC model folder
plotpath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/plots/' #Point at my SCC plot folder
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

#Load the data
in_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
tar_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne'] #Targets from MMS dataset to match

window = 70 #Input window length (100s units)
stride = 16 #Stride away from target (100s units)
inter_frac = 0.15 #How many interpolated points in a row to tolerate
flag = 'tdelt' #Whether to cut inputs as a percent of total or length of maximum interpolated stretch of data
ds = ksc.load_dataset(datapath + 'mms_data.h5', datapath + 'wind_data.h5', 'sw', in_keys, tar_keys, split_frac=0.2, window=window, stride=stride, inter_frac=int(inter_frac*window), flag=flag, conesize = np.pi/2)

#Define the model
def model_builder(hp):
    hp_units0 = hp.Int('units0', min_value=128, max_value=640, step=32) #Units in the first (recurrent) layer
    hp_units1 = hp.Int('units1', min_value=128, max_value=640, step=32) #Units in the second (dense) layer
    hp_units2 = hp.Int('units2', min_value=16, max_value=128, step=16) #Units in the third (dense) layer
    hp_units3 = hp.Int('units3', min_value=16, max_value=128, step=16) #Units in the fourth (dense) layer
    hp_regularize = hp.Choice('regularize', values = [True, False]) #Choice of regularization in the recurrent layers
    hp_drop_rate = hp.Choice('drop_rate', values = [0.2,0.35,0.5]) #Rate of node dropout
    model = ks.Sequential([ks.layers.GRU(units=hp_units0)])
    if hp_regularize: #If we're regularizing, add a dropout layer and a layer normalization layer
        model.add(ks.layers.Dropout(hp_drop_rate))
        model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.Dense(units=hp_units1, activation='elu'))
    if hp_regularize: #If we're regularizing, add a dropout layer and a layer normalization layer
        model.add(ks.layers.Dropout(hp_drop_rate))
        model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.Dense(units=hp_units2, activation='elu'))
    if hp_regularize: #If we're regularizing, add a dropout layer and a layer normalization layer
        model.add(ks.layers.Dropout(hp_drop_rate))
        model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.Dense(units=hp_units3, activation='elu'))
    model.add(ks.layers.LayerNormalization())
    model.add(ks.layers.Dropout(hp_drop_rate))
    model.add(ks.layers.Dense(ds.tar_full.shape[1]*2,activation='linear'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3,1e-4,1e-5]) #Learning rate of Adam
    #Compile the model with an optimization routine and a loss function
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate), loss=ksc.crps_loss, metrics = ksc.mse_metric) #Compile the model with an optimization routine and MAE loss function
    return model

#Make the hyperparemeter tuner
name = 'prime_sw_003' #Name of the training run to be saved (if train) or loaded (if not train)
tuner = kt.Hyperband(model_builder, #Hyperband optimization routine
                     objective='val_loss', #Minimize the validation loss
                     max_epochs = 150, #Maximum number of epochs for each model
                     factor = 3, #Hyperband factor
                     directory = modelpath+'smokesite', #Directory to save the models
                     project_name = name) #Name of the training run

# Conduct the hyperparameter search
tuner.search(ds.in_train, ds.tar_train, validation_split = 0.25) #Search for the best hyperparameters
