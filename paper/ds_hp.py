import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
plt.style.use('../plots/paper.mplstyle') #Use custom stylesheet
import ks_custom as ksc
import tensorflow as tf
from tensorflow import keras as ks


taikonaut_colors = [(38/255.0, 70/255.0, 83/255.0),
                  (42/255.0, 157/255.0, 143/255.0),
                  (233/255.0, 196/255.0, 106/255.0),
                  (244/255.0, 162/255.0, 97/255.0),
                  (231/255.0, 111/255.0, 81/255.0)]
taikonaut = LinearSegmentedColormap.from_list('taikonaut', taikonaut_colors, N=10000)
datapath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/data/' #Point at my SCC data folder
modelpath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/modelstore/' #Point at my SCC model folder
plotpath = '/projectnb/sw-prop/obrienco/swprop/SW-Prop/plots/' #Point at my SCC plot folder

#Fix Random Seed
seed = 888 #Lucky!
tf.random.set_seed(seed) #Seed the TF generator
np.random.seed(seed) #Seed the NP generator

#SCC Boilerplate
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

windows = [30, 35, 40, 45, 50, 55, 60] #Windows of data for each input (100s units)
strides = [14, 16, 18, 20, 22, 24] #Strides away from the target prediction time (100s units)
fractions = [0.05, 0.1, 0.15] #Tolerance of %window data in a row that can be interpolated
epochs = 50 #Epochs to train each model

record = pd.DataFrame([])
for window in windows:
    for stride in strides:
        for fraction in fractions:
            ds = ksc.load_dataset(datapath + 'mms_data.h5', datapath + 'wind_data.h5','sw', in_keys, tar_keys, split_frac=0.2, window=window, stride=stride, inter_frac=int(fraction*window), flag='tdelt', tar_storekey = 'old_targets', conesize = np.pi/2, vx_cut=False)
            #Quick model
            model = ks.Sequential([ks.layers.GRU(units=192),
                                   ks.layers.Dense(units=352, activation='elu'),
                                   ks.layers.Dense(units=48, activation='elu'),
                                   ks.layers.LayerNormalization(),
                                   ks.layers.Dropout(0.2),
                                   ks.layers.Dense(ds.tar_full.shape[1]*2,activation='linear')
                                  ])
            model.compile(optimizer=tf.optimizers.Adamax(learning_rate=1e-4), loss=ksc.crps_loss)

            history = model.fit(ds.in_train, ds.tar_train, epochs=epochs, validation_split = 0.25) #Fit the test model to this dataset
            hist = pd.DataFrame(history.history) #Get the training and validation losses
            entry = pd.DataFrame([[window, stride, fraction, hist['loss'][epochs-1], hist['val_loss'][epochs-1], np.min(hist['val_loss'])]], columns = ['window', 'stride', 'fraction', 'loss', 'val_loss', 'min_val_loss'])
            record = record.append(entry, ignore_index = True)
record.to_hdf(modelpath + 'hyperparameters.h5', key = 'sw_hp_20230526', mode = 'a')

#Plot the results
hps = record
fig = plt.figure(figsize = (7, 7))
dotsize = 2000 #Size of the outer dot
steps = [1/2, 1/6] #Steps for the inner dots
scatter = plt.scatter(hps['window'][hps['fraction']==0.15], hps['stride'][hps['fraction']==0.15], c = hps['val_loss'][hps['fraction']==0.15], s = dotsize, vmax=hps['val_loss'].max(), vmin=hps['val_loss'].min(), cmap = taikonaut)
plt.scatter(hps['window'][hps['fraction']==0.10], hps['stride'][hps['fraction']==0.10], c = hps['val_loss'][hps['fraction']==0.10], s = dotsize*steps[0], vmax=hps['val_loss'].max(), vmin=hps['val_loss'].min(), cmap = taikonaut)
plt.scatter(hps['window'][hps['fraction']==0.05], hps['stride'][hps['fraction']==0.05], c = hps['val_loss'][hps['fraction']==0.05], s = dotsize*steps[1], vmax=hps['val_loss'].max(), vmin=hps['val_loss'].min(), cmap = taikonaut)
plt.xlabel('Window Size')
plt.xticks(hps['window'].unique())
plt.xlim(hps['window'].unique().min()-2.5, hps['window'].unique().max()+2.5)
plt.ylabel('Stride')
plt.yticks(hps['stride'].unique())
plt.ylim(hps['stride'].unique().min()-1, hps['stride'].unique().max()+1)
plt.title('Dataset Hyperparameters', fontsize = 18)
#Remove ticks and spines
plt.gca().tick_params(axis='both', which='major', labelsize=14, color = 'white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
cbar_ax = fig.add_axes([1.02, 0.4, 0.015, 0.5]) #Create an axis for the colorbar
cbar = fig.colorbar(scatter, cax = cbar_ax) #Add the colorbar
#cbar.set_ticks([0.19, 0.20, 0.21])
cbar.outline.set_visible(False) #Remove the colorbar outline
cbar.set_label('CRPS Validation Loss') #Add a label to the colorbar

plt.tight_layout()
plt.savefig(plotpath + 'ds_hyperparameters_20230526.pdf', bbox_inches = 'tight')