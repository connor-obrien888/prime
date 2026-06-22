import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar, Timer, LearningRateFinder
import omegaconf
from loguru import logger


# Add the prime_torch file to the system path so we can import it
import sys
sys.path.append("/glade/u/home/cobrien/prime/prime_lib/primesw")
from data import SWDataset, SWDataModule
from prime_torch import crps, SWRegressor

torch.set_float32_matmul_precision('medium')

# Load the Geotail (strict) model (handles density and ion temperature)
geo_config = '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_stricttemp.yaml'
geo_cfg = omegaconf.OmegaConf.load(
    geo_config
)
geo_model = SWRegressor.load_from_checkpoint(
    "/glade/u/home/cobrien/data/prime/tensorboard_logs/psstrict/version_1/checkpoints/epoch=199-step=11000.ckpt"
)

# Load the MMS model (handles ion temperature, electron temperature, and their ratio)
mms_config = '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_mmstemp.yaml'
mms_cfg = omegaconf.OmegaConf.load(
    mms_config
)
mms_model = SWRegressor.load_from_checkpoint(
    "/glade/u/home/cobrien/data/prime/tensorboard_logs/psmms/version_2/checkpoints/epoch=199-step=1600.ckpt"
)

# Load the data
wind_data = pd.read_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'wind_full', mode = 'r')

# Cut the input wind data to just the interval covered by the themis data
bounds = [
    pd.to_datetime("20220819 05:45:00+0000"), #Start
    pd.to_datetime("20220819 07:15:00+0000"), #Stop
]
wind_data = wind_data.loc[
    (wind_data['Epoch'] <= bounds[1])&
    (wind_data['Epoch'] >= bounds[0] - pd.Timedelta(seconds = (geo_model.stride + geo_model.window) * 100)), # Both models have the same window and stride! EZPZ
    :
]

# Define the grid extent
x_extent = [-30, -5]
y_extent = [-15, 15]
gridsize = 0.1
r_cut = 5 # Cut out this many RE from Earth where the outputs are invalid

# Initialize the grid
x_arr = np.arange(x_extent[0], x_extent[1], gridsize)
y_arr = np.arange(y_extent[0], y_extent[1], gridsize)
output_grid = np.empty((len(wind_data) - mms_cfg.data.window, len(x_arr), len(y_arr), (len(geo_cfg.data.target_features) + len(mms_cfg.data.target_features))*2))

geo_position = pd.DataFrame(np.zeros((len(wind_data), 3)), columns = geo_cfg.data.position_features) #They are the same except for their column names
mms_position = pd.DataFrame(np.zeros((len(wind_data), 3)), columns = mms_cfg.data.position_features) #These dataframes will hold the positions we're decoding to.

# Loop over each grid cell
for i, x in enumerate(x_arr):
    for j, y in enumerate(y_arr):
        geo_position[geo_cfg.data.position_features[0]] = x
        geo_position[geo_cfg.data.position_features[1]] = y
        mms_position[mms_cfg.data.position_features[0]] = x
        mms_position[mms_cfg.data.position_features[1]] = y
        if np.sqrt(x**2 + y**2) < r_cut:
            output_grid[:, i, j, :] = np.nan
        else:
            output_grid[:, i, j, :(len(geo_cfg.data.target_features)*2)] = geo_model.predict(wind_data, geo_position).drop(columns = 'Epoch').to_numpy()
            output_grid[:, i, j, (len(geo_cfg.data.target_features)*2):] = mms_model.predict(wind_data, mms_position).drop(columns = 'Epoch').to_numpy()
    logger.info(f"Cell {i+1} of {len(x_arr)} finished.")
logger.info(f"Saving output grid.")
np.save('20220819_2D.npy', output_grid)
logger.info(f"Saved output grid.")