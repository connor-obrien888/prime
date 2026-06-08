import numpy as np
import pandas as pd
from loguru  import logger

logger.info(f"Loading input data.")
# wind_data = pd.read_hdf('/glade/u/home/cobrien/data/magnetotail/wind_data_full.h5', key = 'wind_full')

############################
# STRICT SECTION
############################

# logger.info(f"Loading Geotail strict data.")
# geotail_strict = pd.read_csv('/glade/u/home/cobrien/data/magnetotail/geotail_strict.csv', index_col=0)
# geotail_strict['Epoch'] = pd.to_datetime(geotail_strict['datetime'], utc = True, format = 'mixed')
# geotail_strict['modified_named_label'] = 'magnetotail'

# logger.info(f"Resampling Geotail strict data.")
# geotail_strict_resample = pd.DataFrame([], columns = geotail_strict.columns)
# for key in geotail_strict.columns:
#     if key == 'Epoch':
#         geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch']].resample('100s', on='Epoch').first().index
#     elif geotail_strict[key].dtype == 'O': # Object columns must use .first()
#         geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch', key]].resample('100s', on='Epoch').first()[key]
#     else: # Float/int/datetime columns must use .mean()
#         geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch', key]].resample('100s', on='Epoch').mean()[key]
# geotail_strict_resample = geotail_strict_resample.dropna() # Empty times are nans
# geotail_strict_resample = geotail_strict_resample.reset_index(drop = True) # Scrub the time index the resampling creates

# logger.info(f"Indexing Geotail strict data.")
# geotail_strict_resample['input_idx'] = np.nan
# for idx, time in enumerate(geotail_strict_resample['Epoch']):
#     geotail_strict_resample.loc[idx, 'input_idx'] = wind_data[(wind_data['Epoch'] == time)].index[0]

# logger.info(f"Saving Geotail strict data.")
# geotail_strict_resample.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'geotail_strict_100s')

############################
# FLEX SECTION
############################

# logger.info(f"Loading Geotail flexible data.")
# geotail_flexible = pd.read_csv('/glade/u/home/cobrien/data/magnetotail/geotail_flexible.csv', index_col=0)
# geotail_flexible['Epoch'] = pd.to_datetime(geotail_flexible['datetime'], utc = True, format = 'mixed')
# geotail_flexible['modified_named_label'] = 'magnetotail'

# logger.info(f"Resampling Geotail flexible data.")
# geotail_flexible_resample = pd.DataFrame([], columns = geotail_flexible.columns)
# for key in geotail_flexible.columns:
#     if key == 'Epoch':
#         geotail_flexible_resample[key] = geotail_flexible.loc[:, ['Epoch']].resample('100s', on='Epoch').first().index
#     elif geotail_flexible[key].dtype == 'O': # Object columns must use .first()
#         geotail_flexible_resample[key] = geotail_flexible.loc[:, ['Epoch', key]].resample('100s', on='Epoch').first()[key]
#     else: # Float/int/datetime columns must use .mean()
#         geotail_flexible_resample[key] = geotail_flexible.loc[:, ['Epoch', key]].resample('100s', on='Epoch').mean()[key]
# geotail_flexible_resample = geotail_flexible_resample.dropna() # Empty times are nans
# geotail_flexible_resample = geotail_flexible_resample.reset_index(drop = True) # Scrub the time index the resampling creates

# logger.info(f"Indexing Geotail flexible data.")
# geotail_flexible_resample['input_idx'] = np.nan
# for idx, time in enumerate(geotail_flexible_resample['Epoch']):
#     geotail_flexible_resample.loc[idx, 'input_idx'] = wind_data[(wind_data['Epoch'] == time)].index[0]

# logger.info(f"Saving Geotail flexible data.")
# geotail_flexible_resample.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'geotail_flexible_100s')

############################
# STRICT + HIGH DENSITY SECTION
############################

# logger.info(f"Loading Geotail strict + high density data.")
# geotail = pd.read_csv('/glade/u/home/cobrien/data/magnetotail/geotail_stricthighdensity.csv', index_col=0)
# geotail['Epoch'] = pd.to_datetime(geotail['datetime'], utc = True)
# geotail['modified_named_label'] = 'magnetotail'

# logger.info(f"Resampling Geotail strict + high density data.")
# geotail_resample = pd.DataFrame([], columns = geotail.columns)
# for key in geotail.columns:
#     if key == 'Epoch':
#         geotail_resample[key] = geotail.loc[:, ['Epoch']].resample('100s', on='Epoch').first().index
#     elif geotail[key].dtype == 'O': # Object columns must use .first()
#         geotail_resample[key] = geotail.loc[:, ['Epoch', key]].resample('100s', on='Epoch').first()[key]
#     else: # Float/int/datetime columns must use .mean()
#         geotail_resample[key] = geotail.loc[:, ['Epoch', key]].resample('100s', on='Epoch').mean()[key]
# geotail_resample = geotail_resample.dropna() # Empty times are nans
# geotail_resample = geotail_resample.reset_index(drop = True) # Scrub the time index the resampling creates

# logger.info(f"Indexing Geotail strict + high density data.")
# geotail_resample['input_idx'] = np.nan
# for idx, time in enumerate(geotail_resample['Epoch']):
#     geotail_resample.loc[idx, 'input_idx'] = wind_data[(wind_data['Epoch'] == time)].index[0]

############################
# MMS SECTION
############################

# logger.info(f"Loading MMS data.")
# mms = pd.read_csv('/glade/u/home/cobrien/data/magnetotail/mms_temperatures_proc_v2.csv')
# mms['Epoch'] = pd.to_datetime(mms['time'], utc = True)
# mms['modified_named_label'] = 'magnetotail'

# logger.info(f"Resampling MMS data.")
# mms_resample = pd.DataFrame([], columns = mms.columns)
# for key in mms.columns:
#     if key == 'Epoch':
#         mms_resample[key] = mms.loc[:, ['Epoch']].resample('100s', on='Epoch').first().index
#     elif mms[key].dtype == 'O': # Object columns must use .first()
#         mms_resample[key] = mms.loc[:, ['Epoch', key]].resample('100s', on='Epoch').first()[key]
#     else: # Float/int/datetime columns must use .mean()
#         mms_resample[key] = mms.loc[:, ['Epoch', key]].resample('100s', on='Epoch').mean()[key]
# mms_resample = mms_resample.dropna() # Empty times are nans
# mms_resample = mms_resample.reset_index(drop = True) # Scrub the time index the resampling creates

# logger.info(f"Indexing MMS data.")
# mms_resample['input_idx'] = np.nan
# for idx, time in enumerate(mms_resample['Epoch']):
#     mms_resample.loc[idx, 'input_idx'] = wind_data[(wind_data['Epoch'] == time)].index[0]

# logger.info(f"Saving data.")
# wind_data.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'wind_full')
# geotail_resample.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'geotail_stricthighdensity_100s')
# mms_resample.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'mms_temperatures_proc_v2_100s')

############################
# TEST DATASET SECTION
############################
import torch
torch.set_float32_matmul_precision('medium')
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar, Timer, LearningRateFinder
import omegaconf

import sys
sys.path.append("/glade/u/home/cobrien/prime/prime_lib/primesw")
from data import SWDataset, SWDataModule
from prime_torch import crps, SWRegressor

configs = [
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet.yaml',
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_strict.yaml',
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_strict20.yaml',
    # "/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_flex20.yaml",
    # "/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_flex.yaml",
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_mms.yaml',
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_mms.yaml',
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_mms.yaml',
    # '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_mms.yaml',
    '/glade/u/home/cobrien/prime/prime_lib/configs/plasmasheet_strict.yaml',
]

checkpoints = [
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/pstesting/version_24/checkpoints/epoch=199-step=11000.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psstrict/version_1/checkpoints/epoch=199-step=11000.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psstrict/version_2/checkpoints/epoch=199-step=10200.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psflex/version_2/checkpoints/epoch=199-step=25800.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psflex/version_3/checkpoints/epoch=199-step=27600.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psmms/version_2/checkpoints/epoch=199-step=1600.ckpt",
    # "/glade/u/home/cobrien/data/prime/checkpoints/psmms_plasmasheet_mmstemp.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psmms/version_5/checkpoints/epoch=499-step=4000.ckpt",
    # "/glade/u/home/cobrien/data/prime/tensorboard_logs/psmms/version_6/checkpoints/epoch=799-step=6400.ckpt",
    "/glade/u/home/cobrien/data/prime/tensorboard_logs/psstrict/version_6/checkpoints/epoch=299-step=16500.ckpt",
]

savekeys = [
    # "pshigh_version24",
    # "psstrict_version1",
    # "psstrict20_version2",
    # "psflex20_version2",
    # "psflex_version3",
    # "psmms_version2",
    # "psmms_version4",
    # "psmms_version5",
    # "psmms_version6",
    "psstrict_version6",
]

for idx, config in enumerate(configs):
    logger.info(f"Generating {savekeys[idx]}")

    cfg = omegaconf.OmegaConf.load(
        config
    )

    # Running .setup() with this DataModule will take like an hour. Find a way to cache this dataloader
    datamodule = SWDataModule(
        target_features = cfg.data.target_features,
        input_features = cfg.data.input_features,
        position_features = cfg.data.position_features,
        interp_flags = cfg.data.interp_flags,
        region = cfg.data.region,
        cuts = cfg.data.cuts,
        cadence = cfg.data.cadence,
        interpolate = cfg.data.interpolate,
        window = cfg.data.window,
        stride = cfg.data.stride,
        interp_frac = cfg.data.interp_frac,
        trn_bounds = cfg.data.trn_bounds,
        val_bounds = cfg.data.val_bounds,
        tst_bounds = cfg.data.tst_bounds,
        batch_size = cfg.opt.batch_size,
        num_workers = cfg.opt.num_workers,
        datastore = cfg.data.datastore,
        in_key = cfg.data.in_key,
        tar_key = cfg.data.tar_key,
        scaler_type = cfg.data.scaler_type,
    )

    model = SWRegressor.load_from_checkpoint(
        checkpoints[idx]
    )
    model.eval() # Disables randomness, dropout, gradients

    datamodule.setup()

    y_hat = model(datamodule.tst_ds.input_data, datamodule.tst_ds.position_data) # Run an actual forward pass
    y_hat = y_hat.detach().cpu().numpy()
    y = datamodule.tst_ds.target_data

    # Try to initialize the return dataframe
    tar_scaled = pd.DataFrame([])
    tar_scaled["Epoch"] = pd.to_datetime(datamodule.tst_ds.target_timestamps)

    if y_hat.shape[1] == 2*len(model.tar_norm.keys()): # If the output is means + stdevs
        for i, feature in enumerate(model.tar_norm.keys()):
            tar_scaled[feature] = (y_hat[:, i*2] * model.tar_norm[feature][1]) + model.tar_norm[feature][0]
            tar_scaled[feature+'_target'] = (y[:, i] * model.tar_norm[feature][1]) + model.tar_norm[feature][0]
            tar_scaled[feature + '_std'] = ((y_hat[:, i*2] + y_hat[:, i*2 + 1]) * model.tar_norm[feature][1]) + model.tar_norm[feature][0] - tar_scaled[feature]
    else:
        for i, feature in enumerate(model.tar_norm.keys()):
            tar_scaled[feature] = (y_hat[:, i] * model.tar_norm[feature][1]) + model.tar_norm[feature][0]
            tar_scaled[feature+'_target'] = (y[:, i] * model.tar_norm[feature][1]) + model.tar_norm[feature][0]

    logger.info(f"Saving {savekeys[idx]}")
    tar_scaled.to_hdf("primeps_outputs.h5", key = savekeys[idx])

    # Synthetic SW 2D maps
    # Init synthetic solar wind datastructures
    synth_data = pd.DataFrame(np.zeros((model.window+1, len(datamodule.input_features))), columns = datamodule.input_features)
    # synth_data['Epoch'] = pd.to_datetime('2000-01-01 00:00:00+0000') # This time is unused, but model.predict() throws a warning when input data has no Epoch
    synth_pos = pd.DataFrame(np.zeros((model.window+1, 3)), columns = datamodule.position_features)

    synth_overrides = {
        'Ni' : 3,
        'Vi_xgse' : -380,
        'Vi_ygse' : - 30,
        'Vi_zgse' : -2.8,
        'Vth' : 36,
        'R_xgse' : 197.8,
        'R_ygse' : -0.03,
        'R_zgse' : -10.7,
        'B_xgsm' : -0.06,
        'B_ygsm' : 0.16,
        'B_zgsm' : -5,
        'SME' : 200,
        'SMR' : 0,
        'tilt' : 15,
    }

    # Override the prior values (currently empty) with new synthetic conditions
    for key in synth_overrides.keys():
        synth_data[key] = synth_overrides[key]

    # Define the grid extent
    x_extent = [-30, -5]
    y_extent = [-15, 15]
    gridsize = 0.1
    r_cut = 10 # Cut out this many RE from Earth where the outputs are invalid

    # Pick the solar wind conditions we're iterating over
    bz = [-5, 5, -5, 5]
    # # ni = [3, 3, 20, 20]
    vx = [-300, -300, -600, -600]
    # Dubyagin plot:
    # bz = [0, -2, 2, 2]
    # vx = [-400, -400, -400, -600]

    # Initialize the grid
    x_arr = np.arange(x_extent[0], x_extent[1], gridsize)
    y_arr = np.arange(y_extent[0], y_extent[1], gridsize)
    output_grid = np.zeros((len(bz), len(x_arr), len(y_arr), len(datamodule.target_features)*2))

    for k in range(len(bz)):
        synth_data['B_zgsm'] = bz[k] # Set new SW conditions for this frame
        # synth_data['Ni'] = ni[k]
        synth_data['Vi_xgse'] = vx[k]
        for i, x in enumerate(x_arr):
            for j, y in enumerate(y_arr):
                synth_pos[datamodule.position_features[0]] = x
                synth_pos[datamodule.position_features[1]] = y
                if np.sqrt(x**2 + y**2) < r_cut:
                    output_grid[k, i, j, :] = np.nan
                else:
                    output_grid[k, i, j, :] = model.predict_df(synth_data, synth_pos)

    np.save(f'/glade/u/home/cobrien/prime/prime_lib/notebooks/{savekeys[idx]}_2D_vxvaried.npy', output_grid)

    # # Vary Ni this time
    synth_data['Vi_xgse'] = synth_overrides['Vi_xgse'] # Reset the velocity now that we're done varying it
    bz = [-5, 5, -5, 5]
    ni = [3, 3, 20, 20]
    # vx = [-300, -300, -600, -600]
    # Dubyagin plot:
    # bz = [0, -2, 2, 2]
    # vx = [-400, -400, -400, -600]

    # Initialize the grid
    x_arr = np.arange(x_extent[0], x_extent[1], gridsize)
    y_arr = np.arange(y_extent[0], y_extent[1], gridsize)
    output_grid = np.zeros((len(bz), len(x_arr), len(y_arr), len(datamodule.target_features)*2))

    for k in range(len(bz)):
        synth_data['B_zgsm'] = bz[k] # Set new SW conditions for this frame
        synth_data['Ni'] = ni[k]
        # synth_data['Vi_xgse'] = vx[k]
        for i, x in enumerate(x_arr):
            for j, y in enumerate(y_arr):
                synth_pos[datamodule.position_features[0]] = x
                synth_pos[datamodule.position_features[1]] = y
                if np.sqrt(x**2 + y**2) < r_cut:
                    output_grid[k, i, j, :] = np.nan
                else:
                    output_grid[k, i, j, :] = model.predict_df(synth_data, synth_pos)

    np.save(f'/glade/u/home/cobrien/prime/prime_lib/notebooks/{savekeys[idx]}_2D_nivaried.npy', output_grid)

logger.info(f"Plasmasheet processing complete.")