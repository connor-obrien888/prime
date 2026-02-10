import numpy as np
import pandas as pd
from loguru  import logger

logger.info(f"Loading input data.")
wind_data = pd.read_hdf('/glade/u/home/cobrien/data/magnetotail/wind_data_full.h5', key = 'wind_full')

logger.info(f"Loading Geotail strict data.")
geotail_strict = pd.read_csv('/glade/u/home/cobrien/data/magnetotail/geotail_strict.csv', index_col=0)
geotail_strict['Epoch'] = pd.to_datetime(geotail_strict['datetime'], utc = True, format = 'mixed')
geotail_strict['modified_named_label'] = 'magnetotail'

logger.info(f"Resampling Geotail strict data.")
geotail_strict_resample = pd.DataFrame([], columns = geotail_strict.columns)
for key in geotail_strict.columns:
    if key == 'Epoch':
        geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch']].resample('100s', on='Epoch').first().index
    elif geotail_strict[key].dtype == 'O': # Object columns must use .first()
        geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch', key]].resample('100s', on='Epoch').first()[key]
    else: # Float/int/datetime columns must use .mean()
        geotail_strict_resample[key] = geotail_strict.loc[:, ['Epoch', key]].resample('100s', on='Epoch').mean()[key]
geotail_strict_resample = geotail_strict_resample.dropna() # Empty times are nans
geotail_strict_resample = geotail_strict_resample.reset_index(drop = True) # Scrub the time index the resampling creates

logger.info(f"Indexing Geotail strict data.")
geotail_strict_resample['input_idx'] = np.nan
for idx, time in enumerate(geotail_strict_resample['Epoch']):
    geotail_strict_resample.loc[idx, 'input_idx'] = wind_data[(wind_data['Epoch'] == time)].index[0]

logger.info(f"Saving Geotail strict data.")
geotail_strict_resample.to_hdf('/glade/u/home/cobrien/data/magnetotail/resampled_datasets.h5', key = 'geotail_strict_100s')

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

logger.info(f"Plasmasheet processing complete.")