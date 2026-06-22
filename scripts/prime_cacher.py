import numpy as np
import pandas as pd
from loguru import logger

DATAPATH = "~/data/" # Base data directory

# Make data to be ingested by a SolarWind kaipy object
datestrs = ['2010-01-01 00:00:00+00:00', '2025-01-01 00:00:00+00:00']
logger.info(f"Loading PRIME data.")
prime_df = pd.read_hdf(DATAPATH+'prime/old_prime.h5', key = 'prime_v1_100s')

# Resample PRIME data into a minute-cadence object
times = pd.date_range(datestrs[0], datestrs[1], freq='1min') #Make new time intervals
input_df = pd.DataFrame([])
input_df['Epoch'] = times[:-1] # For some reason date_range includes the last time
for var in prime_df.columns:
    if (var == 'Epoch'): # Skip the time column because we already have it
        continue
    logger.info(f"Interpolating {var}.")
    input_df[var] = np.interp(input_df['Epoch'], prime_df['Epoch'], prime_df[var])

# Load the supplemental OMNI data
logger.info(f"Loading OMNI data.")
omni_df = pd.read_hdf(DATAPATH+'prime/temp.h5', key = 'OMNI_HRO_1MIN')

# Merge the OMNI data into the input dataframe
logger.info(f"Merging PRIME and OMNI data.")
input_df = input_df.merge(omni_df, on = 'Epoch')

logger.info(f"Saving data.")
input_df.to_hdf(DATAPATH+'prime/prime_cache.h5', key = 'prime_v1_omni_1min')
logger.info(f"Saved combined PRIME and OMNI data.")
prime_df.to_hdf(DATAPATH+'prime/prime_cache.h5', key = 'prime_v1_100s')
logger.info(f"Saved PRIME data.")
omni_df.to_hdf(DATAPATH+'prime/prime_cache.h5', key = 'OMNI_HRO_1MIN')
logger.info(f"Saved OMNI data.")