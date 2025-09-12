import numpy as np
import pandas as pd
import gc
import matplotlib.pyplot as plt
from cdasws import CdasWs
from loguru  import logger
from time import sleep
from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

DATAPATH = "~/data/" # Base data directory
cdas = CdasWs() # Open connection to CDAS

def load_range(dataset_name, var_list, start_date, end_date, load_freq = '1MS', time_name = 'Epoch', resample_freq = None, interp_freq = None, verbose = False, retries = 5):
    dates = pd.date_range(start_date, end_date, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist()
    df_list = []
    for i in np.arange(len(dates)-1):
        if verbose:
            logger.info(f"Processing {dates[i]}")
        loaded = False
        tries = 0
        while loaded == False:
            if tries > retries:
                break
            try:
                df_stage = cdf_to_df_remote(dataset_name, var_list, dates[i], dates[i+1], time_name = time_name)
                loaded = True
                tries = 0
            except (ConnectionError, ProtocolError):
                tries += 1
                sleep(5) # Wait a bit for the connection to chill out
                logger.info(f'Connection error for {dates[i]}. Retrying {tries} of {retries} times.')
            except TypeError:
                logger.info(f'No data for {dates[i]}')
                break
            except ValueError:
                logger.info(f'Corrupted data for {dates[i]}')
                break
            except Exception as err:
                logger.info(f"Unexpected {err=}, {type(err)=} on {dates[i]}")
                break
        if loaded == True: # Did we successfully load the data in the above while loop?
            if resample_freq is not None: # Do we want to downsample the loaded data? (Useful for high-cadence data like FGM)
                times = pd.date_range(dates[i], dates[i+1], freq=resample_freq) #Make new time intervals
                bin_subset = pd.IntervalIndex.from_arrays(times[:-1], times[1:], closed='left') #Create binlike object from times
                group = df_stage.groupby(pd.cut(df_stage[time_name], bin_subset), observed = False) #Bin the data
                binned_df_stage = group.mean() #Get the mean of the binned data
                binned_df_stage['count'] = group.count()[time_name] #Get the number of data points in each bin
                binned_df_stage[time_name] = bin_subset.left #Add the start of the bins as the Epoch
                df_list.append(binned_df_stage)
                del df_stage # Explicitly remove the raw data dataframe from memory in case it's HUGE (like FGM data!)
            elif interp_freq is not None: # Do we want to inerpolate the loaded data? (Useful for low-cadence data like SWE)
                times = pd.date_range(dates[i], dates[i+1], freq=interp_freq) #Make new time intervals
                df_interp = pd.DataFrame([])
                df_interp[time_name] = times[:-1] # For some reason date_range includes the last time
                for var in df_stage.columns:
                    if (var == time_name): # Skip the time column because we already have it
                        continue
                    df_interp[var] = np.interp(df_interp[time_name], df_stage[time_name], df_stage[var])
                df_list.append(df_interp)
            else: # If not resampling, we just append dis_df_stage directly
                df_list.append(df_stage)
        gc.collect() # We're running out of memory when using this function sometimes, so maybe this will help
    dataframe = pd.concat(df_list, ignore_index = True)
    return dataframe

def cdf_to_df_remote(dataset_name, var_list, start_date, end_date, time_name = 'Epoch'):
    dataframe = pd.DataFrame([])
    data = cdas.get_data(dataset_name, var_list, start_date, end_date)
    dataframe[time_name] = data[1][time_name] # This one is special because it's not in the var_list
    for var in var_list:
        if (var == 'mms1_dis_energyspectr_omni_fast'): 
            # If you need to handle more than one special variable, you should define a global with all special variables and run the if
            # statement with is in that list, then call a function handle_special_variable here that handles all the special cases.
            dataframe['SW_table'] = (data[1]['mms1_dis_energy_fast'][:,0] >= 190) #Solar wind energy-azimuth table starts at ~190-~210 eV
        if (data[1][var].ndim == 2): # We gotta handle vector data differently because the ending structure must be 2D
            for i in range(data[1][var].shape[-1]):
                dataframe[var+'_'+str(i)] = data[1][var][:,i]
        elif (data[1][var].ndim == 1):
            dataframe[var] = data[1][var]
        else:
            raise ValueError(f"Variable {var} in {dataset_name} has {data[1][var].ndim} dimensions (perhaps it is a distribution function or a constant).")
    dataframe[time_name] = pd.to_datetime(dataframe[time_name], utc=True)
    return dataframe

# Load the data from MMS and Wind
save_raw = True # Whether to save the raw data in the mms_data.h5 and wind_data.h5 HDF file
cadence = '1min'
datestrs = ['2015-09-01 00:00:00+00:00', '2025-01-01 00:00:00+00:00']
fpi_i_var_list = [
    'mms1_dis_bulkv_gse_fast', # V_gse vector for ions in km/s
    'mms1_dis_numberdensity_fast',  # n_i in cm**-3
    'mms1_dis_energyspectr_omni_fast', # Used to determine if SW energy-azimuth table is active (SW_table bool). Not included directly in outputs
    'mms1_dis_temppara_fast', # T_i parallel to B in eV
    'mms1_dis_tempperp_fast', # T_i perpendicular to B in eV
]
fpi_e_var_list = [
    'mms1_des_numberdensity_fast', # n_e in cm**-3
    'mms1_des_temppara_fast', # T_e parallel to B in eV
    'mms1_des_tempperp_fast', # T_i parallel to B in eV
    'mms1_des_bulkv_gse_fast', # V_gse vector for electrons in km/s
]
mec_var_list = [
    'mms1_mec_r_gsm', # SC position vector (GSM) in km
    'mms1_mec_r_gse', # SC position vector (GSE) in km
]
fgm_var_list = [
    'mms1_fgm_b_gsm_srvy_l2', # DC B field vector (GSM) in nT 
    'mms1_fgm_b_gse_srvy_l2', # DC B field vector (GSE) in nT
    'mms1_fgm_flag_srvy_l2', # B field quality flag (not sure why this is here)
]
mfi_var_list = [
    'BGSM', # DC B field vector (GSM) in nT
    'BGSE', # DC B field vector (GSE) in nT
    'PGSM', # Wind position vector (GSM) in RE
    'PGSE', # Wind position vector (GSE) in RE
]
swe_var_list = [
    'Np', # Proton density in cm**-3
    'V_GSE', # Ion flow velocity vector (GSE) in km/s
    'V_GSM', # Ion flow velocity vecotr (GSM) in km/s
    'THERMAL_SPD', # Proton thermal speed in km/s
    'Pressure', # Dynamic pressure in nPa, technically calculable from the others but why bother
    'QF_V', # Velocity quality flag
    'QF_Np', # Proton density quality flag
]
fpi_i_dataset = 'MMS1_FPI_FAST_L2_DIS-MOMS' # Level 2 fast moments
fpi_e_dataset = 'MMS1_FPI_FAST_L2_DES-MOMS' # Level 2 fast moments
mec_dataset = 'MMS1_MEC_SRVY_L2_EPHT89D' # Level 2 survey with EPHT89D field model
fgm_dataset = 'MMS1_FGM_SRVY_L2' # Level 2 survey
mfi_dataset = 'WI_H0_MFI' # Key parameter B from Wind
swe_dataset = 'WI_K0_SWE' # Key parameter plasma from Wind
logger.info(f"Loading FPI ion data.")
fpi_i_data = load_range(fpi_i_dataset, fpi_i_var_list, datestrs[0], datestrs[1], load_freq='1MS', resample_freq=cadence, verbose=True)
logger.info(f"Loading FPI electron data.")
fpi_e_data = load_range(fpi_e_dataset, fpi_e_var_list, datestrs[0], datestrs[1], load_freq='1MS', resample_freq=cadence)
logger.info(f"Loading MEC data.")
mec_data = load_range(mec_dataset, mec_var_list, datestrs[0], datestrs[1], load_freq='1MS', resample_freq=cadence)
logger.info(f"Loading FGM data.")
fgm_data = load_range(fgm_dataset, fgm_var_list, datestrs[0], datestrs[1], load_freq='1D', resample_freq=cadence) # load_freq is shorter here, if longer than ~10D RAM is overloaded. 1D is no slower than 10D
logger.info(f"Loading SWE data.")
swe_data = load_range(swe_dataset, swe_var_list, datestrs[0], datestrs[1], load_freq='1MS', interp_freq=cadence)
logger.info(f"Loading FPI data.")
mfi_data = load_range(mfi_dataset, mfi_var_list, datestrs[0], datestrs[1], load_freq='1MS', interp_freq=cadence) # MFI data is already minutely, it's just on the half-minute so we still have to interp

if save_raw:
    logger.info(f"Saving raw data.")
    fpi_i_data.to_hdf(DATAPATH + 'mms/mms_data.h5', key = 'fpi_i_1min')
    fpi_e_data.to_hdf(DATAPATH + 'mms/mms_data.h5', key = 'fpi_e_1min')
    mec_data.to_hdf(DATAPATH + 'mms/mms_data.h5', key = 'mec_1min')
    fgm_data.to_hdf(DATAPATH + 'mms/mms_data.h5', key = 'fgm_1min')
    swe_data.to_hdf(DATAPATH + 'wind/wind_data.h5', key = 'swe_1min')
    mfi_data.to_hdf(DATAPATH + 'wind/wind_data.h5', key = 'mfi_1min')

# Load the labeled MMS data from Toy-Edens et al. 2024 https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JA032431
# Downloaded from zenodo.org/records/10491878
logger.info(f"Loading labeled data.")
mms_labels = pd.read_csv(DATAPATH + 'mms/labeled_sunside_data.csv') # This object will have the MMS data from FPI, FGM, and MEC added to it.
mms_labels = mms_labels[mms_labels['probe'] == 'mms1'] # Select only MMS1 so we don't overtrain
mms_labels['Epoch'] = pd.to_datetime(mms_labels['Epoch'], utc = True) # Change Epoch to a datetime rather than an Epoch
mms_labels.index = mms_labels['Epoch'] # Turn the labels index into Epoch to fit it into a larger minutely dataframe to identify gaps

# Load just the times when MMS is "stable" in the solar wind for 15 or more minutes (Used to create dataset stability flag)
logger.info(f"Loading SW regions.")
sw_regions = pd.read_csv(DATAPATH + 'mms/solar_wind_region_list.csv')
sw_regions = sw_regions[sw_regions['probe'] == 'mms1'] # Select only MMS1 so we don't overtrain
sw_regions['start'] = pd.to_datetime(sw_regions['start'], utc = True)
sw_regions['stop'] = pd.to_datetime(sw_regions['stop'], utc = True)  # Stops are on the 59th second of the minute so should be inclusive when binning MMS data

# Load just the times when MMS is "stable" in the magnetosheath for 15 or more minutes (Used to create dataset stability flag)
logger.info(f"Loading SH regions.")
sh_regions = pd.read_csv(DATAPATH + 'mms/magnetosheath_region_list.csv')
sh_regions = sh_regions[sh_regions['probe'] == 'mms1'] # Select only MMS1 so we don't overtrain
sh_regions['start'] = pd.to_datetime(sh_regions['start'], utc = True)
sh_regions['stop'] = pd.to_datetime(sh_regions['stop'], utc = True)  # Stops are on the 59th second of the minute so should be inclusive when binning MMS data

# Load just the times when MMS is "stable" in the magnetosphere for 15 or more minutes (Used to create dataset stability flag)
logger.info(f"Loading MS regions.")
ms_regions = pd.read_csv(DATAPATH + 'mms/magnetosphere_region_list.csv')
ms_regions = ms_regions[ms_regions['probe'] == 'mms1'] # Select only MMS1 so we don't overtrain
ms_regions['start'] = pd.to_datetime(ms_regions['start'], utc = True)
ms_regions['stop'] = pd.to_datetime(ms_regions['stop'], utc = True)  # Stops are on the 59th second of the minute so should be inclusive when binning MMS data

# Load just the times when MMS is "stable" in the ion foreshock for 15 or more minutes (Used to create dataset stability flag)
logger.info(f"Loading FS regions.")
fs_regions = pd.read_csv(DATAPATH + 'mms/ion_foreshock_region_list.csv')
fs_regions = fs_regions[fs_regions['probe'] == 'mms1'] # Select only MMS1 so we don't overtrain
fs_regions['start'] = pd.to_datetime(fs_regions['start'], utc = True)
fs_regions['stop'] = pd.to_datetime(fs_regions['stop'], utc = True)  # Stops are on the 59th second of the minute so should be inclusive when binning MMS data

# Create a dataframe to put all the data into
combo_df = pd.DataFrame([])
logger.info(f"Creating combined dataframe.")
combo_df['Epoch'] = pd.date_range(mms_labels['Epoch'].min(), mms_labels['Epoch'].max(), freq = "1min") # Make a time for every minute so gaps can be identified as nans
# combo_df = combo_df.loc[
#     (combo_df['Epoch'] >= pd.to_datetime(datestrs[0]))& # REMOVE THIS CUT IN PRODUCTION THIS IS JUST FOR TESTING
#     (combo_df['Epoch'] < pd.to_datetime(datestrs[1])),
#     :
# ]
combo_df.index = combo_df['Epoch'] # Make the index a time as well to simplify merging mms_labels in
logger.info(f"Merging raw data into combined dataframe.")
for dataframe in [mms_labels, fpi_i_data, fpi_e_data, mec_data, fgm_data, swe_data, mfi_data]:
    dataframe.index = dataframe['Epoch']
    for key in dataframe.columns:
        if (key == 'Epoch'):
            continue
        combo_df[key] = dataframe[key]
    dataframe = dataframe.reset_index(drop = True)
combo_df = combo_df.reset_index(drop = True)

# Mark entries when MMS is "stable" in a given region
logger.info(f"Marking stable regions.")
combo_df['stable'] = np.zeros(len(combo_df))
for region_df in [sw_regions, sh_regions, ms_regions, fs_regions]:
    for idx in region_df.index: # Loop over rows in the region dataframe to assess stability
        region = region_df.loc[idx]
        combo_df.loc[
            (combo_df['Epoch'] >= region['start'])&
            (combo_df['Epoch'] <= region['stop']),
            'stable'
        ] = 1

# Save the combined dataframe to an HDF
logger.info(f"Saving combined dataframe.")
combo_df.to_hdf(DATAPATH + 'combined_data.h5', key = '1min_mms_wind')