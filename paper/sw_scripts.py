import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
import pytz
import pandas as pd

import sw_lib as sw

#This library contains script-like functions utilized for long operations such as saving data and making website plots.

# REGION CLASSIFICATION

def load_mms_data(start_date, end_date, instrument, dtype, sc, rate):
    '''
    Loads saved data for a given MMS instrument, data type, spacecraft, rate, and time range.
    '''
    raw_data = sw.load_util(start_date, end_date, instrument, dtype, sc, rate)
    data = sw.clean_columns(data, instrument, dtype, rate)
    del raw_data #Delete the raw data to save memory
    return data
    
def class_load(start = '2015-01-01', end = '2023-01-01', load_freq = '1D', storename = 'mms_data', key = 'mms_class', datapath = '../data/'):
    '''
    Loads the data from the MMS FPI instrument and classifies it using algorithm developed by Olshevsky et al. [2021].
    start: Start date of the data to load (YYYY-MM-DD)
    end: End date of the data to load (YYYY-MM-DD)
    load_freq: Frequency of data to load (e.g. '1D' for daily. Distributions are large, more than 1D is not recommended)
    storename: Name of the HDF5 file to store the data in
    key: Key to store the data under in the HDF5 file
    datapath: Path to the HDF5 file
    '''
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    class_df = pd.DataFrame([])
    for i in np.arange(len(dates)-1):
        class_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('MMS1_FPI_FAST_L2_DIS-DIST', ['mms1_dis_dist_fast'], dates[i], dates[i+1]) #Get the data
            dist_arr = np.asarray(data[1]['mms1_dis_dist_fast']) #Pull the 3D distribution
            label, probability = sw.olshevsky_class(dist_arr) #Classify the full array
            p_arr = sw.translate_probabilities(probability) #Store the class probabilities
            f_arr = sw.translate_labels(label, classifier = 'olshevsky') #Region flags (see translate_labels)
            class_df_stage['Epoch'] = data[1]['Epoch'] #Get the time array
            class_df_stage['region'] = f_arr #Get the region array
            class_df_stage['MSP_p'] = p_arr[:, 0] #Get the MSP probabilities
            class_df_stage['MSH_p'] = p_arr[:, 1] #Get the MSH probabilities
            class_df_stage['SW_p'] = p_arr[:, 2] #Get the SW probabilities
            class_df_stage['IF_p'] = p_arr[:, 3] #Get the IF probabilities
            class_df = class_df.append(class_df_stage, ignore_index = True) #Append this day to the full dataframe
            del data, dist_arr, label, probability, p_arr, f_arr, class_df_stage #Delete the data to save memory
        except TypeError: #Throws when date range is empty OR too big
            print('Classification failed for date: '+dates[i]) #Throw a warning
            del data
            continue
    class_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def inter_saver(storename = 'mms_data', class_key = 'mms_class', inter_key = 'stable_intervals', prob_threshold = 0.7, time_threshold = pd.Timedelta('4.6s'), datapath = '../data/'):
    '''
    Saves stable intervals from the classified data.
    storename: Name of the HDF5 file to store the data in
    class_key: Key to load the classified labels from
    inter_key: Key to store the stable intervals under in the HDF5 file
    prob_threshold: Probability threshold for a region to be considered stable
    time_threshold: Time threshold for a region to be considered stable
    '''
    class_df = pd.read_hdf(datapath + storename + '.h5', key = class_key)
    class_df['Epoch'] = pd.to_datetime(class_df['Epoch'], utc = True) #Add timezone information to the time array
    intervals = pd.DataFrame(columns=['start', 'end', 'region']) #Create a dataframe to hold the start and end times of the intervals
    saving = True #Set a flag to indicate that we are saving intervals
    start = class_df['Epoch'][0] #Set the start time to the first time in the dataframe
    for i in range(1, len(class_df)):
        print('Analyzing '+str(i/len(class_df)*100)[0:4]+'% Complete', end='\r') #Print the percentage complete
        if (class_df['region'][i] == class_df['region'][i-1])&(class_df['Epoch'][i] - class_df['Epoch'][i-1] < time_threshold)&(np.max(class_df.loc[i,['MSP_p','MSH_p','SW_p','IF_p']]) > prob_threshold): #Are we in the same region with no outages?
            if saving == False: #Are we not saving intervals?
                start = class_df['Epoch'][i-1] #Set the start time to the previous time
                saving = True #Set the flag to indicate that we are saving intervals
        else: #We are not in the same region or there is an outage
            if saving == True: #Are we saving intervals?
                end = class_df['Epoch'][i-1] #Set the end time to the previous time
                intervals = intervals.append({'start': start, 'end': end, 'region': class_df['region'][i-1]}, ignore_index=True) #Add the interval to the dataframe
                saving = False #Set the flag to indicate that we are not saving intervals
    intervals.to_hdf(datapath + storename + '.h5', key = inter_key, mode = 'a') #Save the intervals to the HDF5 file

def bin_maker(storename = 'mms_data', bin_key = 'bins',inter_key = 'stable_intervals', freq = '100s', datapath = '../data/'):
    '''
    Makes bins from the stable intervals.
    storename: Name of the HDF5 file to store the data in
    inter_key: Key to load the stable intervals from
    freq: Frequency of the bins
    '''
    class_intervals = pd.read_hdf(datapath + storename + '.h5', key=inter_key, mode = 'a')
    #Initialize empty DatetimeIndex
    bins_df = pd.DataFrame([])
    #Loop over all intervals
    for i in range(len(class_intervals)):
        #Create 100s range from start to end of the interval
        bins_stage = pd.date_range(start = class_intervals['start'][i], end = class_intervals['end'][i], freq = freq)
        bins_df_stage = pd.DataFrame([])
        bins_df_stage['start'] = bins_stage[:-1]
        bins_df_stage['end'] = bins_stage[1:]
        bins_df_stage['region'] = np.ones(len(bins_df_stage))*class_intervals['region'][i]
        bins_df = bins_df.append(bins_df_stage, ignore_index = True)
    bins_df.to_hdf(datapath + storename + '.h5', key = bin_key, mode = 'a')

# DATASET DOWNLOADERS
def dis_load(start = '2015-01-01', end = '2023-01-01', load_freq = '6M', storename = 'mms_data', key = 'dis_raw', datapath = '../data/'):
    """
    Loads the DIS data from the CDAS web service and saves it to an HDF5 file
    start: Start date of the data to load (YYYY-MM-DD)
    end: End date of the data to load (YYYY-MM-DD)
    load_freq: Frequency of the data to load (e.g. '6M' for 6 months. DIS data is pretty sparse, so this can be a large value)
    storename: Name of the HDF5 file to save the data to (without the .h5 extension)
    key: Key to save the data to in the HDF5 file
    datapath: Path to the data directory
    """
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    dis_df = pd.DataFrame([])
    for i in range(len(dates)):
        dis_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('MMS1_FPI_FAST_L2_DIS-MOMS', ['mms1_dis_bulkv_gse_fast', 'mms1_dis_numberdensity_fast'], dates[i], dates[i+1])
            dis_df_stage['Epoch_dis'] = data[1]['Epoch']
            dis_df_stage['Vi_xgse'] = data[1]['mms1_dis_bulkv_gse_fast'][:, 0]
            dis_df_stage['Vi_ygse'] = data[1]['mms1_dis_bulkv_gse_fast'][:, 1]
            dis_df_stage['Vi_zgse'] = data[1]['mms1_dis_bulkv_gse_fast'][:, 2]
            dis_df_stage['n_i'] = data[1]['mms1_dis_numberdensity_fast']
            dis_df = dis_df.append(dis_df_stage, ignore_index=True)
        except TypeError: #Throws when date range is empty OR too big
            print('Warning: No data for ' + dates[i] + ' to ' + dates[i+1] + '. Skipping...')
            continue
    dis_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def des_load(start = '2015-01-01', end = '2023-01-01', load_freq = '3M', storename = 'mms_data', key = 'des_raw', datapath = '../data/'):
    """
    Loads the DES data from the CDAS web service and saves it to an HDF5 file
    start: Start date of the data to load (YYYY-MM-DD)
    end: End date of the data to load (YYYY-MM-DD)
    load_freq: Frequency of the data to load (e.g. '6M' for 6 months. DIS data is pretty sparse, so this can be a large value)
    storename: Name of the HDF5 file to save the data to (without the .h5 extension)
    key: Key to save the data to in the HDF5 file
    datapath: Path to the data directory
    """
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    des_df = pd.DataFrame([])
    for i in np.arange(len(dates)-1):
        des_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('MMS1_FPI_FAST_L2_DES-MOMS', ['mms1_des_numberdensity_fast'], dates[i], dates[i+1])
            des_df_stage['Epoch_des'] = data[1]['Epoch']
            des_df_stage['n_e'] = data[1]['mms1_des_numberdensity_fast']
            des_df = des_df.append(des_df_stage, ignore_index=True)
        except TypeError: #Throws when date range is empty OR too big
            print('Warning: No data for ' + dates[i] + ' to ' + dates[i+1] + '. Skipping...')
            continue
    des_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def mec_load(start = '2015-01-01', end = '2023-01-01', load_freq = '3M', storename = 'mms_data', key = 'mec_raw', datapath = '../data/'):
    """
    Loads the MEC data from the CDAS web service and saves it to an HDF5 file
    start: Start date of the data to load (YYYY-MM-DD)
    end: End date of the data to load (YYYY-MM-DD)
    load_freq: Frequency of the data to load (e.g. '3M' for 3 months. MEC data is pretty sparse, so this can be a large value)
    storename: Name of the HDF5 file to save the data to (without the .h5 extension)
    key: Key to save the data to in the HDF5 file
    datapath: Path to the data directory
    """
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    mec_df = pd.DataFrame([])
    for i in np.arange(len(dates)-1):
        mec_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('MMS1_MEC_SRVY_L2_EPHT89D', ['mms1_mec_r_gsm_srvy_l2', 'mms1_mec_r_gse_srvy_l2'], dates[i], dates[i+1])
            mec_df_stage['Epoch_mec'] = data[1]['Epoch']
            mec_df_stage['P_xgsm'] = data[1]['mms1_mec_r_gsm_srvy_l2'][:, 0]
            mec_df_stage['P_ygsm'] = data[1]['mms1_mec_r_gsm_srvy_l2'][:, 1]
            mec_df_stage['P_zgsm'] = data[1]['mms1_mec_r_gsm_srvy_l2'][:, 2]
            mec_df_stage['P_xgse'] = data[1]['mms1_mec_r_gse_srvy_l2'][:, 0]
            mec_df_stage['P_ygse'] = data[1]['mms1_mec_r_gse_srvy_l2'][:, 1]
            mec_df_stage['P_zgse'] = data[1]['mms1_mec_r_gse_srvy_l2'][:, 2]
            mec_df = mec_df.append(mec_df_stage, ignore_index=True)
        except TypeError: #Throws when date range is empty OR too big
            print('Warning: No data for ' + dates[i] + ' to ' + dates[i+1] + '. Skipping...')
            continue
    mec_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def fgm_load(start = '2015-01-01', end = '2023-01-01', load_freq = '1D', bin_freq = '5s', storename = 'mms_data', key = 'fgm_raw', datapath = '../data/', verbose = False):
    '''
    Loads the FGM data from the CDAS database and bins it into 5 second intervals. The data is then saved to an HDF5 file.
    start: The start date of the data to load ('YYYY-MM-DD')
    end: The end date of the data to load ('YYYY-MM-DD')
    load_freq: The frequency of the data to load (e.g. '1D' for 1 day. FGM data is large, so it is recommended to keep this at 1 day)
    bin_freq: The frequency to bin the data to (e.g. '5s' for 5 second bins)
    storename: The name of the HDF5 file to save the data to (without the .h5 extension)
    key: The key to save the data to in the HDF5 file
    datapath: The path to the HDF5 file
    '''
    from cdasws import CdasWs
    import time
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    fgm_df = pd.DataFrame([])
    #Load the FGM data for each day and bin it
    for i in range(len(dates)-1):
        try:
            fgm_data = pd.DataFrame([])
            try:
                data = cdas.get_data('MMS1_FGM_SRVY_L2', ['mms1_fgm_b_gsm_srvy_l2', 'mms1_fgm_b_gse_srvy_l2', 'mms1_fgm_flag_srvy_l2'], dates[i], dates[i+1])
            except: #Sometimes the connection times out, wait a second and see if it works
                time.sleep(1)
                data = cdas.get_data('MMS1_FGM_SRVY_L2', ['mms1_fgm_b_gsm_srvy_l2', 'mms1_fgm_b_gse_srvy_l2', 'mms1_fgm_flag_srvy_l2'], dates[i], dates[i+1])
            fgm_data['Epoch_fgm'] = data[1]['Epoch']
            fgm_data['Bx_gsm'] = data[1]['mms1_fgm_b_gsm_srvy_l2'][:, 0]
            fgm_data['By_gsm'] = data[1]['mms1_fgm_b_gsm_srvy_l2'][:, 1]
            fgm_data['Bz_gsm'] = data[1]['mms1_fgm_b_gsm_srvy_l2'][:, 2]
            fgm_data['Bx_gse'] = data[1]['mms1_fgm_b_gse_srvy_l2'][:, 0]
            fgm_data['By_gse'] = data[1]['mms1_fgm_b_gse_srvy_l2'][:, 1]
            fgm_data['Bz_gse'] = data[1]['mms1_fgm_b_gse_srvy_l2'][:, 2]
            fgm_data['B_flag'] = data[1]['mms1_fgm_flag_srvy_l2']
        except TypeError: #Throws when no FGM data to load
            fgm_data = pd.DataFrame([[np.NaN, np.NaN]], columns = ['Epoch_fgm', 'Bx_gsm'])
            print('Date '+str(i+1)+' of '+str(len(dates))+' has no FGM data')
            continue
        except ValueError: #Throws when FGM data is corrupted
            fgm_data = pd.DataFrame([[np.NaN, np.NaN]], columns = ['Epoch_fgm', 'Bx_gsm'])
            print('Date '+str(i+1)+' of '+str(len(dates))+' has corrupted FGM data')
            continue
        times = pd.date_range(dates[i], dates[i+1], freq=bin_freq) #Five second intervals for the day
        bin_subset = pd.IntervalIndex.from_arrays(times[:-1], times[1:], closed='left') #Get the bins for the current day
        fgm_group = fgm_data.groupby(pd.cut(fgm_data['Epoch_fgm'], bin_subset)) #Bin the FGM data
        fgm_binned_stage = fgm_group.mean() #Get the mean of the binned data
        fgm_binned_stage['count'] = fgm_group.count()['Bx_gsm'] #Get the number of data points in each bin
        fgm_binned_stage['Epoch_fgm'] = bin_subset.left #Add the start of the bins as the Epoch
        fgm_df = fgm_df.append(fgm_binned_stage, ignore_index = True) #Add the binned data to the dataframe
        if verbose: print('Date '+str(i+1)+' of '+str(len(dates)-1)+' ('+dates[i]+') loaded', end='\r')
        del fgm_data #Delete the old data to save memory
        del data #Delete the old data to save memory
    fgm_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def swe_load(start = '2015-01-01', end = '2023-01-01', load_freq = '3M', storename = 'wind_data', key = 'swe_raw', datapath = '../data/'):
    '''
    Loads the SWE data from the CDAS database. The data is then saved to an HDF5 file.
    start: The start date of the data to load ('YYYY-MM-DD')
    end: The end date of the data to load ('YYYY-MM-DD')
    load_freq: The frequency of the data to load (e.g. '3M' for 3 months. SWE data is light so you can load more at once)
    storename: The name of the HDF5 file to save the data to (without the .h5 extension)
    key: The key to save the data to in the HDF5 file
    datapath: The path to the HDF5 file
    '''
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    swe_df = pd.DataFrame([])
    for i in np.arange(len(dates)-1):
        swe_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('WI_K0_SWE', ['Np', 'V_GSE', 'THERMAL_SPD', 'QF_V', 'QF_Np'], dates[i], dates[i+1])
            swe_df_stage['Epoch'] = data[1]['Epoch']
            swe_df_stage['Ni'] = data[1]['Np']
            swe_df_stage['Vi_xgse'] = data[1]['V_GSE'][:, 0]
            swe_df_stage['Vi_ygse'] = data[1]['V_GSE'][:, 1]
            swe_df_stage['Vi_zgse'] = data[1]['V_GSE'][:, 2]
            swe_df_stage['Vth'] = data[1]['THERMAL_SPD']
            swe_df_stage['vflag'] = data[1]['QF_V']
            swe_df_stage['niflag'] = data[1]['QF_Np']
            swe_df = swe_df.append(swe_df_stage, ignore_index=True)
        except TypeError: #Throws when date range is empty OR too big
            print('Warning: No data for ' + dates[i] + ' to ' + dates[i+1] + '. Skipping...')
            continue
    swe_df['Epoch'] = pd.to_datetime(swe_df['Epoch'], utc=True)
    #Remove erroneous Epochs outside downloaded date range (due to CDAS bug)
    swe_df['Epoch'].where(swe_df['Epoch'] >= pd.to_datetime(dates[0], utc=True), np.nan, inplace=True)
    swe_df['Epoch'].where(swe_df['Epoch'] <= pd.to_datetime(dates[-1], utc=True), np.nan, inplace=True)
    #Remove rows with nan Epochs and reset the index
    swe_df.dropna(subset=['Epoch'], inplace=True)
    swe_df.reset_index(drop=True, inplace=True)
    #Set Ni values to nan if they are equal to the fill value of -1e31
    swe_df['Ni'].where(swe_df['Ni'] > -1e30, np.nan, inplace=True)
    #Set Vi values to nan if they are equal to the fill value of -1e31
    swe_df['Vi_xgse'].where(swe_df['Vi_xgse'] > -1e30, np.nan, inplace=True)
    swe_df['Vi_ygse'].where(swe_df['Vi_ygse'] > -1e30, np.nan, inplace=True)
    swe_df['Vi_zgse'].where(swe_df['Vi_zgse'] > -1e30, np.nan, inplace=True)
    #Set Vth values to nan if they are equal to the fill value of -1e31
    swe_df['Vth'].where(swe_df['Vth'] > -1e30, np.nan, inplace=True)
    #Set vflag values to nan if they are equal to the fill value of -2147483648
    swe_df['vflag'].where(swe_df['vflag'] > -2147483648, np.nan, inplace=True)
    #Set niflag values to nan if they are equal to the fill value of -2147483648
    swe_df['niflag'].where(swe_df['niflag'] > -2147483648, np.nan, inplace=True)
    swe_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def mfi_load(start = '2015-01-01', end = '2023-01-01', load_freq = '3M', storename = 'wind_data', key = 'mfi_raw', datapath = '../data/'):
    '''
    Loads the MFI data from the CDAS database. The data is then saved to an HDF5 file.
    start: The start date of the data to load ('YYYY-MM-DD')
    end: The end date of the data to load ('YYYY-MM-DD')
    load_freq: The frequency of the data to load (e.g. '3M' for 3 months. MFI data is light so you can load more at once)
    storename: The name of the HDF5 file to save the data to (without the .h5 extension)
    key: The key to save the data to in the HDF5 file
    datapath: The path to the HDF5 file
    '''
    from cdasws import CdasWs
    cdas = CdasWs() #Initialize CDAS WS Session
    end_delt = pd.to_datetime(end) + pd.Timedelta(load_freq) #Get the last date of the date range to ensure the end is included
    dates = pd.date_range(start, end_delt, freq=load_freq).strftime('%Y-%m-%d %H:%M:%S+0000').tolist() #1 day intervals from 2015 to 2023 (inclusive)
    mfi_df = pd.DataFrame([])
    for i in np.arange(len(dates)-1):
        mfi_df_stage = pd.DataFrame([])
        try:
            data = cdas.get_data('WI_H0_MFI', ['BGSM', 'PGSE'], dates[i], dates[i+1])
            mfi_df_stage['Epoch'] = data[1]['Epoch']
            mfi_df_stage['R_xgse'] = data[1]['PGSE'][:, 0]
            mfi_df_stage['R_ygse'] = data[1]['PGSE'][:, 1]
            mfi_df_stage['R_zgse'] = data[1]['PGSE'][:, 2]
            mfi_df_stage['B_xgsm'] = data[1]['BGSM'][:, 0]
            mfi_df_stage['B_ygsm'] = data[1]['BGSM'][:, 1]
            mfi_df_stage['B_zgsm'] = data[1]['BGSM'][:, 2]
            mfi_df = mfi_df.append(mfi_df_stage, ignore_index=True)
        except TypeError: #Throws when date range is empty OR too big
            print('Error at: ', dates[i], dates[i+1])
            continue
    mfi_df['Epoch'] = pd.to_datetime(mfi_df['Epoch'], utc=True)
    #Set B values to nan if they are equal to the fill value of -1e31
    mfi_df['B_xgsm'].where(mfi_df['B_xgsm'] > -1e30, np.nan, inplace=True)
    mfi_df['B_ygsm'].where(mfi_df['B_ygsm'] > -1e30, np.nan, inplace=True)
    mfi_df['B_zgsm'].where(mfi_df['B_zgsm'] > -1e30, np.nan, inplace=True)
    #Set R values to nan if they are equal to the fill value of -1e31
    mfi_df['R_xgse'].where(mfi_df['R_xgse'] > -1e30, np.nan, inplace=True)
    mfi_df['R_ygse'].where(mfi_df['R_ygse'] > -1e30, np.nan, inplace=True)
    mfi_df['R_zgse'].where(mfi_df['R_zgse'] > -1e30, np.nan, inplace=True)
    mfi_df.to_hdf(datapath + storename + '.h5', key=key, mode = 'a')

def mms_integrator(storename = 'mms_data', key = 'targets', datapath = '../data/', verbose = False):
    '''
    Integrates the MMS data (DIS, DES, MEC, FGM, and classified DIS-DIST) into single dataframe.
    storename: The name of the HDF5 file to save the data to (without the .h5 extension)
    key: The key to save the data to in the HDF5 file
    datapath: The path to the HDF5 file
    verbose: Whether to print out the progress of the integration
    '''
    #Make a combined dataframe with all the data
    dis_df = pd.read_hdf(datapath + storename + '.h5', key='dis_raw', mode = 'a')
    des_df = pd.read_hdf(datapath + storename + '.h5', key='des_raw', mode = 'a')
    mec_df = pd.read_hdf(datapath + storename + '.h5', key='mec_raw', mode = 'a')
    fgm_df = pd.read_hdf(datapath + storename + '.h5', key='fgm_raw', mode = 'a')
    bins = pd.read_hdf(datapath + storename + '.h5', key = 'bins', mode = 'a') #Bins for binning the data
    bins_index = pd.IntervalIndex.from_arrays(bins['start'], bins['end'], closed='left') #Make interval index for binning
    #Group the mec, des, and dis data by the bins
    mec_group = mec_df.groupby(pd.cut(mec_df['Epoch_mec'], bins_index))
    if verbose: print('Finished grouping mec')
    des_group = des_df.groupby(pd.cut(des_df['Epoch_des'], bins_index))
    if verbose: print('Finished grouping des')
    dis_group = dis_df.groupby(pd.cut(dis_df['Epoch_dis'], bins_index))
    if verbose: print('Finished grouping dis')
    fgm_group = fgm_df.groupby(pd.cut(fgm_df['Epoch_fgm'], bins_index))
    if verbose: print('Finished grouping fgm')
    #Bin the mec, des, and dis data into staging dataframes
    mec_binned = mec_group.mean()
    des_binned = des_group.mean()
    dis_binned = dis_group.mean()
    fgm_binned = fgm_group.mean()
    if verbose: print('Finished binning data')
    #Add the counts in each bin
    mec_binned['count_mec'] = mec_group.count()['R_xgsm']
    des_binned['count_des'] = des_group.count()['Ne']
    dis_binned['count_dis'] = dis_group.count()['Ni']
    fgm_binned['count_fgm'] = fgm_group.count()['Bx_gsm']
    if verbose: print('Finished adding counts')
    #Get the first Epoch in each bin
    mec_binned['Epoch_mec'] = mec_group.first()['Epoch_mec']
    des_binned['Epoch_des'] = des_group.first()['Epoch_des']
    dis_binned['Epoch_dis'] = dis_group.first()['Epoch_dis']
    fgm_binned['Epoch_fgm'] = fgm_group.first()['Epoch_fgm']
    if verbose: print('Finished adding Epochs')
    #Set the index to numbers instead of the bins
    mec_binned = mec_binned.reset_index(drop = True)
    des_binned = des_binned.reset_index(drop = True)
    dis_binned = dis_binned.reset_index(drop = True)
    fgm_binned = fgm_binned.reset_index(drop = True)
    if verbose: print('Finished resetting index')
    #Add the region column
    mec_binned['region'] = bins['region'].to_numpy()
    des_binned['region'] = bins['region'].to_numpy()
    dis_binned['region'] = bins['region'].to_numpy()
    fgm_binned['region'] = bins['region'].to_numpy()
    if verbose: print('Finished adding region column')
    #Combine the three dataframes
    mms_data = pd.merge(mec_binned.drop(columns = 'region'), des_binned.drop(columns = 'region'), left_index = True, right_index = True)
    mms_data = pd.merge(mms_data, dis_binned.drop(columns = 'region'), left_index = True, right_index = True)
    mms_data = pd.merge(mms_data, fgm_binned.drop(columns = 'region'), left_index = True, right_index = True)
    mms_data['Epoch'] = bins['start'] #Add the start of the bins as the Epoch
    mms_data['region'] = bins['region'] #Add the region column
    #Drop any rows with NaNs
    mms_data = mms_data.dropna()
    mms_data = mms_data.reset_index(drop = True) #Reset the index
    mms_data[['R_xgsm', 'R_ygsm', 'R_zgsm', 'R_xgse', 'R_ygse', 'R_zgse']] = mms_data[['R_xgsm', 'R_ygsm', 'R_zgsm', 'R_xgse', 'R_ygse', 'R_zgse']]/6378 #Convert the R vectors to be in units of Earth radii
    #Check if MMS data was obtained with SW energy-azimuth table
    mms_data['Vi_ygse'][mms_data['SW_table']==True] += -10.7 #Subtract 10.7 km/s to Vy_gse if MMS data was obtained with SW energy-azimuth table
    mms_data['Vi_ygse'][mms_data['SW_table']==True] += -29.0 #Subtract 29.0 km/s to Vy_gse if MMS data was not obtained with SW energy-azimuth table
    mms_data.to_hdf(datapath + storename + '.h5', key = key, mode = 'a') #Save the data

def wind_integrator(storename = 'wind_data', key = 'inputs', datapath = '../data/', verbose = False, mms_storename = 'mms_data'):
    '''
    Integrates the Wind data (SWE and MFI) into single dataframe.
    storename: The name of the HDF5 file to save the data to (without the .h5 extension)
    key: The key to save the data to in the HDF5 file
    datapath: The path to the HDF5 file
    verbose: Whether to print out the progress of the integration
    mms_storename: The name of the HDF5 file containing the MMS data
    '''
    swe_df = pd.read_hdf(datapath + storename + '.h5', key='swe_raw')
    mfi_df = pd.read_hdf(datapath + storename + '.h5', key='mfi_raw')
    mec_data = pd.read_hdf(datapath + mms_storename + '.h5', key='mec_raw', mode = 'a')

    #Drop rows with dips in the Epoch
    swe_df_new = swe_df.copy()
    for i in np.arange(1,len(swe_df)):  #First we hunt dips in the data
        delt = (swe_df['Epoch'][i] - swe_df['Epoch'][i-1]).total_seconds()
        if (delt < 0): #If the time difference is negative, we have a dip!
            swe_df_new.drop(index=i, inplace=True) #Drop the row
        if verbose: print('Dip Loop ' + str(100*i/len(swe_df))[0:5] + '% complete', end='\r')

    max_allowable_delt = 100 #Maximum allowable time difference between consecutive entries
    swe_df_new.reset_index(drop=True, inplace=True) #Reset the index
    for i in np.arange(1,len(swe_df_new)):  #Then we hunt for large time differences
        delt = (swe_df_new['Epoch'][i] - swe_df_new['Epoch'][i-1]).total_seconds()
        if (delt > max_allowable_delt*2): #If the time difference is too large, we have a gap!
            rows_stage = pd.DataFrame([]) #Initialize the staging dataframe
            rows_stage['Epoch'] = pd.date_range(swe_df_new['Epoch'][i-1], swe_df_new['Epoch'][i], freq='100s')[1:] #Get the new Epochs
            swe_df_new = swe_df_new.append(rows_stage, ignore_index=True) #Append the rows
        if verbose: print('Time Interpolation Loop ' + str(100*i/len(swe_df_new))[0:5] + '% complete', end='\r')
    swe_df_new['interp_flag'] = swe_df_new.isna().any(axis=1) #Get the rows with NaNs and flag them as interpolated
    swe_df_new = swe_df_new.sort_values(by=['Epoch']) #Sort the dataframe by Epoch
    swe_df_new = swe_df_new.reset_index(drop=True) #Reset the index

    #Find elements in mfi_df closest to each Epoch in swe_df
    #The following code is closest_argmin in ks_custom, reproduced here so we don't have to import the whole module
    A = swe_df_new['Epoch'].to_numpy()
    B = mfi_df['Epoch'].to_numpy()
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    args = sidx_B[sorted_idx-mask]
    mfi_df_stage = mfi_df.iloc[args] #Get the closest elements
    mfi_df_stage.index = swe_df_new.index #Reset the index to be sequential
    swe_df_new = pd.merge(swe_df_new, mfi_df_stage, left_index=True, right_index=True) #Merge the dataframes
    swe_df_new.rename(columns={'Epoch_x': 'Epoch', 'Epoch_y': 'Epoch_mfi'}, inplace=True) #Rename the Epoch columns
    swe_df_new['flag'] = swe_df_new.isna().any(axis=1) #Get the rows with NaNs and flag them as interpolated

    #Interpolate the NaNs
    swe_df_new['Ni'] = swe_df_new['Ni'].interpolate(method='linear', axis=0)
    swe_df_new['Vi_xgse'] = swe_df_new['Vi_xgse'].interpolate(method='linear', axis=0)
    swe_df_new['Vi_ygse'] = swe_df_new['Vi_ygse'].interpolate(method='linear', axis=0)
    swe_df_new['Vi_zgse'] = swe_df_new['Vi_zgse'].interpolate(method='linear', axis=0)
    swe_df_new['Vth'] = swe_df_new['Vth'].interpolate(method='linear', axis=0)
    swe_df_new['vflag'] = swe_df_new['vflag'].interpolate(method='linear', axis=0)
    swe_df_new['niflag'] = swe_df_new['niflag'].interpolate(method='linear', axis=0)
    swe_df_new['R_xgse'] = swe_df_new['R_xgse'].interpolate(method='linear', axis=0)
    swe_df_new['R_ygse'] = swe_df_new['R_ygse'].interpolate(method='linear', axis=0)
    swe_df_new['R_zgse'] = swe_df_new['R_zgse'].interpolate(method='linear', axis=0)
    swe_df_new['B_xgsm'] = swe_df_new['B_xgsm'].interpolate(method='linear', axis=0)
    swe_df_new['B_ygsm'] = swe_df_new['B_ygsm'].interpolate(method='linear', axis=0)
    swe_df_new['B_zgsm'] = swe_df_new['B_zgsm'].interpolate(method='linear', axis=0)

    swe_df_new['Epoch_delta'] = swe_df_new['Epoch'] - swe_df_new['Epoch_mfi'] #Get the difference between the Epochs
    mec_data.rename(columns={'R_xgse' : 'target_R_xgse', 'R_ygse' : 'target_R_ygse', 'R_zgse' : 'target_R_zgse'}, inplace=True)
    mec_data.rename(columns={'R_xgsm' : 'target_R_xgsm', 'R_ygsm' : 'target_R_ygsm', 'R_zgsm' : 'target_R_zgsm'}, inplace=True)
    mec_data[['target_R_xgse', 'target_R_ygse', 'target_R_zgse']] = mec_data[['target_R_xgse', 'target_R_ygse', 'target_R_zgse']]/6378 #Convert the R vectors to be in units of Earth radii
    mec_data[['target_R_xgsm', 'target_R_ygsm', 'target_R_zgsm']] = mec_data[['target_R_xgsm', 'target_R_ygsm', 'target_R_zgsm']]/6378 #Convert the R vectors to be in units of Earth radii
    #Find elements in mec_data closest to each Epoch in swe_df_new
    #The following code is closest_argmin in ks_custom, reproduced here so we don't have to import the whole module
    A = swe_df_new['Epoch'].to_numpy()
    B = mec_data['Epoch_mec'].to_numpy()
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    args = sidx_B[sorted_idx-mask]
    mec_data_stage = mec_data.iloc[args] #Get the closest elements
    mec_data_stage.index = swe_df_new.index #Reset the index to be sequential
    swe_df_new = pd.merge(swe_df_new, mec_data_stage, left_index=True, right_index=True) #Merge the dataframes
    swe_df_new.to_hdf(datapath + storename + '.h5', key=key, mode = 'a') #Save the combined data

        
# OMNI LOADING/CONSTRUCTING     

def wind_ascii_prep():
    '''
    Silly little function that assembles wind-only OMNI dataset from downloaded ASCII files
    '''
    wind_2015 = pd.read_csv('../data/omni_data/wind_ascii/2015wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2016 = pd.read_csv('../data/omni_data/wind_ascii/2016wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2017 = pd.read_csv('../data/omni_data/wind_ascii/2017wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2018 = pd.read_csv('../data/omni_data/wind_ascii/2018wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2019 = pd.read_csv('../data/omni_data/wind_ascii/2019wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2020 = pd.read_csv('../data/omni_data/wind_ascii/2020wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2021 = pd.read_csv('../data/omni_data/wind_ascii/2021wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_2022 = pd.read_csv('../data/omni_data/wind_ascii/2022wind.txt', delim_whitespace=True, names = sw.WIND_ASCII_COLS)
    wind_full = wind_2015.append([wind_2016, wind_2017, wind_2018, wind_2019, wind_2020, wind_2021, wind_2022], ignore_index=True)
    wind_full = wind_full.drop_duplicates()
    wind_full.index = np.arange(len(wind_full)) #reindex
    date = []
    for i in np.arange(len(wind_full)):
        date.append(pd.to_datetime(str(int(wind_full['year'][i]))+'-'+str(int(wind_full['doy'][i]))+'-'+str(int(wind_full['hour'][i]))+'-'+str(int(wind_full['minute'][i])), format = '%Y-%j-%H-%M', utc=True))
    wind_full['Epoch'] = date
    wind_full.to_hdf('../data/wind_data.h5', key='wind_omni_bs', mode = 'a')

def omni_shift(wind_store = 'wind_data', wind_key = 'wind_omni_bs', mms_store = 'mms_data', mms_key = 'targets', key = 'wind_omni_shift', datapath = '../data/'):
    '''
    Shifts Wind-specific OMNI data to MMS's position along phase fronts (i.e. the "proper" way).
    wind_store: The name of the HDF5 file to load the wind data from (without the .h5 extension)
    wind_key: The key to load the wind data from in the HDF5 file
    mms_store: The name of the HDF5 file to load the MMS data from (without the .h5 extension)
    mms_key: The key to load the MMS data from in the HDF5 file
    key: The key to save the shifted data to in the HDF5 file
    datapath: The path to the HDF5 file
    '''
    mms_data = pd.read_hdf(datapath + mms_store + '.h5', key= mms_key, mode = 'a')
    wind_data = pd.read_hdf(datapath + wind_store + '.h5', key= wind_key, mode = 'a')

    omni_ind = np.zeros(len(mms_data))
    shift = np.zeros(len(mms_data))

    omni_time = (wind_data['Epoch'] - pd.Timestamp("1970-01-01 00:00:00+0000")) // pd.Timedelta('1s') #Convert to seconds since epoch

    nx = wind_data['Phase_n_x'] #Phase front normal vector
    ny = wind_data['Phase_n_y']
    nz = wind_data['Phase_n_z']

    Vx = wind_data['VX_GSE'] #Solar wind velocity
    Vy = wind_data['VY_GSE']
    Vz = wind_data['VZ_GSE']

    denominator = (nx * Vx) + (ny * Vy) + (nz * Vz) #Denominator of time delay function

    Rox = wind_data['BSN_X'] * sw.EARTH_RADIUS #Wind position (Convert from Re to km)
    Roy = wind_data['BSN_Y'] * sw.EARTH_RADIUS
    Roz = wind_data['BSN_Z'] * sw.EARTH_RADIUS

    for i in np.arange(len(mms_data)):
        mms_time = (mms_data['Epoch'][i]- pd.Timestamp("1970-01-01 00:00:00+0000")) // pd.Timedelta('1s') #Convert to seconds since epoch
        cut_bool = wind_data['Epoch'].between(mms_data['Epoch'][i] - pd.Timedelta('1D'), mms_data['Epoch'][i]) #Boolean array of omni data within 1 day on either side of MMS data
        min_ind = cut_bool[cut_bool].index.min() #Find the lowest index so we can store the right one later

        Rdx = mms_data['R_xgse'][i] * sw.EARTH_RADIUS #MMS position (Convert from Re to km)
        Rdy = mms_data['R_ygse'][i] * sw.EARTH_RADIUS
        Rdz = mms_data['R_zgse'][i] * sw.EARTH_RADIUS

        delta_t = (nx[cut_bool] * (Rdx - Rox[cut_bool]) +  ny[cut_bool] * (Rdy - Roy[cut_bool]) + nz[cut_bool] * (Rdz - Roz[cut_bool])) / denominator[cut_bool] #OMNI algorithm time delay function
        try:
            omni_ind[i] = np.argmin(np.abs(omni_time[cut_bool] + delta_t - mms_time)) + min_ind #which piece of omni_data corresponds to this adjusted time?
            shift[i] = np.min(np.abs(omni_time[cut_bool] + delta_t - mms_time))
        except ValueError: #Throws when there's no data in omni_data within 1 day on either side of MMS data
            omni_ind[i] = 0 #Unfortunately nan values don't work with pandas indexing. Look for nan timeshifts to drop data later
            shift[i] = np.nan #Unfortunately nan values don't work with pandas indexing. Look for nan timeshifts to drop data later
        print('OMNI Propagation '+str(100*i/len(mms_data))[0:5]+'% Complete', end='\r')
    omni_data_shift = wind_data.loc[omni_ind]
    omni_data_shift['Timeshift_mms'] = shift
    omni_data_shift.reset_index(drop=True, inplace=True)
    omni_data_shift = omni_fillval(omni_data_shift) #Replace fill values with nans
    omni_data_shift.to_hdf(datapath + wind_store + '.h5', key = key)

def omni_fillval(omni_data, B_fill = 9999.99, V_fill = 99999.9, n_fill = 999.99, T_fill = 9999999, P_fill = 9999.99):
    '''
    Function that replaces OMNI missing data fill values with nans
    omni_data: The OMNI data to replace fill values in (as a pandas dataframe, see omni_shift()/wind_ascii_prep() for examples)
    B_fill: The fill value for magnetic field
    V_fill: The fill value for solar wind velocity
    n_fill: The fill value for number density
    T_fill: The fill value for temperature
    '''
    omni_data['BX_GSE'].where(omni_data['BX_GSE'] != B_fill, np.nan, inplace=True)
    omni_data['BY_GSM'].where(omni_data['BY_GSM'] != B_fill, np.nan, inplace=True)
    omni_data['BZ_GSM'].where(omni_data['BZ_GSM'] != B_fill, np.nan, inplace=True)
    omni_data['flow_speed'].where(omni_data['flow_speed'] != V_fill, np.nan, inplace=True)
    omni_data['VX_GSE'].where(omni_data['VX_GSE'] != V_fill, np.nan, inplace=True)
    omni_data['VY_GSE'].where(omni_data['VY_GSE'] != V_fill, np.nan, inplace=True)
    omni_data['VZ_GSE'].where(omni_data['VZ_GSE'] != V_fill, np.nan, inplace=True)
    omni_data['proton_density'].where(omni_data['proton_density'] != n_fill, np.nan, inplace=True)
    omni_data['T'].where(omni_data['T'] != T_fill, np.nan, inplace=True)
    omni_data['SC_X'].where(omni_data['SC_X'] != P_fill, np.nan, inplace=True)
    omni_data['SC_Y'].where(omni_data['SC_Y'] != P_fill, np.nan, inplace=True)
    omni_data['SC_Z'].where(omni_data['SC_Z'] != P_fill, np.nan, inplace=True)
    omni_data.loc[omni_data['Timeshift_mms'], ['year', 'doy', 'hour', 'minute', 'IMF_PTS', 'percent_interp', 'CPMV', 'Timeshift', 'Phase_n_x', 'Phase_n_y', 'Phase_n_z', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'RMS_Timeshift', 'RMS_phase', 'flow_speed', 'VX_GSE', 'VY_GSE', 'VZ_GSE', 'proton_density', 'T', 'SC_X', 'SC_Y', 'SC_Z', 'BSN_X', 'BSN_Y', 'BSN_Z', 'RMS_target']
    ] = np.nan #For timeshift failures, replace rows with nans
    return omni_data

# MMS PLOTS
        
def nn_omni_comp_plot(trange, mms_data, predict, omni_data, savename, title = 'MMS', save = False, datapath = './', filetype = 'png'):
    '''
    Makes a quick plot of classified MMS data alongside neural netrwork and OMNI predictions thereof.
    
    '''
    left_time = trange[0].timestamp()
    right_time = trange[1].timestamp()

    cla_df = sw.class_query(trange[0].timestamp(), trange[1].timestamp(), datapath = datapath)

    t_arr = mms_data['time'].to_numpy()
    v_arr = mms_data.loc[:, ['Vx_gse', 'Vy_gse', 'Vz_gse']].to_numpy()
    ni_arr = mms_data['n_e'].to_numpy()
    ol_p_arr = cla_df.loc[:, ['MSP_p','MSH_p','SW_p','IF_p']].to_numpy()
    ct_arr = cla_df['time'].to_numpy()
    ol_f_arr = cla_df['region'].to_numpy()
    p_arr = mms_data.loc[:, ['Px_gse', 'Py_gse', 'Pz_gse']].to_numpy()/sw.EARTH_RADIUS #Convert from km to RE
    pt_arr = mms_data['time'].to_numpy()
    bt_arr = mms_data['time'].to_numpy()
    b_arr = mms_data.loc[:, ['Bx_gsm', 'By_gsm', 'Bz_gsm']].to_numpy()
    
    i_cut = (t_arr >= left_time) & (t_arr <= right_time)
    c_cut = (ct_arr >= left_time) & (ct_arr <= right_time)
    p_cut = (predict['time'] >= left_time) & (predict['time'] <= right_time) #Cuts all data but those in the left and right bounds
    o_cut = (omni_data['time'] > left_time) & (omni_data['time'] < right_time) #Cuts all data but those in the left and right bounds
    
    if (len(t_arr[i_cut]) == 0): #If there's no data in the range to comparte to, hit da bricks
        print('empty data array, no plot')
        return

    fig, ax = plt.subplots(nrows=5,ncols=1,sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.set_size_inches(8.5, 11)
    
    quart_delt = (right_time-left_time)/4.0
    ticks = [left_time,left_time+quart_delt,left_time+2*quart_delt,left_time+3*quart_delt,right_time]
    X_GSE_ticks = np.interp(ticks, pt_arr, p_arr[:, 0])
    Y_GSE_ticks = np.interp(ticks, pt_arr, p_arr[:, 1])
    Z_GSE_ticks = np.interp(ticks, pt_arr, p_arr[:, 2])
    labels = [dt.datetime.fromtimestamp(left_time,tz=pytz.utc).strftime('%H:%M')+'\n'+str(X_GSE_ticks[0])[:5]+'\n'+str(Y_GSE_ticks[0])[:5]+'\n'+str(Z_GSE_ticks[0])[:5],
              dt.datetime.fromtimestamp(left_time+quart_delt,tz=pytz.utc).strftime('%H:%M')+'\n'+str(X_GSE_ticks[1])[:5]+'\n'+str(Y_GSE_ticks[1])[:5]+'\n'+str(Z_GSE_ticks[1])[:5],
              dt.datetime.fromtimestamp(left_time+2*quart_delt,tz=pytz.utc).strftime('%H:%M')+'\n'+str(X_GSE_ticks[2])[:5]+'\n'+str(Y_GSE_ticks[2])[:5]+'\n'+str(Z_GSE_ticks[2])[:5],
              dt.datetime.fromtimestamp(left_time+3*quart_delt,tz=pytz.utc).strftime('%H:%M')+'\n'+str(X_GSE_ticks[3])[:5]+'\n'+str(Y_GSE_ticks[3])[:5]+'\n'+str(Z_GSE_ticks[3])[:5],
              dt.datetime.fromtimestamp(right_time,tz=pytz.utc).strftime('%H:%M')+'\n'+str(X_GSE_ticks[4])[:5]+'\n'+str(Y_GSE_ticks[4])[:5]+'\n'+str(Z_GSE_ticks[4])[:5]]


    ax[0].plot(bt_arr[i_cut], b_arr[i_cut,0],color=sw.r)
    ax[0].plot(bt_arr[i_cut], b_arr[i_cut,1],color=sw.g)
    ax[0].plot(bt_arr[i_cut], b_arr[i_cut,2],color=sw.b)
    ax[0].plot(predict['time'][p_cut], predict['Bx_gsm'][p_cut], color=sw.r, alpha=0.75, linestyle = '--')
    ax[0].fill_between(predict['time'][p_cut], predict['Bx_gsm'][p_cut] + predict['Bx_gsm_sig'][p_cut], predict['Bx_gsm'][p_cut] - predict['Bx_gsm_sig'][p_cut], color=sw.r, alpha=0.2)
    ax[0].plot(predict['time'][p_cut], predict['By_gsm'][p_cut], color=sw.g, alpha=0.75, linestyle = '--')
    ax[0].fill_between(predict['time'][p_cut], predict['By_gsm'][p_cut] + predict['By_gsm_sig'][p_cut], predict['By_gsm'][p_cut] - predict['By_gsm_sig'][p_cut], color=sw.g, alpha=0.2)
    ax[0].plot(predict['time'][p_cut], predict['Bz_gsm'][p_cut], color=sw.b, alpha=0.75, linestyle = '--')
    ax[0].fill_between(predict['time'][p_cut], predict['Bz_gsm'][p_cut] + predict['Bz_gsm_sig'][p_cut], predict['Bz_gsm'][p_cut] - predict['Bz_gsm_sig'][p_cut], color=sw.b, alpha=0.2)
    ax[0].plot(omni_data['time'][o_cut], omni_data['Bx_gsm'][o_cut], color=sw.r, alpha=1, linestyle = ':')
    ax[0].plot(omni_data['time'][o_cut], omni_data['By_gsm'][o_cut], color=sw.g, alpha=1, linestyle = ':')
    ax[0].plot(omni_data['time'][o_cut], omni_data['Bz_gsm'][o_cut], color=sw.b, alpha=1, linestyle = ':')
    ax[0].set_ylabel(r'$B_{GSM}$' 
                      '\n(nT)')
    ax[0].legend(labels  = ['X', 'Y', 'Z'],loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)
    ax[0].set_title(title + ' ('+trange[0].strftime('%Y-%m-%d')+')')


    ax[1].plot(t_arr[i_cut], v_arr[i_cut,1],color=sw.g)
    ax[1].plot(t_arr[i_cut], v_arr[i_cut,2],color=sw.b)
    ax[1].plot(predict['time'][p_cut], predict['Vy_gse'][p_cut], color=sw.g, alpha=0.75, linestyle = '--')
    ax[1].fill_between(predict['time'][p_cut], predict['Vy_gse'][p_cut] + predict['Vy_gse_sig'][p_cut], predict['Vy_gse'][p_cut] - predict['Vy_gse_sig'][p_cut], color=sw.g, alpha=0.2)
    ax[1].plot(predict['time'][p_cut], predict['Vz_gse'][p_cut], color=sw.b, alpha=0.75, linestyle = '--')
    ax[1].fill_between(predict['time'][p_cut], predict['Vz_gse'][p_cut] + predict['Vz_gse_sig'][p_cut], predict['Vz_gse'][p_cut] - predict['Vz_gse_sig'][p_cut], color=sw.b, alpha=0.2)
    ax[1].plot(omni_data['time'][o_cut], omni_data['Vy_gse'][o_cut], color=sw.g, alpha=1, linestyle = ':')
    ax[1].plot(omni_data['time'][o_cut], omni_data['Vz_gse'][o_cut], color=sw.b, alpha=1, linestyle = ':')
    ax[1].set_ylabel(r'$v_{GSM}$' 
                     '\n(km/s)')
    ax[1].legend(labels  = ['Y', 'Z'],loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)
    
    ax[2].plot(t_arr[i_cut], v_arr[i_cut,0],color=sw.r)
    ax[2].plot(predict['time'][p_cut], predict['Vx_gse'][p_cut], color=sw.r, alpha=0.75, linestyle = '--')
    ax[2].fill_between(predict['time'][p_cut], predict['Vx_gse'][p_cut] + predict['Vx_gse_sig'][p_cut], predict['Vx_gse'][p_cut] - predict['Vx_gse_sig'][p_cut], color=sw.r, alpha=0.2)
    ax[2].plot(omni_data['time'][o_cut], omni_data['Vx_gse'][o_cut], color=sw.r, alpha=1, linestyle = ':')
    ax[2].set_ylabel(r'$v_{GSM}$' 
                     '\n(km/s)')
    ax[2].legend(labels  = ['X'],loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)
    
    ax[3].plot(t_arr[i_cut], ni_arr[i_cut],color=sw.b)
    ax[3].plot(predict['time'][p_cut], predict['n_e'][p_cut], color=sw.b, alpha=0.75, linestyle = '--')
    ax[3].fill_between(predict['time'][p_cut], predict['n_e'][p_cut] + predict['n_e_sig'][p_cut], predict['n_e'][p_cut] - predict['n_e_sig'][p_cut], color=sw.b, alpha=0.2)
    ax[3].plot(omni_data['time'][o_cut], omni_data['n_e'][o_cut], color=sw.b, alpha=1, linestyle = ':')
    ax[3].set_ylabel(r'$n_{i/e}$'
                     '\n'
                     r'$(cm^{-3})$')

    ax[4].bar(ct_arr[c_cut], ol_p_arr[c_cut, 0], 4.5, label = 'MSP', color = sw.b)
    ax[4].bar(ct_arr[c_cut], ol_p_arr[c_cut, 1], 4.5, bottom = ol_p_arr[c_cut, 0], label = 'MSH', color = sw.g)
    ax[4].bar(ct_arr[c_cut], ol_p_arr[c_cut, 2], 4.5, bottom = ol_p_arr[c_cut, 0] + ol_p_arr[c_cut, 1], label = 'SW', color = sw.r)
    ax[4].bar(ct_arr[c_cut], ol_p_arr[c_cut, 3], 4.5, bottom = ol_p_arr[c_cut, 0] + ol_p_arr[c_cut, 1] + ol_p_arr[c_cut, 2], label = 'IF', color = 'grey')
    ax[4].set_ylim(0,1)
    ax[4].set_ylabel(r'p')
    ax[4].legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon = False)

    ax[4].set_xlim(left_time,right_time)
    ax[4].set_xticks(ticks)
    ax[4].set_xticklabels(labels)

    ax[4].text(-0.2, -0.5, dt.datetime.fromtimestamp(left_time,tz=pytz.utc).strftime('%Y-%m-%d') + '\nGSE X\nGSE Y\nGSE Z', transform = ax[4].transAxes)
    if save:
        plt.savefig(savename+'.'+filetype, transparent = False, bbox_inches = 'tight')
        plt.close()
    else:
        plt.show()
        
        
if __name__ == '__main__':
    start_date = pd.to_datetime('2015-09-02')
    end_date = pd.to_datetime('2023-04-23')
    dis_load(start = start_date, end = end_date)
    des_load(start = start_date, end = end_date)
    fgm_load(start = start_date, end = end_date)
    mec_load(start = start_date, end = end_date)
    class_load(start = start_date, end = end_date)
    inter_saver()
    bin_maker()
    mms_integrator()