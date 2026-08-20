import numpy as np
import pandas as pd
import cdflib
import os
import sys
import primesw as psw # NOTE: This is the version on PyPI! This is to ensure the CDF content and the locally-run version have identical outputs
from spacepy import pycdf

sys.path.append("/glade/u/home/cobrien/prime/prime_lib/configs") # Enable the direct import of the cdfdicts module with the CDF metadata
import cdfdicts as cdfd

### Parameters TODO: Make these command line arguments
interval = [
    pd.to_datetime('1995-01-01 00:00:00+0000'),
    pd.to_datetime('2026-08-18 00:00:00+0000')
]
savedir = os.path.abspath('/glade/derecho/scratch/cobrien/sc_data/') # Location to create the PRIME CDF directory
overwrite = True # If True, deletes extant files with same names
max_retries = 3 # Sometimes we get rate limited, so try to load each range a few times
###

# Define the CDF filling and writing functions

# #TODO: Change to cdflib because it's better, here's a first pass:
# def fillutil(cdf_writer, varname, attdict, data):
#     '''
#     Function that fills a CDF variable with data and metadata.
#     '''
#     var_spec = {
#         'Vaiable': varname,
#         'Data_Type': None,
#         'Num_Elements': len(data),
#         'Rec_Vary': None,
#     }
#     cdf_writer.write_var(var_spec, var_attrs = attdict, var_data = data)
#     valid_minmax = {} # Store the valid min and max (needed for plotting reasons I guess)
#     if (np.min(data)>=np.max(data)): #If the data is constant/length 1, we need to set VALIDMIN and VALIDMAX to almost the same value
#         valid_minmax['VALIDMIN'] = np.min(data) 
#         try:
#             valid_minmax['VALIDMAX'] = np.min(data)+0.1
#         except TypeError: #Throws when data is a datetime object
#             valid_minmax['VALIDMAX'] = np.min(data)+pd.Timedelta(seconds=1)
#     else:
#         valid_minmax['VALIDMIN'] = np.min(data)
#         valid_minmax['VALIDMAX'] = np.max(data)
#     cdf_writer.write_variableattrs(valid_minmax) # Add VALIDMIN and VALIDMAX to CDF

def fillutil(cdf_struct, varname, attdict, data):
    '''
    Function that fills a CDF variable with data and metadata.
    '''
    cdf_struct[varname] = data
    cdf_struct[varname].attrs = attdict
    if (np.min(data)>=np.max(data)): #If the data is constant/length 1, we need to set VALIDMIN and VALIDMAX to almost the same value
        cdf_struct[varname].attrs['VALIDMIN'] = np.min(data) 
        try:
            cdf_struct[varname].attrs['VALIDMAX'] = np.min(data)+0.1
        except TypeError: #Throws when data is a datetime object
            cdf_struct[varname].attrs['VALIDMAX'] = np.min(data)+pd.Timedelta(seconds=1)
    else:
        cdf_struct[varname].attrs['VALIDMIN'] = np.min(data)
        cdf_struct[varname].attrs['VALIDMAX'] = np.max(data)
        
def labelutil(cdf_struct, varname, attdict, labels):
    '''
    Function that fills a CDF variable with labels and metadata.
    '''
    cdf_struct[varname] = labels
    cdf_struct[varname].attrs = attdict

def cdfw(data,filename):
    """
    Function that takes dataframe of PRIME outputs and saves as CDF with correct metadata.

    :param [float, array-like] data: Dataframe of PRIME outputs.
    :param [str] filename: String filename to save the CDF as.

    :returns: [bool] write: Bool of whether file is written.
    
    """
    if (data.shape[0] == 0): #Were we passed an empty array?
        return False
    cdf = pycdf.CDF(filename, create=True)
    cdf.attrs = cdfd.primebsn_att_dict
    cdf.attrs['Logical_file_id'] = filename
    fillutil(cdf,'Epoch',cdfd.epoch_primebsn_att,data['Epoch'].to_numpy())
    fillutil(cdf,'B_GSE',cdfd.bgse_primebsn_att,data.loc[:, ['mms1_fgm_b_gse_srvy_l2_0', 'mms1_fgm_b_gse_srvy_l2_1', 'mms1_fgm_b_gse_srvy_l2_2']].to_numpy())
    fillutil(cdf,'B_GSE_sig',cdfd.bgsesig_primebsn_att,data.loc[:, ['mms1_fgm_b_gse_srvy_l2_0_std', 'mms1_fgm_b_gse_srvy_l2_1_std', 'mms1_fgm_b_gse_srvy_l2_2_std']].to_numpy())
    fillutil(cdf,'V_GSE',cdfd.vgse_primebsn_att,data.loc[:, ['mms1_dis_bulkv_gse_fast_0', 'mms1_dis_bulkv_gse_fast_1', 'mms1_dis_bulkv_gse_fast_2']].to_numpy())
    fillutil(cdf,'V_GSE_sig',cdfd.vgsesig_primebsn_att,data.loc[:, ['mms1_dis_bulkv_gse_fast_0_std', 'mms1_dis_bulkv_gse_fast_1_std', 'mms1_dis_bulkv_gse_fast_2_std']].to_numpy())
    fillutil(cdf,'Ne',cdfd.n_primebsn_att,data['mms1_des_numberdensity_fast'].to_numpy())
    fillutil(cdf,'Ne_sig',cdfd.nsig_primebsn_att,data['mms1_des_numberdensity_fast_std'].to_numpy())
    labelutil(cdf,'B_GSE_label',cdfd.bgse_primebsn_label,['Bx GSE','By GSE','Bz GSE'])
    labelutil(cdf,'B_GSE_sig_label',cdfd.bgsesig_primebsn_label,['Bx GSE Sigma','By GSE Sigma','Bz GSE Sigma'])
    labelutil(cdf,'V_GSE_label',cdfd.vgse_primebsn_label,['Vx GSE','Vy GSE','Vz GSE'])
    labelutil(cdf,'V_GSE_sig_label',cdfd.vgsesig_primebsn_label,['Vx GSE Sigma','Vy GSE Sigma','Vz GSE Sigma'])
    labelutil(cdf,'Ne_label',cdfd.n_primebsn_label,['Ne'])
    labelutil(cdf,'Ne_sig_label',cdfd.nsig_label,['Ne Sigma'])
    labelutil(cdf,'B_GSE_units',cdfd.bgse_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'B_GSE_sig_units',cdfd.bgsesig_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'V_GSE_units',cdfd.vgse_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'V_GSE_sig_units',cdfd.vgsesig_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'Ne_units',cdfd.n_primebsn_units,['cm^-3'])
    labelutil(cdf,'Ne_sig_units',cdfd.nsig_primebsn_units,['cm^-3'])
    cdf.close()
    return True

# Instantiate PRIME
prime = psw.load('PRIME')

# Define data interval
date_range = pd.date_range(start = interval[0], end = interval[1], freq = '1D') # Daily CDFs

# Pre-make the directory structure
if not os.path.exists(os.path.join(savedir, 'PRIME')):
    os.makedirs(os.path.join(savedir, 'PRIME'))
for year in date_range.year.unique():
    for month in date_range.month.unique():
        if not os.path.exists(os.path.join(savedir, 'PRIME', str(year), str(month))):
            os.makedirs(os.path.join(savedir, 'PRIME', str(year), str(month)))

# Loop over the daily intervals to load the data
for i in range(len(date_range)):
    if i == (len(date_range) - 1): # Skip the last so we can make the indexing ezpz
        continue
    print(f"Saving {i} of {len(date_range)} ({100*i/len(date_range):.02f}%)", end = '\r')
    tries = 0
    while tries < max_retries:
        try:
            data = prime.predict_ts(start = date_range[i], stop = date_range[i+1])
            tries = np.inf
            filename = os.path.join(savedir, 'PRIME', str(date_range[i].year), str(date_range[i].month), f"prime_bsn_{date_range[i].strftime('%Y%m%d')}_v02.cdf")
            if os.path.exists(filename) and overwrite:
                os.remove(filename)
            cdfw(data, filename)
        except (OSError, RuntimeError): # Throws when the data fails to load, either because it doesn't exist or we get rate limited
            tries += 1
        except ValueError: # Telemetry error (too small position, unfilled data) that need not be tried again.
            tries = np.inf
            print(f"Telemetry error on {date_range[i]}. Carrying on.")
