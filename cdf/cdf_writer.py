'''
This python script automatically generates PRIME and PRIME-SH predictions at their default locations and stores those predictions as CDFs.
'''
import numpy as np
from spacepy import pycdf
import cdfdicts as cdfd
import pandas as pd
import primesw as psw
import pathlib

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

def cdfw_sw(data,filename):
    """
    Function that takes dataframe of PRIME outputs and saves as CDF with correct metadata.

    ### Parameters
    
    * data : float, array-like
        >Dataframe of PRIME outputs.
    * filename : str
        >string filename to save the CDF as.
    
    ### Returns
    
    * write : bool
        >Bool of whether file is written.
    
    """
    if (data.shape[0] == 0): #Were we passed an empty array?
        return False
    cdf = pycdf.CDF(filename, create = True)
    cdf.attrs = cdfd.primebsn_att_dict
    cdf.attrs['Logical_file_id'] = filename
    fillutil(cdf,'Epoch',cdfd.epoch_primebsn_att,data['Epoch'].to_numpy())
    fillutil(cdf,'B_GSM',cdfd.bgsm_primebsn_att,data.loc[:, ['B_xgsm', 'B_ygsm', 'B_zgsm']].to_numpy())
    fillutil(cdf,'B_GSM_sig',cdfd.bgsmsig_primebsn_att,data.loc[:, ['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig']].to_numpy())
    fillutil(cdf,'V_GSE',cdfd.vgse_primebsn_att,data.loc[:, ['Vi_xgse', 'Vi_ygse', 'Vi_zgse']].to_numpy())
    fillutil(cdf,'V_GSE_sig',cdfd.vgsesig_primebsn_att,data.loc[:, ['Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig']].to_numpy())
    fillutil(cdf,'Ne',cdfd.n_primebsn_att,data['Ne'].to_numpy())
    fillutil(cdf,'Ne_sig',cdfd.nsig_primebsn_att,data['Ne_sig'].to_numpy())
    #fillutil(cdf,'interpflag',cdfd.flag_primebsn_att,data['interp_frac'].to_numpy())
    labelutil(cdf,'B_GSM_label',cdfd.bgsm_primebsn_label,['Bx GSM','By GSM','Bz GSM'])
    labelutil(cdf,'B_GSM_sig_label',cdfd.bgsmsig_primebsn_label,['Bx GSM Sigma','By GSM Sigma','Bz GSM Sigma'])
    labelutil(cdf,'V_GSE_label',cdfd.vgse_primebsn_label,['Vx GSE','Vy GSE','Vz GSE'])
    labelutil(cdf,'V_GSE_sig_label',cdfd.vgsesig_primebsn_label,['Vx GSE Sigma','Vy GSE Sigma','Vz GSE Sigma'])
    labelutil(cdf,'Ne_label',cdfd.n_primebsn_label,['Ne'])
    labelutil(cdf,'Ne_sig_label',cdfd.nsig_label,['Ne Sigma'])
    labelutil(cdf,'B_GSM_units',cdfd.bgsm_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'B_GSM_sig_units',cdfd.bgsmsig_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'V_GSE_units',cdfd.vgse_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'V_GSE_sig_units',cdfd.vgsesig_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'Ne_units',cdfd.n_primebsn_units,['cm^-3'])
    labelutil(cdf,'Ne_sig_units',cdfd.nsig_primebsn_units,['cm^-3'])
    cdf.close()
    return True

def cdfw_sh(data,filename):
    """
    Function that takes dataframe of PRIME-SH outputs and saves as CDF with correct metadata.

    ### Parameters
    
    * data : float, array-like
        >Dataframe of PRIME-SH outputs.
    * filename : str
        >string filename to save the CDF as.
    
    ### Returns
    
    * write : bool
        >Bool of whether file is written.
    
    """
    if (data.shape[0] == 0): #Were we passed an empty array?
        return False
    cdf = pycdf.CDF(filename, create = True)
    cdf.attrs = cdfd.primesh_att_dict
    cdf.attrs['Logical_file_id'] = filename
    fillutil(cdf,'Epoch',cdfd.epoch_primesh_att,data['Epoch'].to_numpy())
    fillutil(cdf,'B_GSM',cdfd.bgsm_primesh_att,data.loc[:, ['B_xgsm', 'B_ygsm', 'B_zgsm']].to_numpy())
    fillutil(cdf,'B_GSM_sig',cdfd.bgsmsig_primesh_att,data.loc[:, ['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig']].to_numpy())
    fillutil(cdf,'V_GSE',cdfd.vgse_primesh_att,data.loc[:, ['Vi_xgse', 'Vi_ygse', 'Vi_zgse']].to_numpy())
    fillutil(cdf,'V_GSE_sig',cdfd.vgsesig_primesh_att,data.loc[:, ['Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig']].to_numpy())
    fillutil(cdf,'Ni',cdfd.n_primesh_att,data['Ni'].to_numpy())
    fillutil(cdf,'Ni_sig',cdfd.nsig_primesh_att,data['Ni_sig'].to_numpy())
    fillutil(cdf,'Tipar',cdfd.tpar_primesh_att,data['Tipar'].to_numpy())
    fillutil(cdf,'Tipar_sig',cdfd.tparsig_primesh_att,data['Tipar_sig'].to_numpy())
    fillutil(cdf,'Tiperp',cdfd.tpar_primesh_att,data['Tiperp'].to_numpy())
    fillutil(cdf,'Tiperp_sig',cdfd.tparsig_primesh_att,data['Tiperp_sig'].to_numpy())
    #fillutil(cdf,'interpflag',cdfd.flag_primebsn_att,data['interp_frac'].to_numpy())
    labelutil(cdf,'B_GSM_label',cdfd.bgsm_primebsn_label,['Bx GSM','By GSM','Bz GSM'])
    labelutil(cdf,'B_GSM_sig_label',cdfd.bgsmsig_primebsn_label,['Bx GSM Sigma','By GSM Sigma','Bz GSM Sigma'])
    labelutil(cdf,'V_GSE_label',cdfd.vgse_primebsn_label,['Vx GSE','Vy GSE','Vz GSE'])
    labelutil(cdf,'V_GSE_sig_label',cdfd.vgsesig_primebsn_label,['Vx GSE Sigma','Vy GSE Sigma','Vz GSE Sigma'])
    labelutil(cdf,'Ni_label',cdfd.n_primebsn_label,['Ni'])
    labelutil(cdf,'Ni_sig_label',cdfd.nsig_label,['Ni Sigma'])
    labelutil(cdf,'Tipar_label',cdfd.tpar_primesh_label,['Tipar'])
    labelutil(cdf,'Tipar_sig_label',cdfd.tparsig_label,['Tipar Sigma'])
    labelutil(cdf,'Tiperp_label',cdfd.tperp_primesh_label,['Tipar'])
    labelutil(cdf,'Tiperp_sig_label',cdfd.tperpsig_label,['Tiperp Sigma'])
    labelutil(cdf,'B_GSM_units',cdfd.bgsm_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'B_GSM_sig_units',cdfd.bgsmsig_primebsn_units,['nT','nT','nT'])
    labelutil(cdf,'V_GSE_units',cdfd.vgse_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'V_GSE_sig_units',cdfd.vgsesig_primebsn_units,['km/s','km/s','km/s'])
    labelutil(cdf,'Ni_units',cdfd.n_primebsn_units,['cm^-3'])
    labelutil(cdf,'Ni_sig_units',cdfd.nsig_primebsn_units,['cm^-3'])
    labelutil(cdf,'Tipar_units',cdfd.tpar_primesh_units,['eV'])
    labelutil(cdf,'Tipar_sig_units',cdfd.tparsig_primesh_units,['eV'])
    labelutil(cdf,'Tiperp_units',cdfd.tperp_primesh_units,['eV'])
    labelutil(cdf,'Tiperp_sig_units',cdfd.tperpsig_primesh_units,['eV'])
    cdf.close()
    return True

if __name__ == '__main__':
    dates = pd.date_range(pd.to_datetime('1998-11-18 00:00:00+0000'), pd.to_datetime('today', utc=True), freq='1D') #Dates for each one-day cadence file
    sw_model = psw.prime() # class wrapper of PRIME
    sh_model = psw.primesh() # class wrapper of PRIME-SH
    for i in range(len(dates)-1):
        try:
            sw_df = sw_model.predict(start = (dates[i]-pd.Timedelta(seconds = (sw_model.window+sw_model.stride-1)*100)).strftime('%Y-%m-%d %H:%M:%S+0000'), stop = (dates[i+1]-pd.Timedelta(seconds = (sw_model.stride)*100)).strftime('%Y-%m-%d %H:%M:%S+0000')) # Load PRIME prediction
        except (ValueError , RuntimeError) as error: #Throws when there is no solar wind data from Wind for an entire day
            print('SW Error: Wind data missing for '+dates[i].strftime('%Y-%m-%d %H:%M:%S+0000'))
            continue
        try:
            sh_df = sh_model.predict(start = (dates[i]-pd.Timedelta(seconds = (sh_model.window+sh_model.stride-1)*100)).strftime('%Y-%m-%d %H:%M:%S+0000'), stop = (dates[i+1]-pd.Timedelta(seconds = (sh_model.stride)*100)).strftime('%Y-%m-%d %H:%M:%S+0000')) # Load PRIME-SH prediction
        except (ValueError , RuntimeError) as error: #See above
            print('SH Error: Wind data missing for '+dates[i].strftime('%Y-%m-%d %H:%M:%S+0000'))
            continue
        sw_directory = pathlib.Path('./prime/'+dates[i].strftime('%Y/%m')) # Monthly sw and sh file directory
        sh_directory = pathlib.Path('./primesh/'+dates[i].strftime('%Y/%m'))
        sw_directory.mkdir(parents=True, exist_ok = True) # In case the directory doesn't exist, make it
        sh_directory.mkdir(parents=True, exist_ok = True)
        sw_filename = dates[i].strftime('prime_bsn_%Y%m%d.cdf') #Filename of daily sw and sh file
        sh_filename = dates[i].strftime('primesh_mp_%Y%m%d.cdf')
        sw_path = sw_directory / sw_filename #Paths to sw and sh files
        sh_path = sh_directory / sh_filename
        cdfw_sw(sw_df, str(sw_path))
        cdfw_sh(sh_df, str(sh_path))
        print('Saved CDFS for ' + dates[i].strftime('%Y-%m-%d'))
    print('Saving files complete.')