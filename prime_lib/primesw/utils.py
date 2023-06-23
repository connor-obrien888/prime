import numpy as np

IN_KEYS = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
WINDOW = 50 #Number of timesteps to include in each input
STRIDE = 18 #Number of timesteps to skip between each input

def load_input(t0, t1, src = 'Wind', file = None):
    '''
    Utility function that loads input data for PRIME from a file or from CDASWS
    Inputs:
        t0: datetime object, start time of the desired input data
        t1: datetime object, end time of the desired input data
        src: string, name of the source of the input data (currently only 'Wind' is supported)
        file: string, path to an hdf5 file containing the input data (if None, data is loaded from CDASWS)
    Outputs:
        in_arr: numpy array, input data for PRIME (Keras dataset order [samples, time window, features])
        flags: numpy array, fraction of input data that is interpolated
    '''
    if file is None: # load from cdas (requires cdasws, cdflib)
        from cdasws import CdasWs
        cdas = CdasWs()
        #Insert routine that loads Wind kp and cuts/interpolates it sufficiently
        return None
    else: #Load from hdf5 file (requires pandas)
        import pandas as pd
        wind_data = pd.read_hdf(file, key = 'inputs')
        wind_data = wind_data.loc[(wind_data['Epoch'] >= t0-pd.Timedelta(seconds=(WINDOW+STRIDE)*100)) & (wind_data['Epoch'] <= t1-pd.Timedelta(seconds=(STRIDE)*100)), IN_KEYS] #Cut the data to the desired time range
        wind_data['target_Px_gse'] = 6378*13.25 #Point the target at a fixed point (average BS nose location)
        wind_data['target_Py_gse'] = 0
        wind_data['target_Pz_gse'] = 0
        wind_data.index = np.arange(len(wind_data)) #make the indes linear
        inds = np.arange(WINDOW+STRIDE, len(IN_KEYS)) #Indices to start each time series at
        in_arr = np.zeros(( len(inds) , WINDOW, len(IN_KEYS))) #input array staging, Keras dataset order [samples, time window, features]
        flags = np.zeros(len(inds)) #Get a place to put fraction of input data that is interpolated
        for i in np.arange(len(inds)):
            in_arr[i, :, :] =  wind_data.loc[(inds[i] - WINDOW - STRIDE):(inds[i] - STRIDE - 1)] #Get the timeseries that is 'window' long, 'stride' away from the desired time
            tdelt_max = 0
            tdelt = 0
            for n in wind_data.loc[(inds[i] - WINDOW - STRIDE):(inds[i] - STRIDE - 1), 'interp_flag']:
                if (n == 1):
                    tdelt += 1
                    if (tdelt > tdelt_max):
                        tdelt_max = tdelt
                else:
                    tdelt = 0
            flags[i] = tdelt_max #Store the longest number of consecutive interpolated datapoints in the window
        return in_arr, flags





    
