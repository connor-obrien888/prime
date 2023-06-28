import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
################
#Color and style
c1 = '#E76F51'
c2 = '#F4A261'
c3 = '#E9C46A'
c4 = '#2A9D8F'
c5 = '#264653'

taikonaut_colors = [(38/255.0, 70/255.0, 83/255.0),
                  (42/255.0, 157/255.0, 143/255.0),
                  (233/255.0, 196/255.0, 106/255.0),
                  (244/255.0, 162/255.0, 97/255.0),
                  (231/255.0, 111/255.0, 81/255.0)]
taikonaut = LinearSegmentedColormap.from_list('taikonaut', taikonaut_colors, N=10000)

new_oranges_colors = [(253/255.0, 240/255.0, 237/255.0),
                      (220/255.0, 60/255.0, 24/255.0)]
new_oranges = LinearSegmentedColormap.from_list('new_oranges', new_oranges_colors, N=10000)

new_greens_colors = [(239/255.0, 251/255.0, 249/255.0),
                     (33/255.0, 131/255.0, 114/255.0)]
new_greens = LinearSegmentedColormap.from_list('new_greens', new_greens_colors, N=10000)

rb_colors = [(183/255.0, 9/255.0, 76/255.0),
             (137/255.0, 43/255.0, 100/255.0),
             (92/255.0, 77/255.0, 125/255.0),
             (0/255.0, 145/255.0, 173/255.0)]
rb = LinearSegmentedColormap.from_list('rb', rb_colors, N=10000)
################

EARTH_RADIUS = 6378 #Re in km

WIND_ASCII_COLS = ['year', 'doy', 'hour', 'minute', 'IMF_PTS', 'percent_interp', 'CPMV', 'Timeshift', 'Phase_n_x', 'Phase_n_y', 
                   'Phase_n_z', 'BX_GSE', 'BY_GSM', 'BZ_GSM', 'RMS_Timeshift', 'RMS_phase', 'flow_speed', 'VX_GSE', 'VY_GSE', 
                   'VZ_GSE', 'proton_density', 'T', 'SC_X', 'SC_Y', 'SC_Z', 'BSN_X', 'BSN_Y', 'BSN_Z', 'RMS_target'
                  ] #Columns in Wind ASCII files from OMNIWeb (as downloaded Sept. 7 2022)

I_ENG_BINS = np.array([1.133000e+01, 1.455000e+01, 1.869000e+01, 2.400000e+01,
                       3.083000e+01, 3.960000e+01, 5.087000e+01, 6.534000e+01,
                       8.392000e+01, 1.078000e+02, 1.384600e+02, 1.778600e+02,
                       2.284500e+02, 2.934400e+02, 3.769200e+02, 4.841500e+02,
                       6.218800e+02, 7.988000e+02, 1.026040e+03, 1.317930e+03,
                       1.692860e+03, 2.174450e+03, 2.793050e+03, 3.587620e+03,
                       4.608240e+03, 5.919200e+03, 7.603110e+03, 9.766070e+03,
                       1.254435e+04, 1.611301e+04, 2.069688e+04, 2.658479e+04])
I_ENG_NAMES = np.array(['i_eng_0', 'i_eng_1', 'i_eng_2', 'i_eng_3', 'i_eng_4', 'i_eng_5', 'i_eng_6', 'i_eng_7', 'i_eng_8', 'i_eng_9', 
                        'i_eng_10', 'i_eng_11', 'i_eng_12', 'i_eng_13', 'i_eng_14', 'i_eng_15', 'i_eng_16', 'i_eng_17', 'i_eng_18', 'i_eng_19', 
                        'i_eng_20', 'i_eng_21', 'i_eng_22', 'i_eng_23', 'i_eng_24', 'i_eng_25', 'i_eng_26', 'i_eng_27', 'i_eng_28', 'i_eng_29', 'i_eng_30', 'i_eng_31'])

E_ENG_BINS = np.array([1.166000e+01, 1.495000e+01, 1.917000e+01, 2.458000e+01,
                       3.152000e+01, 4.042000e+01, 5.183000e+01, 6.645000e+01,
                       8.521000e+01, 1.092600e+02, 1.400900e+02, 1.796300e+02,
                       2.303300e+02, 2.953300e+02, 3.786900e+02, 4.855600e+02,
                       6.226000e+02, 7.983200e+02, 1.023630e+03, 1.312530e+03,
                       1.682970e+03, 2.157950e+03, 2.766990e+03, 3.547920e+03,
                       4.549250e+03, 5.833180e+03, 7.479480e+03, 9.590410e+03,
                       1.229711e+04, 1.576772e+04, 2.021784e+04, 2.592391e+04])
E_ENG_NAMES = np.array(['e_eng_0', 'e_eng_1', 'e_eng_2', 'e_eng_3', 'e_eng_4', 'e_eng_5', 'e_eng_6', 'e_eng_7', 'e_eng_8', 'e_eng_9', 
                        'e_eng_10', 'e_eng_11', 'e_eng_12', 'e_eng_13', 'e_eng_14', 'e_eng_15', 'e_eng_16', 'e_eng_17', 'e_eng_18', 'e_eng_19', 
                        'e_eng_20', 'e_eng_21', 'e_eng_22', 'e_eng_23', 'e_eng_24', 'e_eng_25', 'e_eng_26', 'e_eng_27', 'e_eng_28', 'e_eng_29', 'e_eng_30', 'e_eng_31'])

def dasilva_class(ni, t, 
               mat = np.asarray([[4.323,1.871],[-2.208,1.908],[-1.702,-3.866]]),
               bvec = np.asarray([-8.794,-3.636,9.752])):
    '''
    Classifies MMS regions as magnetosheath (0), magnetosphere (1), or solar wind (2). From da Silva et al. 2020.

    Parameters
    ----------
    ni : float, array-like
        Array of ion number density in cm-3.
    t : float, array-like
        Array of time in UNIX UTC TIMESTAMP.
    mat : float, array-like, optional
        Matrix for the region partition. The default is 
        np.asarray([[4.323,1.871],[-2.208,1.908],[-1.702,-3.866]]).
    bvec : float, array-like, optional
        Bias vector for the region partition. The default is
        np.asarray([-8.794,-3.636,9.752])).

    Returns
    -------
    region : float, array-like
        Array of region flags. magnetosheath = 0, magnetosphere = 1, 
        and solar wind = 2.

    '''
    try:
        itertest = [iter(ni),iter(t)] #tests if inputs are iterable, if not "except" lines fire
        vec = np.asarray([ni,t]) #Vectorize the input params
        bias = np.ones((len(vec.transpose()),3)) #Construct linear "bias" term
        bias[:,0] *= bvec[0]
        bias[:,1] *= bvec[1]
        bias[:,2] *= bvec[2]
        score = np.dot(mat,np.log10(vec)).transpose()+bias #dot with matrix, add bias
        region = np.argmax(score,axis=1) #What's the max score for each point
    except TypeError: #This fires if ni/t are scalars
        vec = np.asarray([ni,t]) #Vectorize the input params
        bias = bvec #Construct linear "bias" term
        score = np.dot(mat,np.log10(vec)).transpose()+bias #dot with matrix, add bias
        region = np.argmax(score) #What's the max score
    return region

def olshevsky_class(dist, model_file = '../libraries/olshevsky_lib/cnn_dis_201711_verify.h5'):
    '''
    Classifies MMS regions as solar wind (0), ion foreshock (1), magnetosheath (2), or magnetosphere (3).
    Uses full 3D distribution function. From Olshevsky et al. 2021. REQUIRES OLSHEVSKY LIBRARY AND KERAS

    Parameters
    ----------
    dist : float, array-like
        3D ion distribution as measured by DIS. Corresponds to variable mmsX_dis_dist_fast in FPI dis-dist data.
    model_file : string, optional
        Path to saved keras model file. Default '../libraries/olshevsky_lib/cnn_dis_201711_verify.h5', can also use
        '../libraries/olshevsky_lib/cnn_dis_201712_verify.h5' for the other model trained in Olshevsky et al. 2021.

    Returns
    -------
    label : float, array-like
        Array of region flags. solar wind = 0, ion foreshock = 1, 
        magnetosheath = 2, magnetosphere = 3.
    predictions : float, array-like
        Array of normalized label probabilities. Index corresponds to region as in labels output.

    '''
    from tensorflow.keras.models import Model, load_model
    dist_norm = normalize_data(dist, verbose=False)
    dist_norm = dist_norm.reshape(dist_norm.shape + (1,))
    
    model = load_model(model_file) #Load model from saved HDF

    # Probability for each class
    probability = model.predict(dist_norm)
    # The most probable class
    label = probability.argmax(axis=1)
    return label, probability

def normalize_data(X, verbose=True):
    """ Compute logarithm and normalize the data for learning.
    FROM OLSHEVSY ET AL 2021

    Parameters:
        X - [epoch, Phi, Theta, Energy]

    """

    # Old way
    if verbose:
        print('Normalizing data array', X.shape)
    try:
        min_value = np.ma.masked_equal(X, 0.0, copy=False).min() #NOTE: this is modified to improve execution time/memory use. Original operation is np.min(X[np.nonzero(X)])
    except ValueError:
        print('Warning! All elements of X are zero, returning a zero-array.')
        return X
    if verbose:
        print('Replacing zeros with min...')
    #X = np.where(np.isclose(X, 0, rtol=0, atol=1e-30), min_value, X)
    #Replaced above line with below line to improve memory usage. Since X>=0 everywhere, they are equivalent.
    X[X <= 1e-30] = min_value
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X -= X.min()

    '''
    # New way
    min_value = 1e-30
    if verbose:
        print('Replacing negatives with zeros...')
    X = np.where(X > 0, X, min_value)
    if verbose:
        print('Computing log10...')
    X = np.log10(X)
    if verbose:
        print('Subtracting min...')
    X += 30
    '''
    if verbose:
        print('Normalizing to 1...')
    X /= X.max()
    if verbose:
        print('Rolling along Phi...')
    X = np.roll(X, 16, axis=X.ndim-2)
    return X

def class_query(t1, t2, class_df = None, datastore = 'ol_class.h5', key = 'ol_class', datapath = '../data/'):
    '''
    Queries the saved classifier output dataframe (since it's so huge you can't do normal cuts).
    t1 and t2 are UNIX timestamps. You can provide class_df for speed, otherwise it will load at call time.

    '''
    if class_df is None:
        class_df = pd.read_hdf(datapath + datastore, key = key, mode ='a')
    ind1 = round(np.interp(t1, class_df['time'].to_numpy(), np.arange(len(class_df)))) #Closest index for t1
    ind2 = round(np.interp(t2, class_df['time'].to_numpy(), np.arange(len(class_df)))) #Closest index for t2
    return class_df[ind1:ind2]

def translate_labels(label, classifier = 'dasilva'):
    '''
    Changes labels output by classifiers ('dasilva', 'olshevsky') into more apt order.
    This order is 0:magnetosphere, 1:magnetosheath, 2:solar wind, 3:foreshock.
    Physically, MMS must only move between adjacent numbers (except foreshock, which can be moved to from the magnetosheath). 
    This makes potentially unphysical predictions easier to spot.

    '''
    if (classifier == 'dasilva'):
        label_new = np.where((label==0)|(label==1), (label+1)%2, label) #This changes 0s to 1s and vice versa
    if (classifier == 'olshevsky'):
        label_new = (label+2)%4 #This changes 0->2, 1->3, 2->0, 3-1
        label_new = np.where((label_new==0)|(label_new==1), (label_new+1)%2, label_new) #This changes 0s to 1s and vice versa
    return label_new

def translate_probabilities(probability):
    '''
    Changes order of probabilities output by Olshevsky classifier to align with label order in translate_labels.
    This order is 0:magnetosphere, 1:magnetosheath, 2:solar wind, 3:foreshock.
    Physically, MMS must only move between adjacent numbers (except foreshock, which can be moved to from the magnetosheath). 
    This makes potentially unphysical predictions easier to spot.

    '''
    new_probability = np.zeros(probability.shape) #Make an empty array in the shape of probability array
    new_probability[:, 0] = probability[:, 3] #3->0
    new_probability[:, 1] = probability[:, 2] #2->1
    new_probability[:, 2] = probability[:, 0] #0->2
    new_probability[:, 3] = probability[:, 1] #1->3
    return new_probability

def get_cdas_keys(instrument, dtype, sc, rate):
    '''
    Returns the name of the dataset in CDAS and the CDF keys to load.
    '''
    if instrument == 'fpi':
        if dtype == 'dis-moms':
            if rate == 'fast':
                dataset = 'MMS'+sc[3]+'_FPI_FAST_L2_DIS-MOMS'
            if rate == 'brst':
                dataset = 'MMS'+sc[3]+'_FPI_BRST_L2_DIS-MOMS'
        if dtype == 'des-moms':
            if rate == 'fast':
                dataset = 'MMS'+sc[3]+'_FPI_FAST_L2_DES-MOMS'
            if rate == 'brst':
                dataset = 'MMS'+sc[3]+'_FPI_BRST_L2_DES-MOMS'
        keys = [sc+'_'+dtype[0:3]+'_bulkv_gse_'+rate, sc+'_'+dtype[0:3]+'_numberdensity_'+rate, sc+'_'+dtype[0:3]+'_errorflags_'+rate]
    if instrument == 'fgm':
        if rate == 'srvy':
            dataset = 'MMS'+sc[3]+'_FGM_SRVY_L2'
        if rate == 'brst':
            dataset = 'MMS'+sc[3]+'_FGM_BRST_L2'
        keys = [sc+'_fgm_b_gsm_'+rate+'_l2', sc+'_fgm_flag_'+rate+'_l2']
    if instrument == 'mec':
        if rate == 'srvy':
            dataset = 'MMS'+sc[3]+'_MEC_SRVY_L2_EPHT89D'
        if rate == 'brst':
            dataset = 'MMS'+sc[3]+'_MEC_BRST_L2_EPHT89D'
        keys = [sc+'_mec_r_gsm', sc+'_mec_r_gse']
    return dataset, keys

def load_util(start_date, end_date, instrument, dtype, sc, rate, freq='1D'):
    '''
    Loads downloaded data products for given instrument/spacecraft from start_date to end_date.
    For large datasets, adjust freq to load in chunks. For too large a chunk (RAM limited), the data will not load.

    '''
    from cdasws import CdasWs
    cdas = CdasWs()
    data = pd.DataFrame(dtype = object) #Initialize empty dataframe to put data
    dates = pd.date_range(start_date, end_date+pd.Timedelta(freq), freq=freq).strftime('%Y-%m-%d').tolist() #List of dates to loop over
    for idx, date in enumerate(dates):
        if idx == len(dates)-1: #If we're at the last date, break
            break
        dataset, keys = get_cdas_keys(instrument, dtype, sc, rate) #Get the name of the dataset in CDAS and the CDF keys to load
        data = cdas.get_data(dataset, keys, date, dates[idx+1]) #Load data from CDAS
        df_stage = pd.DataFrame(dtype=object)
        if instrument == 'fpi':
            df_stage['Epoch'] = data[1]['Epoch']
            for key in keys:
                if key != sc+'_'+dtype[0:3]+'_bulkv_gse_'+rate: #Bulk velocity is three components, so we need to split it up in the else statement
                    df_stage[key] = data[1][key]
                else:
                    df_stage[key+'_x'] = data[1][key][:,0]
                    df_stage[key+'_y'] = data[1][key][:,1]
                    df_stage[key+'_z'] = data[1][key][:,2]
        if instrument == 'fgm':
            df_stage['Epoch'] = data[1]['Epoch']
            for key in keys:
                if key != sc+'_'+instrument+'_b_gsm_'+rate+'_l2': #Magnetic field is three components, so we need to split it up in the else statement
                    df_stage[key] = data[1][key]
                else:
                    df_stage[key+'_x'] = data[1][key][:,0]
                    df_stage[key+'_y'] = data[1][key][:,1]
                    df_stage[key+'_z'] = data[1][key][:,2]
                    df_stage[key+'_mag'] = data[1][key][:,3]
        elif instrument == 'mec':
            keys = ['Epoch', sc+'_'+instrument+'_r_gsm', sc+'_'+instrument+'_r_gse']
            df_stage['Epoch'] = data[1]['Epoch']
            for key in keys:
                if (key != 'Epoch', sc+'_'+instrument+'_r_gsm')&(key!= sc+'_'+instrument+'_r_gse'): #Position is three components, so we need to split it up in the else statement
                    df_stage[key] = data[1][key]
                else:
                    df_stage[key+'_x'] = data[1][key][:,0]
                    df_stage[key+'_y'] = data[1][key][:,1]
                    df_stage[key+'_z'] = data[1][key][:,2]
        data = pd.concat([data, df_stage], ignore_index = True) #Concatenate dataframes
    return data

def clean_columns(raw_data, instrument, dtype, rate, sort = False):
    '''
    Returns dataframe with new/expanded column names and sensible time values. Designed to operate on df produced by load_util.
    Use sort = True to sort by time, slow for large dataframes.
    '''
    if instrument == 'fpi':
        clean_df = raw_data.copy()
        clean_df.rename(columns={'mms1_'+dtype[0:3]+'_numberdensity_'+rate: 'n_'+dtype[1], 'mms1_'+dtype[0:3]+'_bulkv_gse_'+rate+'_x': 'V'+dtype[1]+'x_gse', 'mms1_'+dtype[0:3]+'_bulkv_gse_'+rate+'_y': 'V'+dtype[1]+'y_gse', 'mms1_'+dtype[0:3]+'_bulkv_gse_'+rate+'_z': 'V'+dtype[1]+'z_gse', 'mms1_'+dtype[0:3]+'_temppara_'+rate: 'T'+dtype[1]+'_para', 'mms1_'+dtype[0:3]+'_tempperp_'+rate: 'T'+dtype[1]+'_perp', 'mms1_'+dtype[0:3]+'_errorflags_'+rate: 'eflags'}, inplace=True)
        clean_df['Epoch'] = pd.to_datetime(raw_data['Epoch'], utc=True) #Convert to UTC aware datetime
    elif instrument == 'fgm':
        clean_df = raw_data.copy()
        clean_df.rename(columns={'mms1_'+instrument+'_b_gsm_'+rate+'_l2_x': 'Bx_gsm', 'mms1_'+instrument+'_b_gsm_'+rate+'_l2_y': 'By_gsm', 'mms1_'+instrument+'_b_gsm_'+rate+'_l2_z': 'Bz_gsm', 'mms1_'+instrument+'_b_gsm_'+rate+'_l2_mag': 'B_gsm', 'mms1_'+instrument+'_flag_'+rate+'_l2': 'eflags'}, inplace=True)
        clean_df['Epoch'] = pd.to_datetime(raw_data['Epoch'], utc=True) #Convert to UTC aware datetime
    elif instrument == 'mec':
        clean_df = raw_data.copy()
        clean_df.rename(columns={'mms1_'+instrument+'_r_gsm_x': 'Px_gsm', 'mms1_'+instrument+'_r_gsm_y': 'Py_gsm', 'mms1_'+instrument+'_r_gsm_z': 'Pz_gsm', 'mms1_'+instrument+'_r_gse_x': 'Px_gse', 'mms1_'+instrument+'_r_gse_y': 'Py_gse', 'mms1_'+instrument+'_r_gse_z': 'Pz_gse'}, inplace=True)
        clean_df['Epoch'] = pd.to_datetime(raw_data['Epoch'], utc=True) #Convert to UTC aware datetime
    return clean_df

def mms_load_util(start_date, end_date, keys, clear_data = False, datapath = '../data/pydata'):
    '''
    Loads MMS data for given time range, as well as classifies. Specify tplot keys in keys array.
    
    '''
    import pyspedas.mms as mms
    import pytplot
    from pytplot import tplot
    import pytplot
    from pytplot import tplot
    load_trange = [start_date.strftime('%Y-%m-%d/%H:%M'), end_date.strftime('%Y-%m-%d/%H:%M')]
    sc = 'mms1'
    fpi_i_load = False #Flags for whether we need to download the data
    fpi_e_load = False
    fgm_load = False
    mec_load = False
    class_load = False
    
    fpi_df = pd.DataFrame(dtype = object) #Initialize empty dataframes to put each type of data
    fgm_df = pd.DataFrame(dtype = object)
    mec_df = pd.DataFrame(dtype = object)
    cla_df = pd.DataFrame(dtype = object)
    
    for key in keys:
        if (key == 'mms1_dis_bulkv_gse_fast')|(key == 'mms1_dis_numberdensity_fast')|(key == 'mms1_dis_tempperp_fast')|(key == 'mms1_dis_energyspectr_omni_fast'):
            if not fpi_i_load:
                fpi_i_data = mms.fpi(trange=load_trange,datatype='dis-moms',data_rate='fast', level='l2')
                fpi_t_arr = pytplot.data_quants[key].coords['time'].values #FPI Timestamp
                fpi_df['time'] = fpi_t_arr #Store the time in the dataframe
                fpi_i_load = True #We've loaded the data! Don't do it again!
            data_arr = pytplot.data_quants[key].values #Put the data in an array
            if (key == 'mms1_dis_bulkv_gse_fast'): #The velocity data is actually in three component vectors
                fpi_df['Vx_gse'] = data_arr[:, 0]
                fpi_df['Vy_gse'] = data_arr[:, 1]
                fpi_df['Vz_gse'] = data_arr[:, 2]
            if (key == 'mms1_dis_numberdensity_fast'):
                fpi_df['n_i'] = data_arr
            if (key == 'mms1_dis_tempperp_fast'):
                fpi_df['T_i'] = data_arr
            if (key == 'mms1_dis_energyspectr_omni_fast'):
                staging_df = pd.DataFrame(data_arr, columns = I_ENG_NAMES)
                fpi_df = pd.concat([fpi_df, staging_df], axis = 1)
        
        if (key == 'mms1_des_energyspectr_omni_fast'):
            if not fpi_e_load:
                fpi_e_data = mms.fpi(trange=load_trange,datatype='des-moms',data_rate='fast', level='l2')
                fpi_t_arr = pytplot.data_quants[key].coords['time'].values #FPI Timestamp
                fpi_df['time'] = fpi_t_arr #Store the time in the dataframe
                fpi_e_load = True #We've loaded the data! Don't do it again!
            data_arr = pytplot.data_quants[key].values #Put the data in an array
            staging_df = pd.DataFrame(data_arr, columns = E_ENG_NAMES)
            fpi_df = pd.concat([fpi_df, staging_df], axis = 1)
            
        if (key == 'mms1_fgm_b_gsm_srvy_l2'):
            if not fgm_load:
                fgm_data = mms.fgm(trange=load_trange,data_rate='srvy', level='l2')
                fgm_t_arr = pytplot.data_quants['mms1_fgm_b_gsm_srvy_l2'].coords['time'].values #FGM timestamp
                fgm_df['time'] = fgm_t_arr #Store the time in the dataframe
                fgm_load = True #We've loaded the data! Don't do it again!
            b_arr = pytplot.data_quants['mms1_fgm_b_gsm_srvy_l2'].values #B GSM X/Y/Z/tot (nT)
            fgm_df['Bx_gsm'] = b_arr[:, 0]
            fgm_df['By_gsm'] = b_arr[:, 1]
            fgm_df['Bz_gsm'] = b_arr[:, 2]
            fgm_df['B_tot']  = b_arr[:, 3]
        
        if (key == 'mms1_mec_r_gse'):
            if not mec_load:
                mec_data = mms.mec(trange=load_trange)
                mec_t_arr = pytplot.data_quants['mms1_mec_r_gse'].coords['time'].values #MEC Timestamp
                mec_df['time'] = mec_t_arr #Store the time in the dataframe
                mec_load = True #We've loaded the data! Don't do it again!
            pos_arr = pytplot.data_quants['mms1_mec_r_gse'].values/EARTH_RADIUS #GSE Position (RE)
            mec_df['Px_gse'] = pos_arr[:, 0]
            mec_df['Py_gse'] = pos_arr[:, 1]
            mec_df['Pz_gse'] = pos_arr[:, 2]
        
        if (key == 'ol_class'):
            cla_df = class_query(start_date.timestamp(), end_date.timestamp(), datapath = datapath)
        try:
            pytplot.del_data(name=key)
        except:
            pass
        
    return fpi_df, fgm_df, mec_df, cla_df

def omni_shifter(omni_data, mms_data):
    '''
    Propogates omni_data from bow shock nose to mms observations using planar propagation method.
    
    '''
    omni_ind = np.zeros(len(mms_data))
    shift = np.zeros(len(mms_data))

    for i in np.arange(len(mms_data)):
        nx = omni_data['Phase_n_x']
        ny = omni_data['Phase_n_y']
        nz = omni_data['Phase_n_z']

        Rox = omni_data['BSN_X'] * EARTH_RADIUS #Convert from Re to km
        Roy = omni_data['BSN_Y'] * EARTH_RADIUS
        Roz = omni_data['BSN_Z'] * EARTH_RADIUS

        Vx = omni_data['VX_GSE']
        Vy = omni_data['VY_GSE']
        Vz = omni_data['VZ_GSE']

        Rdx = mms_data['Px_gse'][i]
        Rdy = mms_data['Py_gse'][i]
        Rdz = mms_data['Pz_gse'][i]

        mms_time = mms_data['time'][i]
        omni_time = omni_data['time']

        #Time delay between mms_data[i] and every entry in omni_data
        delta_t = (nx * (Rdx - Rox) +  ny * (Rdy - Roy) + nz * (Rdz - Roz)) / ((nx * Vx) + (ny * Vy) + (nz * Vz))

        omni_ind[i] = np.argmin(np.abs(omni_time + delta_t - mms_time)) #which piece of omni_data corresponds to this adjusted time?
        shift[i] = np.min(np.abs(omni_time + delta_t - mms_time))
        print('OMNI Propagation '+str(100*i/len(mms_data))[0:5]+'% Complete', end='\r')
    omni_data_shift = omni_data[omni_ind]
    return omni_data_shift

def r_quick(n, vx, vy, vz, bx, by, bz):
    '''
    Calculates R_quick (mV/m) from solar wind n (cm^-3), V (km/s), and B (nT).
    See Borovsky and Birn 2013 for derivation. (doi.org/10.1002/2013JA019193)
    
    '''
    v = np.sqrt(vx**2 + vy**2 + vz**2) #Flow velocity magnitude (km/s)
    b = np.sqrt(bx**2 + by**2 + bz**2) #IMF magnitude (nT)
    theta = np.arctan2(by, bz) #IMF clock angle (radians)
    ma = (v * (n**0.5) / b) * 0.045846 #Alfven mach number
    c = (2.44e-4 + (1 + 1.38*np.log(ma))**(-6))**(-1/6) #Bow shock compression ratio
    beta = (ma/6)**1.92 #Magnetosheath plasma beta
    r_q = 0.4 * (np.sin(theta/2)**2) * (c**(-1/2)) * (n**(1/2)) * (v**2) * ((1 + beta)**(-3/4)) * 4.5846e-5 #Reconnection rate (mV/m)
    return r_q

def Em(vx, vy, vz, by, bz):
    '''
    Calculates solar wind electric field Em (mV/m) from solar wind V (km/s) and B (nT).
    
    '''
    theta = np.arctan2(by, bz) #Clock angle to calculate Em
    B = np.sqrt(by**2 + bz**2) #B mag perp to GSM X axis to calculate Em
    V = np.sqrt(vx**2 + vy**2 + vz**2) #SW Velocity magnitude
    Em = V * B * (np.sin(theta / 2)**2) * 1e-3 #Dawn dusk electric field
    return Em

def Em_err(vx, vx_sig, vy, vy_sig, vz, vz_sig, by, by_sig, bz, bz_sig):
    '''
    Calculates solar wind electric field Em error (mV/m) from solar wind V (km/s) and B (nT) and associated uncertainties.
    
    '''
    dEmvx = vx * (vx**2 + vy**2 + vz**2)**(-1/2) * (by**2 + bz**2)**(1/2) * (np.sin(np.arctan2(by, bz))**2) #Partial derivative of Em wrt Vx
    dEmvy = vy * (vx**2 + vy**2 + vz**2)**(-1/2) * (by**2 + bz**2)**(1/2) * (np.sin(np.arctan2(by, bz))**2) #Partial derivative of Em wrt Vy
    dEmvz = vz * (vx**2 + vy**2 + vz**2)**(-1/2) * (by**2 + bz**2)**(1/2) * (np.sin(np.arctan2(by, bz))**2) #Partial derivative of Em wrt Vz
    dEmby = by * (vx**2 + vy**2 + vz**2)**(1/2) * (by**2 + bz**2)**(-1/2) * (np.sin(np.arctan2(by, bz))**2) + (vx**2 + vy**2 + vz**2)**(1/2) * (by**2 + bz**2)**(1/2) * 2 * (by / (bz**2)) / ((by**2) / (bz**2) + 1)**2 #Partial derivative of Em wrt By
    dEmbz = bz * (vx**2 + vy**2 + vz**2)**(1/2) * (by**2 + bz**2)**(-1/2) * (np.sin(np.arctan2(by, bz))**2) - (vx**2 + vy**2 + vz**2)**(1/2) * (by**2 + bz**2)**(1/2) * 2 * ((by**2) / (bz**3)) / ((by**2) / (bz**2) + 1)**2  #Partial derivative of Em wrt Bz
    Em_err = np.sqrt((dEmvx * vx_sig)**2 + (dEmvy * vy_sig)**2 + (dEmvz * vz_sig)**2 + (dEmby * by_sig)**2 + (dEmbz * bz_sig)**2) #Add errors in quadrature
    return Em_err

def pdyn(n, vx):
    '''
    Calculates solar wind dynamic pressure (nPa) from solar wind n (cm^-3), Vx (km/s).
    
    '''
    pdyn = n * vx**2 * 1.673e-6 #Dynamic pressure (nPa)
    return pdyn

def pdyn_err(n, vx, n_sig, vx_sig):
    '''
    Propagates solar wind dynamic pressure error (nPa) from solar wind n (cm^-3), Vx (km/s).
    
    '''
    err = np.sqrt((n_sig * vx**2 * 1.673e-6)**2 + (n * vx * vx_sig * 2 * 1.673e-6)**2) #Dynamic pressure (nPa)
    return err

def mag_err(x, x_sig, y, y_sig, z, z_sig):
    '''
    Calculates the error of a vector magnitude from the vector components and their associated uncertainties.
    x, y, z: vector components
    '''
    err = np.sqrt((x * x_sig)**2 + (y * y_sig)**2 + (z * z_sig)**2)/np.sqrt(x**2 + y**2 +z**2) #Add errors in quadrature
    return err

def theta_err(y, y_sig, z, z_sig):
    err = np.sqrt((y * y_sig)**2 + (z * z_sig)**2)/(y**2 + z**2) #Add errors in quadrature
    return err

def index_shifter(in_df, shift, keys):
    '''
    Helper function that generates a shifted index for a dataframe with missing/NaN values.
    in_df: dataframe index is pulled from (assumed to be sequential starting from 0)
    shift: number of entries to shift index
    keys: columns containing the relevant missing/NaN values
    '''
    safe_index = in_df[~in_df[keys].isna()].index #Find non-nan values
    shift_index = safe_index + shift #shift the index
    shift_index = shift_index[(shift_index>=0)&(shift_index<len(in_df))] #Ensure the index is still within bounds
    return shift_index

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

    ### Parameters
    
    * data : float, array-like
        >Dataframe of PRIME outputs.
    * filename : str
        >string filename to save the CDF as.
    
    ### Returns
    
    * write : bool
        >Bool of whether file is written.
    
    """
    from spacepy import pycdf
    import cdfdicts as cdfd
    if (data.shape[0] == 0): #Were we passed an empty array?
        return False
    cdf = pycdf.CDF(filename, create=True)
    cdf.attrs = cdfd.primebsn_att_dict
    cdf.attrs['Logical_file_id'] = filename
    fillutil(cdf,'Epoch',cdfd.epoch_primebsn_att,data['Epoch'].to_numpy())
    fillutil(cdf,'B_GSM',cdfd.bgsm_primebsn_att,data.loc[:, ['B_xgsm', 'B_ygsm', 'B_zgsm']].to_numpy())
    fillutil(cdf,'B_GSM_sig',cdfd.bgsmsig_primebsn_att,data.loc[:, ['B_xgsm_sig', 'B_ygsm_sig', 'B_zgsm_sig']].to_numpy())
    fillutil(cdf,'V_GSE',cdfd.vgse_primebsn_att,data.loc[:, ['Vi_xgse', 'Vi_ygse', 'Vi_zgse']].to_numpy())
    fillutil(cdf,'V_GSE_sig',cdfd.vgsesig_primebsn_att,data.loc[:, ['Vi_xgse_sig', 'Vi_ygse_sig', 'Vi_zgse_sig']].to_numpy())
    fillutil(cdf,'Ne',cdfd.n_primebsn_att,data['Ne'].to_numpy())
    fillutil(cdf,'Ne_sig',cdfd.nsig_primebsn_att,data['Ne_sig'].to_numpy())
    fillutil(cdf,'interpflag',cdfd.flag_primebsn_att,data['interp_frac'].to_numpy())
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