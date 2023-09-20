import numpy as np

IN_KEYS = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
WINDOW = 50 #Number of timesteps to include in each input
STRIDE = 18 #Number of timesteps to skip between each input

#MMS orbit that ends at bow shock nose stride minutes from the end of the window 
#(13.25RE, 0RE, 0RE) (from 2023-01-24 02:46:30+0000 - window - stride to 2023-01-24 02:46:30+0000 - stride)
synth_xpos = np.array([69215.97057508, 69480.44662705, 69706.40911294, 69969.18467343,
                       70231.11857452, 70454.91674057, 70715.18415549, 70974.62662114,
                       71196.30325009, 71454.11198298, 71711.11182052, 71930.70839187,
                       72186.10608653, 72440.71045452, 72658.26693929, 72911.29961567,
                       73163.55400328, 73379.10893932, 73629.82119345, 73879.76950962,
                       74093.36015524, 74341.79489098, 74589.47977345, 74801.14208574,
                       75047.34088887, 75292.80336019, 75502.57231563, 75746.57534655,
                       75989.85512801, 76197.76442711, 76439.61054906, 76680.7461448 ,
                       76886.82833966, 77126.55527427, 77365.58400364, 77569.8706008 ,
                       77807.51485289, 78044.47276422, 78246.99446109, 78482.59130139,
                       78717.51337097, 78918.29986461, 79151.88352716, 79384.80363257,
                       79583.88365476, 79815.48746062, 80046.43841324, 80243.83985851,
                       80473.4959342 , 80702.50977538])
synth_ypos = np.array([-6242.531374  , -6141.43603983, -6054.73851884, -5953.54260001,
                       -5852.30035979, -5765.48374707, -5664.15672177, -5562.79111177,
                       -5475.87538835, -5374.44022431, -5272.97399371, -5185.97841963,
                       -5084.45744369, -4982.91267323, -4895.85591308, -4794.27072076,
                       -4692.66879919, -4605.56897563, -4503.9403974 , -4402.3019605 ,
                       -4315.17661655, -4213.52495494, -4111.87001052, -4024.73625516,
                       -3923.08129418, -3821.42952425, -3734.30395957, -3632.66493195,
                       -3531.03538868, -3443.93418239, -3342.32979659, -3240.74102942,
                       -3153.67997007, -3052.12846626, -2950.598573  , -2863.59314841,
                       -2762.11237883, -2660.65906956, -2573.72426935, -2472.33166917,
                       -2370.97225436, -2284.122821  , -2182.83554019, -2081.58702282,
                       -1994.83737245, -1893.67203926, -1792.55094402, -1705.91518784,
                       -1604.88808488, -1503.91065025])
synth_zpos = np.array([1428.22663895, 1404.9257232 , 1384.88999865, 1361.43852111,
                       1337.90827722, 1317.66995298, 1293.97504102, 1270.1942372 ,
                       1249.73509015, 1225.77542686, 1201.72276543, 1181.02453785,
                       1156.77888242, 1132.43315908, 1111.47770227, 1086.92488254,
                       1062.26489785, 1041.03400953, 1016.15277964,  991.1573987 ,
                       969.6329908 ,  944.40232752,  919.05058092,  897.21474963,
                       871.61379271,  845.88483511,  823.71973863,  797.7278038 ,
                       771.60099544,  749.08907437,  722.68572634,  696.14074783,
                       673.26462524,  646.42970201,  619.4464814 ,  596.18909891,
                       568.90278803,  541.46164749,  517.80622717,  490.04905226,
                       462.13065323,  438.0606711 ,  409.81356785,  381.39897532,
                       356.89834051,  328.14264026,  299.21335127,  274.26632698,
                       244.98385268,  215.52178328])
synth_pos = np.array([synth_xpos, synth_ypos, synth_zpos]).T

def load_input(t0, t1, loc=None, src = 'Wind', file = None, hdf_key = 'inputs', in_scaler = None):
    '''
    Utility function that loads input data for PRIME from a file or from CDASWS
    Inputs:
        t0: datetime object, start time of the desired input data
        t1: datetime object, end time of the desired input data
        loc: array, location of the target in GSE coordinates with units of Earth radii (if None, target is pointed at the bow shock). Shape (3,).
        src: string, name of the source of the input data (currently only 'Wind' is supported)
        file: string, path to an hdf5 file containing the input data (if None, data is loaded from CDASWS)
        hdf_key: string, key to the input data in the hdf5 file
        in_scaler: sklearn scaler object, scaler to use to scale the input data (if None, default scaler is used)
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
        wind_data = pd.read_hdf(file, key = hdf_key)
        wind_data = wind_data.loc[(wind_data['Epoch'] >= t0-pd.Timedelta(seconds=(WINDOW+STRIDE)*100)) & (wind_data['Epoch'] <= t1-pd.Timedelta(seconds=(STRIDE)*100)), IN_KEYS] #Cut the data to the desired time range
        if (loc is not None):
            wind_data['target_Px_gse'] = 6378*loc[0]
            wind_data['target_Py_gse'] = 6378*loc[1]
            wind_data['target_Pz_gse'] = 6378*loc[2]
        else:
            wind_data['target_Px_gse'] = synth_pos[:,0]
            wind_data['target_Py_gse'] = synth_pos[:,1]
            wind_data['target_Pz_gse'] = synth_pos[:,2]
        wind_data.index = np.arange(len(wind_data)) #make the indices linear
        wind_data_cp = wind_data.copy() #Make a copy of the data to rescale
        if in_scaler is None:
            import joblib
            in_scaler = joblib.load('model_bin/in_scaler_005.pkl') #Load the default scaler
        wind_data_cp.loc[:,IN_KEYS] = in_scaler.transform(wind_data_cp.loc[:,IN_KEYS].to_numpy()) #Scale the input data
        inds = np.arange(WINDOW+STRIDE, len(IN_KEYS)) #Indices to start each time series at
        in_arr = np.zeros(( len(inds) , WINDOW, len(IN_KEYS))) #input array staging, Keras dataset order [samples, time window, features]
        flags = np.zeros(len(inds)) #Get a place to put fraction of input data that is interpolated
        for i in np.arange(len(inds)):
            in_arr[i, :, :] =  wind_data_cp.loc[(inds[i] - WINDOW - STRIDE):(inds[i] - STRIDE - 1)] #Get the timeseries that is 'window' long, 'stride' away from the desired time
            tdelt_max = 0
            tdelt = 0
            for n in wind_data_cp.loc[(inds[i] - WINDOW - STRIDE):(inds[i] - STRIDE - 1), 'interp_flag']:
                if (n == 1):
                    tdelt += 1
                    if (tdelt > tdelt_max):
                        tdelt_max = tdelt
                else:
                    tdelt = 0
            flags[i] = tdelt_max #Store the longest number of consecutive interpolated datapoints in the window
        return in_arr, flags





    
