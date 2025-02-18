import importlib.resources
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from sklearn.preprocessing import RobustScaler #RobustScaler is used to scale the input/target data but is not called directly below
import joblib

#MMS orbit that ends at bow shock nose stride*100s from the end of the window 
#(13.25RE, 0RE, 0RE) (from 2023-01-24 02:46:30+0000 - window - stride to 2023-01-24 02:46:30+0000 - stride)
SYNTH_XPOS = np.array([69215.97057508, 69480.44662705, 69706.40911294, 69969.18467343,
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
                       80473.4959342 , 80702.50977538]) #: Synthetic MMS-1 X position for prediction at bow shock
SYNTH_YPOS = np.array([-6242.531374  , -6141.43603983, -6054.73851884, -5953.54260001,
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
                       -1604.88808488, -1503.91065025]) #: Synthetic MMS-1 Y position for prediction at bow shock
SYNTH_ZPOS = np.array([1428.22663895, 1404.9257232 , 1384.88999865, 1361.43852111,
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
                       244.98385268,  215.52178328]) #: Synthetic MMS-1 Z position for prediction at bow shock
SYNTH_POS = np.array([SYNTH_XPOS, SYNTH_YPOS, SYNTH_ZPOS]).T #: Synthetic MMS-1 orbit for prediction at bow shock

class prime(ks.Model):
    def __init__(
            self, 
            model = None, 
            in_scaler = None, 
            tar_scaler = None, 
            in_keys = None, 
            tar_keys = None, 
            out_keys = None, 
            hps = [60, 15, 5.0/60.0]
        ):
        '''
        Class to wrap a keras model to be used with the SW-trained PRIME architecture.

        Parameters:
            model (keras model): Keras model to be used for prediction
            in_scaler (sklearn scaler): Scaler to be used for input data
            tar_scaler (sklearn scaler): Scaler to be used for target data
        '''
        super(prime, self).__init__()
        if in_scaler is None:
            resource_path = importlib.resources.path(prime, '/model_bin/primeinsc_v0.1.0.pkl')
            with resource_path as in_scaler_file:
                self.in_scaler = joblib.load(in_scaler_file)
        else:
            self.in_scaler = in_scaler
        if tar_scaler is None:
            resource_path = importlib.resources.path(prime, '/model_bin/primetarsc_v0.1.0.pkl')
            with resource_path as tar_scaler_file:
                self.tar_scaler = joblib.load(tar_scaler_file)
        else:
            self.tar_scaler = tar_scaler
        if in_keys is None:
            self.in_keys = [
                'B_xgsm', 
                'B_ygsm', 
                'B_zgsm', 
                'Vi_xgse', 
                'Vi_ygse', 
                'Vi_zgse', 
                'Ni', 
                'Vth', 
                'R_xgse', 
                'R_ygse', 
                'R_zgse', 
                'target_R_xgse', 
                'target_R_ygse', 
                'target_R_zgse',
            ] # Wind data keys to include in input dataset
        else:
            self.in_keys = in_keys
        if tar_keys is None:
            self.tar_keys = [
                'B_xgsm', 
                'B_ygsm', 
                'B_zgsm', 
                'Vi_xgse', 
                'Vi_ygse', 
                'Vi_zgse', 
                'Ne',
                ] # Targets from MMS dataset to match with input data
        else:
            self.tar_keys = tar_keys
        if out_keys is None:
            self.out_keys = [
                'B_xgsm', 
                'B_xgsm_sig', 
                'B_ygsm', 
                'B_ygsm_sig', 
                'B_zgsm', 
                'B_zgsm_sig', 
                'Vi_xgse', 
                'Vi_xgse_sig', 
                'Vi_ygse', 
                'Vi_ygse_sig', 
                'Vi_zgse', 
                'Vi_zgse_sig', 
                'Ne', 
                'Ne_sig',
                ] # Features in PRIME output (in general, tar_keys with 1sigma uncertainties denoted '_sig')
        else:
            self.out_keys = out_keys
        self.window = hps[0] # Input window length from hyperparameter list
        self.stride = hps[1] # Input stride length from hyperparameter list
        self.fraction = hps[2] # Input maximum tolerable fraction of interpolated data from hyperparameter list
        if model is None:
            self.model = self.build_model() # Instantiate model architecture with hyperparameters
            resource_path = importlib.resources.path(prime, '/model_bin/prime_v0.1.0.h5')
            with resource_path as model_weights_file:
                self.model.load_weights(model_weights_file)  # Load the saved weights
            self.model = model # Store in class
        else:
            self.model = model

    def predict(self, input = None, start = None, stop = None, pos = [13.25, 0, 0]):
        '''
        Generate prime predictions from input dataframes or time ranges.
        
        Parameters:
            input (dataframe, ndarray): Input data to be scaled and predicted
            start (string, optional): Start time of prediction (will use real data). Format 'YYYY-MM-DD HH:MM:SS'.
            stop (string, optional): Stop time of prediction (will use real data). Format 'YYYY-MM-DD HH:MM:SS'.
            pos (list, optional): Position propagated to if 'start' and 'stop' are specified.
        Returns:
            output (dataframe): Scaled output data
        '''
        if input is None:
            if (start is not None)&(stop is not None):
                input = self.build_real_input(start = start, stop = stop, pos = pos)
            else:
                raise RuntimeWarning('Must specify either input or (start and stop).')
                return None
        if isinstance(input, pd.DataFrame): # If input is a dataframe
            input_arr = input[self.in_keys].to_numpy() # Convert input dataframe to array
        if isinstance(input, np.ndarray): # If input is an array
            input_arr = input # Set input array to input
        output_arr = self.predict_raw(input_arr) # Predict with the keras model
        output = pd.DataFrame(output_arr, columns = self.out_keys) # Convert output array to dataframe
        output_epoch = input['Epoch'].to_numpy()[(self.window-1):] # Stage an epoch column to be added to the output dataframe
        output_epoch += pd.Timedelta(seconds = 100*self.stride) # Add lead time to the epoch column
        output['Epoch'] = output_epoch # Add the epoch column to the output dataframe
        return output
    def predict_raw(self, input):
        '''
        Wrapper function to predict with a keras model.
        '''
        input_scaled = self.in_scaler.transform(input) # Rescale the input data
        input_arr = np.zeros((len(input_scaled)-(self.window-1), self.window, len(self.in_keys))) # Reshape input data to be 3D
        for i in np.arange(len(input_scaled)-(self.window-1)):
            input_arr[i,:,:] = input_scaled[i:(i+self.window)] # Move the 55 unit window through the input data
        output_unscaled = self.model.predict(input_arr) # Use stored keras model to make prediction
        output = np.zeros((len(output_unscaled),len(self.out_keys))) #Stage output data to be 2x target dimensions (to account for uncertainties)
        output[:, ::2] = self.tar_scaler.inverse_transform(output_unscaled[:, ::2]) #Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_unscaled[:, ::2] + output_unscaled[:, 1::2]) - self.tar_scaler.inverse_transform(output_unscaled[:, ::2])) #Standard deviations
        return output
    def predict_grid(
        self,
        gridsize,
        x_extent,
        framenum,
        bx,
        by,
        bz,
        vx,
        vy,
        vz,
        ni,
        vt,
        rx,
        ry,
        rz,
        y_extent=None,
        z_extent=None,
        y = 0,
        z = 0,
        subtract_ecliptic=False,
    ):
        """
        Generate predictions from prime model on a grid of points.

        Parameters:
            gridsize (float): Spacing of grid points
            x_extent (list): Range of x values to calculate on
            framenum (int): Number of frames to calculate
            bx (float, array-like): IMF Bx value. If array like, must be of length framenum.
            by (float, array-like): IMF By value. If array like, must be of length framenum.
            bz (float, array-like): IMF Bz value. If array like, must be of length framenum.
            vx (float, array-like): Solar wind Vx value. If array like, must be of length framenum.
            vy (float, array-like): Solar wind Vy value. If array like, must be of length framenum.
            vz (float, array-like): Solar wind Vz value. If array like, must be of length framenum.
            ni (float, array-like): Solar wind ion density value. If array like, must be of length framenum.
            vt (float, array-like): Solar wind ion thermal speed value. If array like, must be of length framenum.
            rx (float, array-like): Wind spacecraft position x value. If array like, must be of length framenum.
            ry (float, array-like): Wind spacecraft position y value. If array like, must be of length framenum.
            rz (float, array-like): Wind spacecraft position z value. If array like, must be of length framenum.
            y_extent (list): Range of y values to calculate on. If None, z_extent must be specified.
            z_extent (list): Range of z values to calculate on. If None, y_extent must be specified.
            y (float, array-like): Y position that is held constant if y_extent is not specified. Default 0.
            z (float, array-like): Z position that is held constant if z_extent is not specified. Default 0.
            subtract_ecliptic (bool): Whether or not to subtract the Earth's motion in the ecliptic from Vy
        Returns:
            output_grid (ndarray): Array of predicted values on the grid. Shape (framenum, x_extent/gridsize, y_extent/gridsize, 18)
        """
        x_arr = np.arange(x_extent[0], x_extent[1], gridsize)  # Create a grid to calculate the solar wind conditions on
        y_arr = np.asarray([y]) # This array is overwritten if y_extent is specified
        z_arr = np.asarray([z]) # This array is overwritten if z_extent is specified
        if y_extent is None and z_extent is None:
            raise ValueError("Must specify y_extent or z_extent")
        if y_extent is not None:
            y_arr = np.arange(y_extent[0], y_extent[1], gridsize)  # Y positions to calculate the solar wind conditions on
        if z_extent is not None:
            z_arr = np.arange(z_extent[0], z_extent[1], gridsize)  # Z positions to calculate the solar wind conditions on
        x_grid, y_grid, z_grid = np.meshgrid(x_arr, y_arr, z_arr)  # Create a grid to calculate the solar wind conditions on
        input_seed = np.zeros((len(x_grid.flatten()) * framenum, len(self.in_keys)))  # Initialize array to hold the input data before unfolding it
        for idx, element in enumerate([bx, by, bz, vx, vy, vz, ni, vt, rx, ry, rz]):  # Loop through the input data and repeat it
            try:
                iter(element)  # Check if the element is iterable
                input_seed[:, idx] = np.repeat(element, len(x_grid.flatten()))  # If it is, repeat it for each grid point
            except TypeError:  # This error throws if iter(element) fails (i.e. element is not iterable)
                input_seed[:, idx] = np.repeat(element, framenum * len(x_grid.flatten()))  # If it isn't, repeat it for each grid point *and frame*
        loc_arr = np.zeros((len(x_grid.flatten()) * framenum, 3))  # Initialize array to hold the location data
        loc_arr[:, 0] = np.tile(x_grid.flatten(), framenum)
        loc_arr[:, 1] = np.tile(y_grid.flatten(), framenum)
        loc_arr[:, 2] = np.tile(z_grid.flatten(), framenum)
        input_seed_scaled = self.in_scaler.transform(input_seed)  # Scale the input data
        input_seed_scaled[:, 11:14] = self.loc_scaler.transform(loc_arr)  # Scale the location data
        input_seed_scaled = np.repeat(input_seed_scaled, self.window, axis=0)  # Repeat the input data 55 times to make static timeseries
        input_arr = input_seed_scaled.reshape(len(x_grid.flatten()) * framenum, self.window, len(self.in_keys))  # Reshape the input data into the correct shape
        output_arr = self.model.predict(input_arr)  # Predict the output data
        output = np.zeros((len(output_arr), len(self.out_keys)))  # Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_arr[:, ::2])  # Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_arr[:, ::2] + output_arr[:, 1::2]) - self.tar_scaler.inverse_transform(output_arr[:, ::2]))  # Standard deviations
        output_grid = output.reshape(framenum, len(y_arr), len(x_arr), len(z_arr), len(self.out_keys))  # Reshape the output data into the correct shape
        output_grid = np.swapaxes(output_grid, 1, 2)  # Move the y axis to the second axis (new order is frame, x, y, z, param)
        if subtract_ecliptic:  # If subtract_ecliptic is true, subtract the Earth's motion in the ecliptic from Vy
            output_grid[:, :, :, :, 8] -= 29.8
        return output_grid
    def build_model(self, units = [352, 192, 48, 48], activation = 'elu', dropout = 0.20, lr = 1e-4):
        '''
        Function to build keras model

        Parameters:
            units (list): Number of units in each layer of the model
            activation (str): Activation function to use in hidden layers
            dropout (float): Dropout rate to use in hidden layers
            lr (float): Learning rate to use in optimizer
        
        Returns:
            model (keras model): Keras model to be used for prediction (weights not initialized)
        '''
        model = ks.Sequential([ks.layers.GRU(units=units[0]),
                               ks.layers.Dense(units=units[1], activation=activation),
                               ks.layers.Dense(units=units[2], activation=activation),
                               ks.layers.Dense(units=units[3], activation=activation),
                               ks.layers.LayerNormalization(),
                               ks.layers.Dropout(dropout),
                               ks.layers.Dense(len(self.tar_keys),activation='linear')
                               ])
        model.compile(optimizer=tf.optimizers.Adamax(learning_rate=lr), loss=crps_loss)
        model.build(input_shape = (1, self.window, len(self.in_keys)))
        return model
    def build_synth_input(
            self,
            epoch = pd.to_datetime('1970-01-01 00:00:00+0000'),
            bx = 0, 
            by = 0, 
            bz = -5, 
            vx = -400, 
            vy = 0, 
            vz = 0, 
            ni = 5, 
            vt = 30, 
            rx = 200, 
            ry = 0, 
            rz = 0,
            tar_rx = 13.25,
            tar_ry = 0,
            tar_rz = 0,
            ):
        '''
        Builds a synthetic input array from user-specified quantities at L1.
        For input arrays made from measured data at L1, see build_real_input().
        
        Parameters:
            epoch (datetime): Datetime of start of input Dataframe.
            bx (float, array-like): IMF Bx value.
            by (float, array-like): IMF By value.
            bz (float, array-like): IMF Bz value.
            vx (float, array-like): Solar wind Vx value.
            vy (float, array-like): Solar wind Vy value.
            vz (float, array-like): Solar wind Vz value.
            ni (float, array-like): Solar wind ion density value.
            vt (float, array-like): Solar wind ion thermal speed value.
            rx (float, array-like): Wind spacecraft position x value.
            ry (float, array-like): Wind spacecraft position y value.
            rz (float, array-like): Wind spacecraft position z value.
            tar_rx (float, array-like): Propagation target position x value.
            tar_ry (float, array-like): Propagation target position y value.
            tar_rz (float, array-like): Propagation target position z value.
        
        Returns:
            input (Dataframe): Input dataframe suitable to predict from with self.predict(). 
        '''
        input = pd.DataFrame(columns = self.in_keys) #Initialize single-point input dataframe
        input['B_xgsm'] = bx*np.ones(self.window) #SW BX in nT (GSM coordinates)
        input['B_ygsm'] = by*np.ones(self.window) #SW BY in nT (GSM coordinates)
        input['B_zgsm'] = bz*np.ones(self.window) #SW BZ in nT (GSM coordinates)
        input['Vi_xgse'] = vx*np.ones(self.window) #SW X velocity in km/s (GSE coordinates)
        input['Vi_ygse'] = vy*np.ones(self.window) #SW Y velocity in km/s (GSE coordinates)
        input['Vi_zgse'] = vz*np.ones(self.window) #SW Z velocity in km/s (GSE coordinates)
        input['Ni'] = ni*np.ones(self.window) #SW density in cm^-3
        input['Vth'] = ni*np.ones(self.window) #SW thermal velocity in km/s
        input['R_xgse'] = rx*np.ones(self.window) #Wind position in X GSE (RE)
        input['R_ygse'] = ry*np.ones(self.window) #Wind position in Y GSE
        input['R_zgse'] = rz*np.ones(self.window) #Wind position in Z GSE

        #This is where the location you're propagating to is set.
        #Each entry in the "timeseries" is set to the same value, no need to make a fake MMS orbit.
        input['target_R_xgse'] = tar_rx*np.ones(self.window) #MMS position in X GSE (RE)
        input['target_R_ygse'] = tar_ry*np.ones(self.window) #MMS position in Y GSE
        input['target_R_zgse'] = tar_rz*np.ones(self.window) #MMS position in Z GSE

        #PRIME-SH also currently expects an 'Epoch' for its inputs.
        input['Epoch'] = pd.date_range(start=epoch, periods=self.window, freq='100s') - pd.Timedelta(seconds = 100*(self.window + self.stride))
        return input
    
    def build_real_input(self, start, stop, pos =[13.25, 0, 0], load_freq = '3M'):
        '''
        Load Wind spacecraft input data for PRIME in between specified date strings.

        Parameters:
            start (string): The start date of the data to load ('YYYY-MM-DD')
            end (string): The end date of the data to load ('YYYY-MM-DD')
            pos (list): Location of propagation in GSE coordinates (Earth Radii). Default [13.25, 0, 0].
            load_freq (string): Max length of data loaded by CdasWs. If throwing RuntimeError, try modifying this parameter. Default '3M' (three months).
        Return:
            input (Dataframe): Input dataframe suitable to predict from with self.predict(). 
        '''
        try:
            from cdasws import CdasWs
        except:
            raise RuntimeError('Unable to import CdasWs. Predicting using non-synthetic input requires CdasWs package (see https://cdaweb.gsfc.nasa.gov/WebServices/REST/py/cdasws/).')
        cdas = CdasWs() #Initialize CDAS WS Session
        mfi_df = pd.DataFrame([]) #Staging dataframe for Wind spacecraft Magnetic Field Investigation data
        try:
            data = cdas.get_data('WI_H0_MFI', ['BGSM', 'PGSE'], start, stop) #Load GSM B field and GSE SC position
            mfi_df['Epoch'] = data[1]['Epoch'] #MFI timestamps
            mfi_df['R_xgse'] = data[1]['PGSE'][:, 0] #Wind SC position
            mfi_df['R_ygse'] = data[1]['PGSE'][:, 1]
            mfi_df['R_zgse'] = data[1]['PGSE'][:, 2]
            mfi_df['B_xgsm'] = data[1]['BGSM'][:, 0] #GSM B field
            mfi_df['B_ygsm'] = data[1]['BGSM'][:, 1]
            mfi_df['B_zgsm'] = data[1]['BGSM'][:, 2]
        except TypeError: #Throws when date range is empty OR too big
            raise RuntimeError('CDASWS failed to load MFI data. Date range ('+start+' to '+stop+') may be too large or data may be missing.')
        mfi_df['Epoch'] = pd.to_datetime(mfi_df['Epoch'], utc=True) #Convert to UTC aware datetime
        #Set B values to nan if they are equal to the fill value of -1e31
        mfi_df['B_xgsm'].where(mfi_df['B_xgsm'] > -1e30, np.nan, inplace=True)
        mfi_df['B_ygsm'].where(mfi_df['B_ygsm'] > -1e30, np.nan, inplace=True)
        mfi_df['B_zgsm'].where(mfi_df['B_zgsm'] > -1e30, np.nan, inplace=True)
        #Set R values to nan if they are equal to the fill value of -1e31
        mfi_df['R_xgse'].where(mfi_df['R_xgse'] > -1e30, np.nan, inplace=True)
        mfi_df['R_ygse'].where(mfi_df['R_ygse'] > -1e30, np.nan, inplace=True)
        mfi_df['R_zgse'].where(mfi_df['R_zgse'] > -1e30, np.nan, inplace=True)
        swe_df = pd.DataFrame([]) #Staging dataframe for Wind spacecraft Solar Wind Experiment data
        try:
            data = cdas.get_data('WI_K0_SWE', ['Np', 'V_GSE', 'THERMAL_SPD', 'QF_V', 'QF_Np'], start, stop)
            swe_df['Epoch'] = data[1]['Epoch'] #SWE timestamps
            swe_df['Ni'] = data[1]['Np'] #Proton density (cm-3)
            swe_df['Vi_xgse'] = data[1]['V_GSE'][:, 0] #GSE flow velocity (km/s)
            swe_df['Vi_ygse'] = data[1]['V_GSE'][:, 1]
            swe_df['Vi_zgse'] = data[1]['V_GSE'][:, 2]
            swe_df['Vth'] = data[1]['THERMAL_SPD'] #SW thermal speed (km/s)
            swe_df['vflag'] = data[1]['QF_V'] #Velocity quality flag
            swe_df['niflag'] = data[1]['QF_Np'] #Density quality flag
        except TypeError: #Throws when date range is empty OR too big
            raise RuntimeError('CDASWS failed to load SWE data. Date range ('+start+' to '+stop+') may be too large or data may be missing.')
        swe_df['Epoch'] = pd.to_datetime(swe_df['Epoch'], utc=True) #Convert to UTC aware datetime
        #Remove erroneous Epochs outside downloaded date range (due to CDAS bug)
        swe_df['Epoch'].where(swe_df['Epoch'] >= pd.to_datetime(start, utc=True), np.nan, inplace=True)
        swe_df['Epoch'].where(swe_df['Epoch'] <= pd.to_datetime(stop, utc=True), np.nan, inplace=True)
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
        #Bin the data to 100s bins (default PRIME input cadence)
        bins = pd.date_range(start, stop, freq='100s')
        bins_index = pd.IntervalIndex.from_arrays(bins[:-1], bins[1:], closed='left') #Make interval index for binning
        swe_group = swe_df.groupby(pd.cut(swe_df['Epoch'], bins_index)) #Group the SWE and MFI data
        mfi_group = mfi_df.groupby(pd.cut(mfi_df['Epoch'], bins_index))
        swe_binned = swe_group.mean() #Take the mean in each group (bin)
        mfi_binned = mfi_group.mean()
        swe_binned.reset_index(drop=True, inplace=True) #Reset index to integers instead of group labels
        mfi_binned.reset_index(drop=True, inplace=True)
        input = pd.merge(swe_binned, mfi_binned, left_index = True, right_index = True) #Combine the SWE and MFI dataframes
        input['Epoch'] = bins[:-1] #The last bin gets dropped.
        input['flag'] = input.isna().any(axis=1) #Get the rows with NaNs and flag them as interpolated
        input['Ni'] = input['Ni'].interpolate(method='linear', axis=0) #Interpolate the data columns
        input['Vi_xgse'] = input['Vi_xgse'].interpolate(method='linear', axis=0)
        input['Vi_ygse'] = input['Vi_ygse'].interpolate(method='linear', axis=0)
        input['Vi_zgse'] = input['Vi_zgse'].interpolate(method='linear', axis=0)
        input['Vth'] = input['Vth'].interpolate(method='linear', axis=0)
        input['vflag'] = input['vflag'].interpolate(method='linear', axis=0)
        input['niflag'] = input['niflag'].interpolate(method='linear', axis=0)
        input['R_xgse'] = input['R_xgse'].interpolate(method='linear', axis=0)
        input['R_ygse'] = input['R_ygse'].interpolate(method='linear', axis=0)
        input['R_zgse'] = input['R_zgse'].interpolate(method='linear', axis=0)
        input['B_xgsm'] = input['B_xgsm'].interpolate(method='linear', axis=0)
        input['B_ygsm'] = input['B_ygsm'].interpolate(method='linear', axis=0)
        input['B_zgsm'] = input['B_zgsm'].interpolate(method='linear', axis=0)
        input['target_R_xgse'] = pos[0] #Insert propagation location
        input['target_R_ygse'] = pos[1]
        input['target_R_zgse'] = pos[2]
        input = input.dropna() #If a nan snuck in get it outta here!
        input = input.reset_index(drop=True) #Reset the index
        return input

    
#Custom loss function (Continuous Rank Probability Score) and associated helper functions

def crps_loss(y_true, y_pred):
    """
    Continuous rank probability score function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma^2 values of predicted distribution.
        
    Returns
    -------
    crps : tf.Tensor
        Continuous rank probability score.
    """
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6 = unstack_helper(y_true, y_pred)
    
    # CRPS (assuming gaussian distribution)
    crps0 = tf.math.reduce_mean(crps_f(ep_f(y_true0, mu0), sg0))
    crps1 = tf.math.reduce_mean(crps_f(ep_f(y_true1, mu1), sg1))
    crps2 = tf.math.reduce_mean(crps_f(ep_f(y_true2, mu2), sg2))
    crps3 = tf.math.reduce_mean(crps_f(ep_f(y_true3, mu3), sg3))
    crps4 = tf.math.reduce_mean(crps_f(ep_f(y_true4, mu4), sg4))
    crps5 = tf.math.reduce_mean(crps_f(ep_f(y_true5, mu5), sg5))
    crps6 = tf.math.reduce_mean(crps_f(ep_f(y_true6, mu6), sg6))
    
    # Average the continuous rank probability scores
    crps = (crps0 + crps1 + crps2 + crps3 + crps4 + crps5 + crps6) / 7.0
    
    return crps

def crps_f(ep, sg):
    '''
    Helper function that calculates continuous rank probability score
    '''
    crps = sg * ((ep/sg) * tf.math.erf((ep/(np.sqrt(2)*sg))) + tf.math.sqrt(2/np.pi) * tf.math.exp(-ep**2 / (2*sg**2)) - 1/tf.math.sqrt(np.pi))
    return crps

def ep_f(y, mu):
    '''
    Helper function that calculates epsilon (error) for CRPS
    '''
    ep = tf.math.abs(y - mu)
    return ep

def unstack_helper(y_true, y_pred):
    '''
    Helper function that unstacks the outputs and targets
    '''
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6 = tf.unstack(y_pred, axis=-1)
    
    # Separate the ground truth into each parameter
    y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6 = tf.unstack(y_true, axis=-1)
    
    # Add one dimension to make the right shape
    mu0 = tf.expand_dims(mu0, -1)
    sg0 = tf.expand_dims(sg0, -1)
    mu1 = tf.expand_dims(mu1, -1)
    sg1 = tf.expand_dims(sg1, -1)
    mu2 = tf.expand_dims(mu2, -1)
    sg2 = tf.expand_dims(sg2, -1)
    mu3 = tf.expand_dims(mu3, -1)
    sg3 = tf.expand_dims(sg3, -1)
    mu4 = tf.expand_dims(mu4, -1)
    sg4 = tf.expand_dims(sg4, -1)
    mu5 = tf.expand_dims(mu5, -1)
    sg5 = tf.expand_dims(sg5, -1)
    mu6 = tf.expand_dims(mu6, -1)
    sg6 = tf.expand_dims(sg6, -1)
    y_true0 = tf.expand_dims(y_true0, -1)
    y_true1 = tf.expand_dims(y_true1, -1)
    y_true2 = tf.expand_dims(y_true2, -1)
    y_true3 = tf.expand_dims(y_true3, -1)
    y_true4 = tf.expand_dims(y_true4, -1)
    y_true5 = tf.expand_dims(y_true5, -1)
    y_true6 = tf.expand_dims(y_true6, -1)
    return mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6
