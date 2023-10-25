import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from sklearn.preprocessing import RobustScaler #RobustScaler is used to scale the input/target data but is not called directly below
import joblib

#MMS orbit that ends at bow shock nose stride minutes from the end of the window 
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
                       80473.4959342 , 80702.50977538])
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
                       -1604.88808488, -1503.91065025])
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
                       244.98385268,  215.52178328])
SYNTH_POS = np.array([SYNTH_XPOS, SYNTH_YPOS, SYNTH_ZPOS]).T

class prime(ks.Model):
    def __init__(self, model = None, in_scaler = None, tar_scaler = None, in_keys = None, tar_keys = None, out_keys = None, hps = [60, 15, 5.0/60.0]):
        '''
        Class to wrap a keras model to be used with the SH dataset.

        Parameters:
            model (keras model): Keras model to be used for prediction
            in_scaler (sklearn scaler): Scaler to be used for input data
            tar_scaler (sklearn scaler): Scaler to be used for target data
        '''
        super(prime, self).__init__()
        if model is None:
            self.model = self.build_model()
            self.model.load_weights('./model_bin/prime_v0.1.0.h5')
            self.model = model
        else:
            self.model = model
        if in_scaler is None:
            self.in_scaler = joblib.load('./model_bin/primeinsc_v0.1.0.pkl')
        else:
            self.in_scaler = in_scaler
        if tar_scaler is None:
            self.tar_scaler = joblib.load('./model_bin/primetarsc_v0.1.0.pkl')
        else:
            self.tar_scaler = tar_scaler
        if in_keys is None:
            self.in_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ni', 'Vth', 'R_xgse', 'R_ygse', 'R_zgse', 'target_R_xgse', 'target_R_ygse', 'target_R_zgse'] #Wind data keys to include in input dataset
        else:
            self.in_keys = in_keys
        if tar_keys is None:
            self.tar_keys = ['B_xgsm', 'B_ygsm', 'B_zgsm', 'Vi_xgse', 'Vi_ygse', 'Vi_zgse', 'Ne'] #Targets from MMS dataset to match with input data
        else:
            self.tar_keys = tar_keys
        if out_keys is None:
            self.out_keys = ['B_xgsm', 'B_xgsm_sig', 'B_ygsm', 'B_ygsm_sig', 'B_zgsm', 'B_zgsm_sig', 'Vi_xgse', 'Vi_xgse_sig', 'Vi_ygse', 'Vi_ygse_sig', 'Vi_zgse', 'Vi_zgse_sig', 'Ne', 'Ne_sig']
        else:
            self.out_keys = out_keys
        self.window = hps[0]
        self.stride = hps[1]
        self.fraction = hps[2]
    def predict(self, input):
        '''
        High-level wrapper function to generate prime predictions from input dataframes.
        
        Parameters:
            input (dataframe, ndarray): Input data to be scaled and predicted
        Returns:
            output (dataframe): Scaled output data
        '''
        if isinstance(input, pd.DataFrame): #If input is a dataframe
            input_arr = input[self.in_keys].to_numpy() #Convert input dataframe to array
        if isinstance(input, np.ndarray): #If input is an array
            input_arr = input #Set input array to input
        output_arr = self.predict_raw(input_arr) #Predict with the keras model
        output = pd.DataFrame(output_arr, columns = self.out_keys) #Convert output array to dataframe
        output_epoch = input['Epoch'].to_numpy()[(self.window-1):] #Stage an epoch column to be added to the output dataframe
        output_epoch += pd.Timedelta(seconds = 100*self.stride) #Add lead time to the epoch column
        output['Epoch'] = output_epoch #Add the epoch column to the output dataframe
        return output
    def predict_raw(self, input):
        '''
        Wrapper function to predict with a keras model.
        '''
        input_scaled = self.in_scaler.transform(input)
        input_arr = np.zeros((len(input_scaled)-(self.window-1), self.window, len(self.in_keys))) #Reshape input data to be 3D
        for i in np.arange(len(input_scaled)-(self.window-1)):
            input_arr[i,:,:] = input_scaled[i:(i+self.window)] #Move the 55 unit window through the input data
        output_unscaled = self.model.predict(input_arr)
        output = np.zeros((len(output_unscaled),len(self.out_keys))) #Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_unscaled[:, ::2]) #Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_unscaled[:, ::2] + output_unscaled[:, 1::2]) - self.tar_scaler.inverse_transform(output_unscaled[:, ::2])) #Standard deviations
        return output
    def predict_grid(self, gridsize, x_extent, y_extent, framenum, bx, by, bz, vx, vy, vz, ni, vt, rx, ry, rz):
        '''
        Generate predictions from prime model on a grid of points.

        Parameters:
            gridsize (float): Spacing of grid points
            x_extent (list): Range of x values to calculate on
            y_extent (list): Range of y values to calculate on
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
        Returns:
            output_grid (ndarray): Array of predicted values on the grid. Shape (framenum, x_extent/gridsize, y_extent/gridsize, 18)
        '''
        x_arr = np.arange(x_extent[0], x_extent[1], gridsize) #Create a grid to calculate the magnetosheath conditions on
        y_arr = np.arange(y_extent[0], y_extent[1], gridsize) #Create a grid to calculate the magnetosheath conditions on
        x_grid, y_grid = np.meshgrid(x_arr, y_arr) #Create a grid to calculate the magnetosheath conditions on
        input_seed = np.zeros((len(x_grid.flatten())*framenum, len(self.in_keys))) #Initialize array to hold the input data before unfolding it
        for idx, element in enumerate([bx, by, bz, vx, vy, vz, ni, vt, rx, ry, rz]): #Loop through the input data and repeat it
            try:
                iter(element) #Check if the element is iterable
                input_seed[:, idx] = np.repeat(element, len(x_grid.flatten())) #If it is, repeat it for each grid point
            except TypeError:
                input_seed[:, idx] = np.repeat(element, framenum*len(x_grid.flatten())) #If it isn't, repeat it for each grid point *and frame*
        input_seed[:, 11] = np.tile(x_grid.flatten(), framenum)
        input_seed[:, 12] = np.tile(y_grid.flatten(), framenum)
        input_seed[:, 13] = 0 #Set target z to 0
        input_seed_scaled = self.in_scaler.transform(input_seed) #Scale the input data
        input_seed_scaled = np.repeat(input_seed_scaled, self.window, axis = 0) #Repeat the input data 55 times to make static timeseries
        input_arr = input_seed_scaled.reshape(len(x_grid.flatten())*framenum, self.window, len(self.in_keys)) #Reshape the input data into the correct shape
        output_arr = self.model.predict(input_arr) #Predict the output data
        output = np.zeros((len(output_arr),len(self.out_keys))) #Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_arr[:, ::2]) #Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_arr[:, ::2] + output_arr[:, 1::2]) - self.tar_scaler.inverse_transform(output_arr[:, ::2])) #Standard deviations
        output_grid = output.reshape(framenum, len(y_arr), len(x_arr), len(self.out_keys)) #Reshape the output data into the correct shape
        return output_grid
    def load_weights(self, modelpath, scalerpath):
        '''
        Wrapper function to load saved keras model and scalers
        
        Parameters:
            modelpath (str): Path to saved keras model
            scalerpath (str): Path to saved scalers
        '''
        self.model.load_weights(modelpath)
        self.in_scaler = joblib.load(scalerpath + 'in_scaler.pkl')
        self.tar_scaler = joblib.load(scalerpath + 'tar_scaler.pkl')
    def save_weights(self, modelpath, scalerpath):
        '''
        Wrapper function to save keras model and scalers

        Parameters:
            modelpath (str): Path to save keras model
            scalerpath (str): Path to save scalers
        '''
        self.model.save_weights(modelpath)
        joblib.dump(self.in_scaler, scalerpath + 'in_scaler.pkl')
        joblib.dump(self.tar_scaler, scalerpath + 'tar_scaler.pkl')
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
