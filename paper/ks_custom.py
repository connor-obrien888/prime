import numpy as np
import tensorflow as tf
from tensorflow import keras as ks
import keras_tuner as kt
from sklearn.preprocessing import RobustScaler
from scipy.signal import savgol_filter

class GaussianLayer(tf.keras.layers.Layer):
    def __init__(self, activation = 'sigmoid', relu_del = 0):
        self.activ = activation
        self.d = relu_del
        super(GaussianLayer, self).__init__()

    def call(self, inputs):
        # Separate the parameters into means and squared standard deviations
        mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7 = tf.unstack(inputs, axis=-1)

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
        mu7 = tf.expand_dims(mu7, -1)
        sg7 = tf.expand_dims(sg7, -1)

        # Apply a linear to pass the mean unmodified
        mu0 = tf.keras.activations.linear(mu0)
        mu1 = tf.keras.activations.linear(mu1)
        mu2 = tf.keras.activations.linear(mu2)
        mu3 = tf.keras.activations.linear(mu3)
        mu4 = tf.keras.activations.linear(mu4)
        mu5 = tf.keras.activations.linear(mu5)
        mu6 = tf.keras.activations.linear(mu6)
        mu7 = tf.keras.activations.linear(mu7)
        
        if (self.activ == 'softplus'):
            # Apply a softplus activation to bound standard deviation [0, inf)
            sg0 = tf.keras.activations.softplus(sg0)
            sg1 = tf.keras.activations.softplus(sg1)
            sg2 = tf.keras.activations.softplus(sg2)
            sg3 = tf.keras.activations.softplus(sg3)
            sg4 = tf.keras.activations.softplus(sg4)
            sg5 = tf.keras.activations.softplus(sg5)
            sg6 = tf.keras.activations.softplus(sg6)
            sg7 = tf.keras.activations.softplus(sg7)
        elif (self.activ == 'sigmoid'):
            # Apply a sigmoid activation to bound standard deviation [0, 1]
            sg0 = tf.keras.activations.sigmoid(sg0)
            sg1 = tf.keras.activations.sigmoid(sg1)
            sg2 = tf.keras.activations.sigmoid(sg2)
            sg3 = tf.keras.activations.sigmoid(sg3)
            sg4 = tf.keras.activations.sigmoid(sg4)
            sg5 = tf.keras.activations.sigmoid(sg5)
            sg6 = tf.keras.activations.sigmoid(sg6)
            sg7 = tf.keras.activations.sigmoid(sg7)
        elif (self.activ == 'relu'):
            # Apply a RELU activation to bound standard deviation [self.del, inf)
            sg0 = tf.keras.activations.relu(sg0) + self.d
            sg1 = tf.keras.activations.relu(sg1) + self.d
            sg2 = tf.keras.activations.relu(sg2) + self.d
            sg3 = tf.keras.activations.relu(sg3) + self.d
            sg4 = tf.keras.activations.relu(sg4) + self.d
            sg5 = tf.keras.activations.relu(sg5) + self.d
            sg6 = tf.keras.activations.relu(sg6) + self.d
            sg7 = tf.keras.activations.relu(sg7) + self.d
        else: #If unknown activation specified, default to sigmoid
            # Apply a sigmoid activation to bound standard deviation [0, 1]
            sg0 = tf.keras.activations.sigmoid(sg0)
            sg1 = tf.keras.activations.sigmoid(sg1)
            sg2 = tf.keras.activations.sigmoid(sg2)
            sg3 = tf.keras.activations.sigmoid(sg3)
            sg4 = tf.keras.activations.sigmoid(sg4)
            sg5 = tf.keras.activations.sigmoid(sg5)
            sg6 = tf.keras.activations.sigmoid(sg6)
            sg7 = tf.keras.activations.sigmoid(sg7)
        # Join back together again
        out_tensor = tf.concat([mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7], axis=-1)
        return out_tensor
    
def negative_gaussian_loss(y_true, y_pred):
    """
    Negative gaussian loss function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma^2 values of predicted distribution.
        
    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7 = unstack_helper(y_true, y_pred)
    
    # Calculate the negative log likelihood
    nll0 = tf.math.reduce_mean(tf.math.log(sg0) + tf.math.square(y_true0 - mu0) / sg0)
    nll1 = tf.math.reduce_mean(tf.math.log(sg1) + tf.math.square(y_true1 - mu1) / sg1)
    nll2 = tf.math.reduce_mean(tf.math.log(sg2) + tf.math.square(y_true2 - mu2) / sg2)
    nll3 = tf.math.reduce_mean(tf.math.log(sg3) + tf.math.square(y_true3 - mu3) / sg3)
    nll4 = tf.math.reduce_mean(tf.math.log(sg4) + tf.math.square(y_true4 - mu4) / sg4)
    nll5 = tf.math.reduce_mean(tf.math.log(sg5) + tf.math.square(y_true5 - mu5) / sg5)
    nll6 = tf.math.reduce_mean(tf.math.log(sg6) + tf.math.square(y_true6 - mu6) / sg6)
    nll7 = tf.math.reduce_mean(tf.math.log(sg7) + tf.math.square(y_true7 - mu7) / sg7)
    
    # Average the negative log likelihoods
    nll = (nll0 + nll1 + nll2 + nll3 + nll4 + nll5 + nll6 + nll7) / 8.0

    return nll

def gaussian_resample_loss(y_true, y_pred):
    """
    Random resampling loss function.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma^2 values of predicted distribution.
        
    Returns
    -------
    mse : tf.Tensor
        Mean square error.
    """
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7 = unstack_helper(y_true, y_pred)
    
    # Generate random samples from the distribution
    samp0 = (tf.random.normal(tf.shape(mu0)) * sg0) + mu0
    samp1 = (tf.random.normal(tf.shape(mu1)) * sg1) + mu1
    samp2 = (tf.random.normal(tf.shape(mu2)) * sg2) + mu2
    samp3 = (tf.random.normal(tf.shape(mu3)) * sg3) + mu3
    samp4 = (tf.random.normal(tf.shape(mu4)) * sg4) + mu4
    samp5 = (tf.random.normal(tf.shape(mu5)) * sg5) + mu5
    samp6 = (tf.random.normal(tf.shape(mu6)) * sg6) + mu6
    samp7 = (tf.random.normal(tf.shape(mu7)) * sg7) + mu7
    
    # Calculate the mean squared error
    mse0 = tf.math.reduce_mean(tf.math.square(y_true0 - samp0))
    mse1 = tf.math.reduce_mean(tf.math.square(y_true1 - samp1))
    mse2 = tf.math.reduce_mean(tf.math.square(y_true2 - samp2))
    mse3 = tf.math.reduce_mean(tf.math.square(y_true3 - samp3))
    mse4 = tf.math.reduce_mean(tf.math.square(y_true4 - samp4))
    mse5 = tf.math.reduce_mean(tf.math.square(y_true5 - samp5))
    mse6 = tf.math.reduce_mean(tf.math.square(y_true6 - samp6))
    mse7 = tf.math.reduce_mean(tf.math.square(y_true7 - samp7))
    
    mse = (mse0 + mse1 + mse2 + mse3 + mse4 + mse5 + mse6 + mse7) / 8

    return mse

def multisample(y_true, mu, sg):
    samp0 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp1 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp2 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp3 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp4 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp5 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp6 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp7 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp8 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    samp9 = (tf.random.normal(tf.shape(mu)) * sg) + mu
    mse0 = tf.math.reduce_mean(tf.math.square(y_true - samp0))
    mse1 = tf.math.reduce_mean(tf.math.square(y_true - samp1))
    mse2 = tf.math.reduce_mean(tf.math.square(y_true - samp2))
    mse3 = tf.math.reduce_mean(tf.math.square(y_true - samp3))
    mse4 = tf.math.reduce_mean(tf.math.square(y_true - samp4))
    mse5 = tf.math.reduce_mean(tf.math.square(y_true - samp5))
    mse6 = tf.math.reduce_mean(tf.math.square(y_true - samp6))
    mse7 = tf.math.reduce_mean(tf.math.square(y_true - samp7))
    mse8 = tf.math.reduce_mean(tf.math.square(y_true - samp8))
    mse9 = tf.math.reduce_mean(tf.math.square(y_true - samp9))
    mse = (mse0 + mse1 + mse2 + mse3 + mse4 + mse5 + mse6 + mse7) / 8
    return mse

def continuous_rank_prob(y_true, y_pred):
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
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7 = unstack_helper(y_true, y_pred)
    
    # CRPS (assuming gaussian distribution)
    crps0 = tf.math.reduce_mean(crps_f(ep_f(y_true0, mu0), sg0))
    crps1 = tf.math.reduce_mean(crps_f(ep_f(y_true1, mu1), sg1))
    crps2 = tf.math.reduce_mean(crps_f(ep_f(y_true2, mu2), sg2))
    crps3 = tf.math.reduce_mean(crps_f(ep_f(y_true3, mu3), sg3))
    crps4 = tf.math.reduce_mean(crps_f(ep_f(y_true4, mu4), sg4))
    crps5 = tf.math.reduce_mean(crps_f(ep_f(y_true5, mu5), sg5))
    crps6 = tf.math.reduce_mean(crps_f(ep_f(y_true6, mu6), sg6))
    crps7 = tf.math.reduce_mean(crps_f(ep_f(y_true7, mu7), sg7))
    
    # Average the continuous rank probability scores
    crps = (crps0 + crps1 + crps2 + crps3 + crps4 + crps5 + crps6 + crps7) / 8.0
    
    return crps

def accrue_loss(y_true, y_pred):
    """
    ACCRUE loss function for homoskedastic regression.
    Assumes tensorflow backend.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Errors between black box model and ground truth
    y_pred : tf.Tensor
        predicted variances
        
    Returns
    -------
    accrue : tf.Tensor
        ACCRUE loss score.
    """
    sig = tf.math.square(y_pred) #Square the predicted value to ensure positive value
    accrue = tf.reduce_mean(accrue_f(y_true, sig), axis = 1) #Calculate the ACCRUE cost function and average over the features
    return accrue

def accrue_f(ep, sg):
    '''
    Helper function that calculates ACCRUE cost function
    '''
    eta = ep / (np.sqrt(2)*sg)
    accrue = ksc.beta_f(ep, eta) * tf.math.reduce_mean(ksc.crps_f(ep, sg)) + (1 - ksc.beta_f(ep, eta)) * ksc.rs_f(eta)
    return accrue

def beta_f(ep, eta):
    '''
    Helper function that calculates scale factor beta
    '''
    beta = rs_min_f(eta) / (crps_min_f(ep) + rs_min_f(eta))
    return beta

def crps_f(ep, sg):
    '''
    Helper function that calculates continuous rank probability scores
    '''
    crps = sg * ((ep/sg) * tf.math.erf((ep/(np.sqrt(2)*sg))) + tf.math.sqrt(2/np.pi) * tf.math.exp(-ep**2 / (2*sg**2)) - 1/tf.math.sqrt(np.pi))
    return crps

def crps_min_f(ep):
    '''
    Helper function that calculates min continuous rank probability scores
    '''
    N = tf.cast(tf.size(ep), dtype = tf.float32)
    crps_min = (tf.math.sqrt(tf.math.log(4.0))/(2*N)) * tf.reduce_sum(ep)
    return crps_min

def rs_f(eta):
    '''
    Helper function that calculates reliability score
    '''
    ind = tf.cast(tf.argsort(eta)+1, dtype = tf.float32)
    N = tf.cast(tf.size(eta), dtype = tf.float32)
    rs = tf.reduce_sum((eta/N)*(tf.math.erf(eta)+1) - (eta/N**2)*(2*ind - 1) + tf.math.exp(-eta**2)/(tf.math.sqrt(np.pi)*N)) - 0.5*tf.math.sqrt(2/np.pi)
    return rs

def rs_min_f(eta):
    '''
    Helper function that calculates min reliability score
    '''
    N = tf.cast(tf.size(eta), dtype = tf.float32)
    ind = tf.cast(tf.range(N)+1, dtype = tf.float32)
    rs_min = (1/(tf.math.sqrt(np.pi)*N)) * tf.reduce_sum(tf.math.exp(-(tf.math.erf((2*ind - 1)/N - 1)**(-1))**2)) - 0.5*tf.math.sqrt(2/np.pi)
    return rs_min

def eta_f(y, mu, sg):
    '''
    Helper function that calculates eta for reliability score
    '''
    eta = ep_f(y, mu) / (np.sqrt(2)*sg)
    return eta

def ep_f(y, mu):
    '''
    Helper function that calculates epsilon for reliability score
    '''
    ep = tf.math.abs(y - mu)
    return ep

def unstack_helper(y_true, y_pred):
    '''
    Helper function that unstacks the outputs and targets
    '''
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7 = tf.unstack(y_pred, axis=-1)
    
    # Separate the ground truth into each parameter
    y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7 = tf.unstack(y_true, axis=-1)
    
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
    mu7 = tf.expand_dims(mu7, -1)
    sg7 = tf.expand_dims(sg7, -1)
    y_true0 = tf.expand_dims(y_true0, -1)
    y_true1 = tf.expand_dims(y_true1, -1)
    y_true2 = tf.expand_dims(y_true2, -1)
    y_true3 = tf.expand_dims(y_true3, -1)
    y_true4 = tf.expand_dims(y_true4, -1)
    y_true5 = tf.expand_dims(y_true5, -1)
    y_true6 = tf.expand_dims(y_true6, -1)
    y_true7 = tf.expand_dims(y_true7, -1)
    return mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7, y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7

def mse_metric(y_true, y_pred):
    """
    Mean squared error metric for use with GaussianLayer (mean-to-sample mse).
    NOT SUITABLE FOR USE AS LOSS CRITERION.
    
    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        mu and sigma^2 values of predicted distribution.
        
    Returns
    -------
    mse : tf.Tensor
        MSE between mu and y_true.
    """
    # Separate the parameters into means and squared standard deviations
    mu0, sg0, mu1, sg1, mu2, sg2, mu3, sg3, mu4, sg4, mu5, sg5, mu6, sg6, mu7, sg7 = tf.unstack(y_pred, axis=-1)
    
    # Separate the ground truth into each parameter
    y_true0, y_true1, y_true2, y_true3, y_true4, y_true5, y_true6, y_true7 = tf.unstack(y_true, axis=-1)
    
    # Add one dimension to make the right shape
    mu0 = tf.expand_dims(mu0, -1)
    mu1 = tf.expand_dims(mu1, -1)
    mu2 = tf.expand_dims(mu2, -1)
    mu3 = tf.expand_dims(mu3, -1)
    mu4 = tf.expand_dims(mu4, -1)
    mu5 = tf.expand_dims(mu5, -1)
    mu6 = tf.expand_dims(mu6, -1)
    mu7 = tf.expand_dims(mu7, -1)
    y_true0 = tf.expand_dims(y_true0, -1)
    y_true1 = tf.expand_dims(y_true1, -1)
    y_true2 = tf.expand_dims(y_true2, -1)
    y_true3 = tf.expand_dims(y_true3, -1)
    y_true4 = tf.expand_dims(y_true4, -1)
    y_true5 = tf.expand_dims(y_true5, -1)
    y_true6 = tf.expand_dims(y_true6, -1)
    y_true7 = tf.expand_dims(y_true7, -1)
    
    # Calculate the MSE
    mse0 = tf.math.reduce_mean(tf.math.square(y_true0 - mu0))
    mse1 = tf.math.reduce_mean(tf.math.square(y_true1 - mu1))
    mse2 = tf.math.reduce_mean(tf.math.square(y_true2 - mu2))
    mse3 = tf.math.reduce_mean(tf.math.square(y_true3 - mu3))
    mse4 = tf.math.reduce_mean(tf.math.square(y_true4 - mu4))
    mse5 = tf.math.reduce_mean(tf.math.square(y_true5 - mu5))
    mse6 = tf.math.reduce_mean(tf.math.square(y_true6 - mu6))
    mse7 = tf.math.reduce_mean(tf.math.square(y_true7 - mu7))
    
    # Average the MSEs
    mse = (mse0 + mse1 + mse2 + mse3 + mse4 + mse5 + mse6 + mse7) / 8.0
    return mse

class dataset():
    def __init__(self, tar_full, in_full, tar_scaler, in_scaler, tar_train, in_train, tar_test, in_test, inds_train, inds_test, ds_mask, in_keys, tar_keys):
        self.tar_full = tar_full
        self.in_full = in_full
        self.tar_scaler = tar_scaler
        self.in_scaler = in_scaler
        self.tar_train = tar_train
        self.in_train = in_train
        self.tar_test = tar_test
        self.in_test = in_test
        self.inds_train = inds_train
        self.inds_test = inds_test
        self.ds_mask = ds_mask
        self.in_keys = in_keys
        self.tar_keys = tar_keys

    def get_train(self):
        return self.in_train, self.tar_train

    def get_test(self):
        return self.in_test, self.tar_test

    def get_full(self):
        return self.in_full, self.tar_full

    def get_inds(self):
        return self.inds_train, self.inds_test

    def get_mask(self):
        return self.ds_mask

    def get_data(self):
        in_data = self.in_scaler.inverse_transform(self.in_full)
        tar_data = self.tar_scaler.inverse_transform(self.tar_full)
        return in_data, tar_data


def load_dataset(mms_store, wind_store, region, in_keys, tar_keys, split_frac, window, stride, inter_frac, flag, in_storekey = 'inputs', tar_storekey = 'targets', conesize = np.pi/2, tar_scaler = RobustScaler(), in_scaler = RobustScaler(), vx_cut = True, table_cut = False):
    '''
    Helper function that loads a dataset from a file.
    '''
    tar_raw = pd.read_hdf(mms_store, key = tar_storekey, mode = 'a') #Load the target data
    tar_raw = nightside_cut(tar_raw, conesize=conesize) #Cut out the nightside data where classification is not possible
    if vx_cut:
        tar_raw = tar_raw[tar_raw['Vi_xgse'] <= -250] #Cut out erroneous super slow solar wind (maybe it's the magnetosheath?)
    if region == 'ms':
        tar_raw = tar_raw.drop(tar_raw[tar_raw['region']!=0].index).dropna() #Magnetosphere data
    elif region == 'sh':
        tar_raw = tar_raw.drop(tar_raw[tar_raw['region']!=1].index).dropna() #Sheath data
    elif region == 'sw':
        tar_raw = tar_raw.drop(tar_raw[tar_raw['region']!=2].index).dropna() #Solar Wind Data
        if table_cut:
            list = pd.read_hdf('../data/sw_table_list.h5', key = 'sw_table_list',  mode = 'a')
            tar_raw = tar_raw.loc[tar_raw['Epoch'].round('2H').isin(list)]
    elif region == 'if':
        tar_raw = tar_raw.drop(tar_raw[tar_raw['region']!=3].index).dropna() #Ion Foreshock Data
    else:
        tar_raw = tar_raw.dropna() #The entire dataset
    in_raw = pd.read_hdf(wind_store, key = in_storekey, mode = 'a') #Load the input data
    in_ind = closest_argmin(tar_raw['Epoch'].to_numpy(), in_raw['Epoch'].to_numpy()) #Get the indices of the input dataset closest to each target time
    tar_full, in_full, tar_scaler, in_scaler, ds_mask = ds_constructor(tar_raw, in_raw, in_ind, in_keys, tar_keys, window = window, stride = stride, inter_thresh = inter_frac, tar_scaler = tar_scaler, in_scaler = in_scaler, return_mask = True, flag = flag) #Scale, stride, window the datasets
    in_train, in_test = chunker(in_full, window*2, split_frac) #Divide the input datasets into independent chunks
    tar_train, tar_test= chunker(tar_full, window*2, split_frac) #Divide the target datasets into independent chunks, save indices so we can do comparison/validation
    inds_train, inds_test = chunker(np.arange(len(tar_full)), window*2, split_frac)
    return dataset(tar_full, in_full, tar_scaler, in_scaler, tar_train, in_train, tar_test, in_test, inds_train, inds_test, ds_mask, in_keys, tar_keys)

def nightside_cut(mms_data, conesize = 2, x_key = 'R_xgse', y_key = 'R_ygse', z_key = 'R_zgse'):
    '''
    Helper function that cuts out mms data outside a cone conesize radians away from the GSE X axis.
    '''
    mms_data_new = mms_data[np.arctan2(np.sqrt(mms_data[y_key]**2 + mms_data[z_key]**2) , mms_data[x_key]) <= conesize]
    mms_data_new.index = np.arange(len(mms_data_new))
    return mms_data_new

def closest_argmin(A, B):
    '''
    Helper function that returns indices of elements in array B closest to each element in array A.
    '''
    L = B.size
    sidx_B = B.argsort()
    sorted_B = B[sidx_B]
    sorted_idx = np.searchsorted(sorted_B, A)
    sorted_idx[sorted_idx==L] = L-1
    mask = (sorted_idx > 0) & \
    ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )
    return sidx_B[sorted_idx-mask]

def input_window(input_data, inds, in_keys, window, stride, flag = 'percent'):
    '''
    Helper function that splits time series input data into windows

    Parameters
    ----------
    input_data : float, array-like
        Dataframe of rescaled input data.
    inds : float, array-like
        Array of indices in input_data corresponding to the start times of each target.
    in_keys : list
        List of keys for input data.
    window : int
        Window size in 100s entries.
    stride : int
        Stride between end of window and time of prediction in 100s entries.
    flag : str, optional
        What type of interpolation flag to return with the windowed array. 'percent'
        yields percent of data in window is interpolated. 'tdelt' is longest stretch
        of interpolated data in input window (in indices). Default 'percent'.

    Returns
    -------
    in_arr : float, array-like
        Input array of windows of input_data.
    inter_flags : float, array-like
        Array of percentages of each window that are interpolated data.

    '''
    in_arr = np.zeros(( len(inds) , window, len(in_keys) )) #input array staging, Keras dataset order [samples, time window, features]
    inter_flags = np.zeros(len(inds)) #Get a place to put fraction of input data that is interpolated
    for i in np.arange(len(inds)):
        in_arr[i, :, :] =  input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), in_keys] #Get the timeseries that is 'window' long, 'stride' away from the MMS target time
        if (flag == 'percent'): #Percent of input window that is interpolated
            inter_flags[i] = np.sum(input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), 'interp_flag'])/window #calculate percentage of data that is interpolated
        if (flag == 'tdelt'): #Longest number of consecutive interpolated datapoints in window
            tdelt_max = 0
            tdelt = 0
            for n in input_data.loc[(inds[i] - window - stride):(inds[i] - stride - 1), 'interp_flag']:
                if (n == 1):
                    tdelt += 1
                    if (tdelt > tdelt_max):
                        tdelt_max = tdelt
                else:
                    tdelt = 0
            inter_flags[i] = tdelt_max     
    return in_arr, inter_flags

def ds_constructor(target_data, input_data, inds, in_keys, tar_keys, window = 140, stride = 1, inter_thresh = 0.5, tar_scaler = RobustScaler(), in_scaler = RobustScaler(), night_cut = False, return_mask = False, flag = 'percent'):
    '''
    Helper function that constructs keras datasets from target and input Dataframes.

    Parameters
    ----------
    target_data : float, array-like
        Dataframe of unscaled target data.
    input_data : float, array-like
        Dataframe of rescaled input data.
    inds : float, array-like
        Array of indices in input_data corresponding to the start times of each target.
    in_keys : list
        List of input keys to use in input_data.
    tar_keys : list
        List of target keys to use in target_data.
    window : int, optional
        Window size in 100s entries. Default 140
    stride : int, optional
        Stride between end of window and time of prediction in 100s entries. Default 1
    inter_thresh : float, optional
        Fraction of interpolated data that is acceptable to include in input window. 
        Default 0.5
    tar_scaler : Scaler, optional
        Instance of a Scaler class from module sklearn.preprocessing._data.
        Default skl.preprocessing.RobustScaler().
    in_scaler : Scaler, optional
        Instance of a Scaler class from module sklearn.preprocessing._data.
        Default skl.preprocessing.RobustScaler().
    night_cut : bool, optional
        Do we want to cut out nightside data? Recommended for solar wind data.
        Default False.
    return_mask : bool, optional
        Do we want to return a mask of what data is below inter_thresh? Useful for reconstructing datasets.
        Default False.

    Returns
    -------
    tar_ds : float, array-like
        Target dataset rescaled target_data.
    in_ds : float, array-like
        Input dataset of rescaled windows of input_data.
    tar_tf : Scaler
        Scaler to invert scaled target data.
    in_tf : Scaler
        Scaler to invert scaled input data.
    '''
    #Copy the DataFrames for safety
    target_data_cp = target_data.copy()
    input_data_cp = input_data.copy()
    
    #Rescale here
    tar_tf = tar_scaler.fit(target_data_cp.loc[:,tar_keys].to_numpy()) #This rescales data to inter-quartile range to reduce outlier sensitivity
    in_tf = in_scaler.fit(input_data_cp.loc[:,in_keys].to_numpy())
    target_data_cp.loc[:,tar_keys] = tar_tf.transform(target_data_cp.loc[:,tar_keys].to_numpy()) #Look, its pretty annoying that this is the syntax for this operation, but here we are
    input_data_cp.loc[:,in_keys] = in_tf.transform(input_data_cp.loc[:,in_keys].to_numpy())
    
    #Target array
    tar_arr = target_data_cp.loc[:, tar_keys].to_numpy() #Target array is a subset of the MMS data (just parameters)
    
    #Input Array
    in_arr, inter_flags = input_window(input_data_cp, inds, in_keys, window, stride, flag = flag)
    
    mask = (inter_flags < inter_thresh) #Mask out data thats above the inter-thresh
    tar_ds = tar_arr[mask, :] #Only keep points that have a good fraction of real, non-interpolated data
    in_ds = in_arr[mask, :, :]
    
    if return_mask:
        return tar_ds, in_ds, tar_tf, in_tf, mask
    else:
        return tar_ds, in_ds, tar_tf, in_tf


def chunker(A, n, f, return_inds = False):
    '''
    Helper function that splits array into chunks and assigns them to two datasets.

    Parameters
    ----------
    A : float, array-like
        Array to be split along axis 0.
    n : int
        Length of each chunk to be split
    f : float
        Fraction of data to end up in the smaller (test) array
    return_inds : bool, optional
        Do we want to return the locations of the train data? Useful for reconstructing datasets.
        Default False.
    Returns
    -------
    tar_ds : float, array-like
        Target dataset rescaled target_data.
    in_ds : float, array-like
        Input dataset of rescaled windows of input_data.
    '''
    k = int(1/f)
    A_tmp = np.copy(np.array_split(A, len(A)//(n-1), axis=0))
    A_test = np.concatenate(A_tmp[::k])
    index = np.ones(len(A_tmp), dtype = bool)
    index[::k] = False
    A_train = np.concatenate(A_tmp[index])
    if return_inds:
        return A_train, A_test, index
    else:
        return A_train, A_test

def despike(data, window = 13, order = 3, threshold = 1):
    '''
    Simple despike routine that linearly interpolates over data that spike above some threshold.
    
    Parameters
    ----------
    data : float, array-like
        DataFrame of data to be despiked 
    window : int, optional
        Length of window for Savitzky Golay filter, default 13
    order : int, optional
        Order of Savitzky Golay filter, default 3
    threshold : int, optional
        Size of spike to be removed in standard deviations, default 1
    Returns
    -------
    data_smooth : float, array-like
        Despiked version of data.
    '''
    noise = np.abs(data - savgol_filter(data, window, order))
    thresh = threshold*np.std(data)

    data_smooth = data.copy()
    data_smooth[noise>thresh] = np.nan
    data_smooth = data_smooth.interpolate()
    return data_smooth

def prediction(model_name, in_data, datapath = '/projectnb/sw-prop/obrienco/', window = True, batch_size = None, WINDOW = 200, STRIDE = 30):
    '''
    
    '''
    
    #Load the models and scalers
    model = ks.models.load_model(datapath+'models/'+model_name, custom_objects = {'negative_gaussian_loss': negative_gaussian_loss})
    in_sc = skl.externals.joblib.load(datapath+'models/'+model_name+'_iscale') #Load in the scalers for the target and input data
    tar_sc = skl.externals.joblib.load(datapath+'models/'+model_name+'_tscale')
    
    in_data_cp = in_data.copy()
    in_data_cp.iloc[:,1:15] = in_sc.transform(in_data_cp.iloc[:,1:15]) #Rescale the data
    if window: #Do we need to split the dataset into chunks?
        inds = np.arange(WINDOW+STRIDE, len(in_data_cp)) #Indices to start each time series at
        in_data_arr, flags = input_window(in_data_cp, inds, WINDOW, STRIDE) #Break the array into input timeseries
    else:
        in_data_arr = in_data_arr_cp
        flags = -1 * np.ones(len(ind_data_arr))
    predict_arr = np.zeros((len(in_data_arr), 16)) #Stage the prediction array
    
    if batch_size is not None: #If we were passed a batch size, break prediction inbto chunks
        for i in np.arange(len(predict_arr)//batch_size+1):
            predict_arr[i*batch_size:(i+1)*batch_size, :] = model.predict(in_data_arr[i*batch_size:(i+1)*batch_size, :, :])
            print(str(100*i/(len(predict_arr)//batch_size))[0:5]+'% Complete', end = '\r')
    else: #Otherwise, just do it all at once
        predict_arr = model.predict(in_data_arr)
    #Stage the dataframe to put the data in
    predict = pd.DataFrame(columns = ['time', 'Bx_gsm', 'Bx_gsm_sig', 'By_gsm', 'By_gsm_sig', 'Bz_gsm', 'Bz_gsm_sig', 'Vx_gse', 'Vx_gse_sig', 'Vy_gse', 'Vy_gse_sig', 'Vz_gse', 'Vz_gse_sig', 'n_i', 'n_i_sig', 'T_i', 'T_i_sig', 'interp_frac'])
    predict['time'] = in_data_cp['time'][(WINDOW+STRIDE):]
    predict['interp_frac'] = flags
    predict[['Bx_gsm', 'By_gsm', 'Bz_gsm', 'Vx_gse', 'Vy_gse', 'Vz_gse', 'n_i', 'T_i']] = tar_sc.inverse_transform(predict_arr[:, ::2])
    predict[['Bx_gsm_sig', 'By_gsm_sig', 'Bz_gsm_sig', 'Vx_gse_sig', 'Vy_gse_sig', 'Vz_gse_sig', 'n_i_sig', 'T_i_sig']] = np.abs(tar_sc.inverse_transform(predict_arr[:, ::2] + predict_arr[:, 1::2]) - tar_sc.inverse_transform(predict_arr[:, ::2]))
    return predict