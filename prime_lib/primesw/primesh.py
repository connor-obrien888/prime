import importlib.resources
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as ks
from sklearn.preprocessing import RobustScaler #RobustScaler is used to scale the input/target data but is not called directly below
import joblib
from .prime import prime, crps_loss, mse_metric

__all__ = ['primesh', 'jelinek_bs', 'shue_mp'] # Here __all__ is defined so that docs tools can read the appropriate docstrings

class primesh(prime):
    '''
        Class to wrap a keras model to be used with the Sheath-trained PRIME-SH architecture (for solar wind prediction see `primesw.prime`). It is recommended to instantiate `primesh` objects in their default configuration:
        ```
        import primesw as psw
        propagator = psw.primesh()
        ```
        Users will most likely use this class primarily for its `primesh.predict` method. This method is inherited from `primesw.prime` and thus operates the same way.
        To generate solar wind predictions from Wind spacecraft data, specify `start` and `stop` times for the desired prediction.
        `start` and `stop` are strings with format 'YYYY-MM-DD HH:MM:SS'.
        ```
        import primesw as psw
        propagator = psw.primesh()
        propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00')
        ```
        If using data from an L1 monitor to make predictions, pass the input data using `input` argument.
        If `input` is specified, `start` and `stop` should not be (and vice versa).
        `input` is also useful for making predicitons from synthetic solar wind data (see inherited method `primesw.prime.build_synth_input`).
        For instance, one can predict what the magnetosheath conditions at the magnetopause nose would be if the solar wind flow at L1 was 700km/s:
        ```
        import primesw as psw
        propagator = psw.primesh()
        propagator.predict(input = propagator.build_synth_input(vx=-700))
        ```
        By default, predictions are made at the average middle of Earth's magnetosheath 12.25 Earth Radii upstream on the Geocentric Solar Ecliptic (GSE) x-axis.
        One can also specify a position to propagate to besides the default by specifying `pos`:
        ```
        import primesw as psw
        propagator = psw.primesh()
        propagator.predict(start = '2020-01-01 00:00:00', stop = '2020-01-02 00:00:00', pos = [11.25, 5, 0])
        ```
        All positions are in GSE coordinates with units of Earth Radii.
        It is not recommended to make predictions outside of the region PRIME-SH was trained on (within 30 Earth radii of the Earth on the dayside).

        When instantiating a `primesh` object, one can specify a predefined `model` to be used instead of the automatically-loaded PRIME-SH model. 
        In that case, the scaling functions for the input and target datasets (`in_scaler` and `tar_scaler`), the input and target features (`in_keys` and `tar_keys`), and the output features (`out_keys`) must be specified.
        The full list of arguments that can be passed to `primesh` is given below.
    '''
    def __init__(self, model = None, in_scaler = None, tar_scaler = None, loc_scaler = None, in_keys = None, tar_keys = None, out_keys = None, hps = [55, 18, 0.05]):
        '''
        `hps` is an array of dataset-pertinent hyperparameters. The three elements correspond to `window`, `input`, and `stride`:
        '''
        super(primesh, self).__init__()
        if in_scaler is None:
            resource_path = importlib.resources.path('primesw', 'primeshinsc_v0.1.0.pkl')
            with resource_path as in_scaler_file:
                self.in_scaler = joblib.load(in_scaler_file)
        else:
            self.in_scaler = in_scaler
        if tar_scaler is None:
            resource_path = importlib.resources.path('primesw', 'primeshtarsc_v0.1.0.pkl')
            with resource_path as tar_scaler_file:
                self.tar_scaler = joblib.load(tar_scaler_file)
        else:
            self.tar_scaler = tar_scaler
        if loc_scaler is None:
            resource_path = importlib.resources.path('primesw', 'primeshlocsc_v0.1.0.pkl')
            with resource_path as loc_scaler_file:
                self.loc_scaler = joblib.load(loc_scaler_file)
        else:
            self.loc_scaler = loc_scaler
        if in_keys is None:
            self.in_keys = [
                "B_xgsm",
                "B_ygsm",
                "B_zgsm",
                "Vi_xgse",
                "Vi_ygse",
                "Vi_zgse",
                "Ni",
                "Vth",
                "R_xgse",
                "R_ygse",
                "R_zgse",
                "target_R_xgse",
                "target_R_ygse",
                "target_R_zgse",
            ]  # Wind data keys to include in input dataset
        else:
            self.in_keys = in_keys
        if tar_keys is None:
            self.tar_keys = [
                "B_xgsm",
                "B_ygsm",
                "B_zgsm",
                "Vi_xgse",
                "Vi_ygse",
                "Vi_zgse",
                "Ni",
                "Tipar",
                "Tiperp",
            ]  # Targets from MMS dataset to match with input data
        else:
            self.tar_keys = tar_keys
        if out_keys is None:
            self.out_keys = [
                "B_xgsm",
                "B_xgsm_sig",
                "B_ygsm",
                "B_ygsm_sig",
                "B_zgsm",
                "B_zgsm_sig",
                "Vi_xgse",
                "Vi_xgse_sig",
                "Vi_ygse",
                "Vi_ygse_sig",
                "Vi_zgse",
                "Vi_zgse_sig",
                "Ni",
                "Ni_sig",
                "Tipar",
                "Tipar_sig",
                "Tiperp",
                "Tiperp_sig",
            ]
        else:
            self.out_keys = out_keys
        self.window = hps[0] #Input window length from hyperparameter list
        self.stride = hps[1] #Input stride length from hyperparameter list
        self.fraction = hps[2] #Input maximum tolerable fraction of interpolated data from hyperparameter list
        if model is None:
            # self.model = self.build_model() # Instantiate model architecture with hyperparameters
            resource_path = importlib.resources.path('primesw', 'primesh_v0.1.0.keras')
            with resource_path as model_weights_file:
                self.model = ks.models.load_model(model_weights_file, custom_objects = {'crps_loss' : crps_loss, 'mse_metric' : mse_metric})  # Load the saved model
        else:
            self.model = model
        
    def predict_raw(self, input):
        """
        Wrapper function to predict with a keras model. Differs from `prime.predict_raw` by the inclusion of separate location scaling. Not recommended for direct use, see `prime.predict` instead.
        """
        loc_scaled = self.loc_scaler.transform(input[:, 11:14]) #Get the target position from input, scale separately
        input_scaled = self.in_scaler.transform(input)
        input_scaled[:,11:14] = loc_scaled #Reinsert scaled location
        input_arr = np.zeros((len(input_scaled) - (self.window - 1), self.window, len(self.in_keys)))  # Reshape input data to be 3D
        for i in np.arange(len(input_scaled) - (self.window - 1)):
            input_arr[i, :, :] = input_scaled[i : (i + self.window)]  # Move the 55 unit window through the input data
        output_unscaled = self.model.predict(input_arr)
        output = np.zeros((len(output_unscaled), len(self.out_keys)))  # Stage output data to be 2x target dimensions
        output[:, ::2] = self.tar_scaler.inverse_transform(output_unscaled[:, ::2])  # Mean values
        output[:, 1::2] = np.abs(self.tar_scaler.inverse_transform(output_unscaled[:, ::2] + output_unscaled[:, 1::2]) - self.tar_scaler.inverse_transform(output_unscaled[:, ::2]))  # Standard deviations
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
        loc_mask=False,
        subtract_ecliptic=False,
    ):
        """
        Generate predictions from PRIME on a grid of points in GSE coordinates.

        Parameters:
        -----------
        -    gridsize (float): Spacing of grid points (RE)
        -    x_extent (list): Range of x values to calculate on (GSE RE).
        -    framenum (int): Number of frames to calculate (GSE RE).
        -    bx (float, array-like): IMF Bx value (nT). If array like, must be of length framenum.
        -    by (float, array-like): IMF By value (nT). If array like, must be of length framenum.
        -    bz (float, array-like): IMF Bz value (nT). If array like, must be of length framenum.
        -    vx (float, array-like): Solar wind Vx value (km/s). If array like, must be of length framenum.
        -    vy (float, array-like): Solar wind Vy value (km/s). If array like, must be of length framenum.
        -    vz (float, array-like): Solar wind Vz value (km/s). If array like, must be of length framenum.
        -    ni (float, array-like): Solar wind ion density value (cm^-3). If array like, must be of length framenum.
        -    vt (float, array-like): Solar wind ion thermal speed value (km/s). If array like, must be of length framenum.
        -    rx (float, array-like): Wind spacecraft position x value (GSE RE). If array like, must be of length framenum.
        -    ry (float, array-like): Wind spacecraft position y value (GSE RE). If array like, must be of length framenum.
        -    rz (float, array-like): Wind spacecraft position z value (GSE RE). If array like, must be of length framenum.
        -    y_extent (list): Range of y values to calculate on (GSE RE). If None, z_extent must be specified.
        -    z_extent (list): Range of z values to calculate on (GSE RE). If None, y_extent must be specified.
        -    y (float, array-like): Y position (GSE RE) that is held constant if y_extent is not specified. Default 0.
        -    z (float, array-like): Z position (GSE RE) that is held constant if z_extent is not specified. Default 0.
        -    subtract_ecliptic (bool): Whether or not to subtract the Earth's motion in the ecliptic from Vy. Default False.
        Returns:
        --------
        -    output_grid (ndarray): Array of predicted values on the grid. Shape (framenum, x_extent/gridsize, y_extent/gridsize, 14). Features as in `prime.out_keys`.
        """
        x_arr = np.arange(x_extent[0], x_extent[1], gridsize)  # Create a grid to calculate the magnetosheath conditions on
        y_arr = np.asarray([y]) # This array is overwritten if y_extent is specified
        z_arr = np.asarray([z]) # This array is overwritten if z_extent is specified
        if y_extent is None and z_extent is None:
            raise ValueError("Must specify y_extent or z_extent")
        if y_extent is not None:
            y_arr = np.arange(y_extent[0], y_extent[1], gridsize)  # Y positions to calculate the magnetosheath conditions on
        if z_extent is not None:
            z_arr = np.arange(z_extent[0], z_extent[1], gridsize)  # Z positions to calculate the magnetosheath conditions on
        x_grid, y_grid, z_grid = np.meshgrid(x_arr, y_arr, z_arr)  # Create a grid to calculate the magnetosheath conditions on
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
        if loc_mask:
            r_grid = np.swapaxes(np.sqrt(x_grid**2 + y_grid**2 + z_grid**2), 0, 1) # radial distance to origin at all grid points
            rho_grid = np.swapaxes(np.sqrt(y_grid**2 + z_grid**2), 0, 1) # radial distance to x axis at all grid points
            theta_grid = np.arctan2(rho_grid, np.swapaxes(x_grid, 0, 1)) # Angular distance to x axis for all grid points
            output_mask = np.zeros(output_grid.shape, dtype=bool)  # Initialize array to hold the frame mask
            # Make a mask for all points outside the bow shock or inside the magnetopause
            for i in np.arange(framenum):
                if (framenum == 1):  # If there is only one frame turn pdyn and bz into iterables
                    pdyn_iter = [ni * vx**2 * 1.673e-6]  # Dynamic pressure (nPa)
                    bz_iter = [bz]
                else:
                    pdyn_iter = ni * vx**2 * 1.673e-6
                    bz_iter = bz
                bs_array = jelinek_bs(rho_grid, pdyn_iter[i])
                mp_array = shue_mp(theta_grid, pdyn_iter[i], bz_iter[i])
                for j in np.arange(len(self.out_keys)):
                    output_mask[i, :, : , :, j] = (np.swapaxes(x_grid, 0, 1) > bs_array) | (r_grid < mp_array)
            # Make a masked version of the output grid
            output_grid = np.ma.masked_array(output_grid, mask=output_mask)
        if subtract_ecliptic:  # If subtract_ecliptic is true, subtract the Earth's motion in the ecliptic from Vy
            output_grid[:, :, :, :, 8] -= 29.8
        return output_grid
    
# Analytic surface functions for bow shock and magnetopause
def jelinek_bs(y, pdyn, r0=15.02, l=1.17, e=6.55):
    """
    Bow shock model from Jelinek et al 2012. Assumes GSE Z=0 RE.

    Parameters:
    -----------
    -    y (float): GSE Y coordinate (RE)
    -    pdyn (float): Solar wind dynamic pressure (nPa)
    -    r0 (float): Bow shock average standoff distance tuning parameter (RE)
    -    l (float): Lambda tuning parameter
    -    e (float): Epsilon tuning parameter
    Returns:
    --------
    -    bs_x (float): GSE X position of bow shock (RE)
    """
    bs_x = r0 * (pdyn ** (-1 / e)) - (y**2) * (l**2) / (4 * r0 * (pdyn ** (-1 / e)))
    return bs_x


def shue_mp(theta, pdyn, bz):
    """
    Magnetopause model from Shue et al 1998. Assumes GSE Z=0 RE.

    Parameters:
    -----------
    -    theta (float): Polar angle position of desired MP location (radians)
    -    pdyn (float): Solar wind dynamic pressure (nPa)
    -    bz (float): IMF Bz (nT)
    Returns:
    -    rmp (float): Magnetopause radial distance to Earth (RE)
    """
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * (pdyn ** (-1 / 6.6))
    a1 = (0.58 - 0.007 * bz) * (1 + 0.024 * np.log(pdyn))
    rmp = r0 * (2 / (1 + np.cos(theta))) ** a1
    return rmp