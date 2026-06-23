import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
import torchmetrics
import gc
import importlib.resources
import numpy as np
import pandas as pd
from scipy.special import erf as errorfunc
from scipy.special import erfinv as errorfuncinv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from loguru  import logger
import warnings

from .models import LinearDecoder, RecurrentEncoder, TSPassthroughEncoder
    
class SWRegressor(pl.LightningModule):
    '''
        This class wraps instances of the PRIME architecutre trained on different plasma regions throughout perigeospace.
        It is recommended to instantiate `SWRegressor` objects using `primesw.load()` and specifying the desired model (`PRIME`, `PRIME-SH`, `PRIME-PS`) rather than calling this class directly.
    '''
    def __init__(
            self,
            optimizer = "adam",
            lr = 1e-3,
            lr_scheduler = None,
            patience=3,
            factor=0.5,
            weight_decay = 0,
            total_iters = 40,
            in_dim = 14,
            tar_dim = 1,
            pos_dim = 3,
            in_norm = None,
            tar_norm = None,
            pos_norm = None,
            window = 1,
            stride = 1,
            interp_frac = 1,
            decoder_type = 'linear',
            encoder_type = 'rnn',
            decoder_hidden_layers = [128],
            encoder_hidden_dim = 128,
            encoder_num_layers = 1,
            p_drop = 0.1,
            #Might need a section here to indicate how to handle position
            pos_encoding_size = None,
            loss = 'mae',
            save_debug_ckpt = False,
            *args,
            **kwargs,
    ):
        '''
        :param [str] optimizer: Optimization algorithm used to update model weights. Accepts any Pytorch optimizer alias.
        :param [float] lr: Optimization algorithm learning rate.
        :param [str] lr_scheduler: Optimization algorithm learning rate scheduler. Options are 'cosine', 'cosine_warm', 'plateau', 'linear', or 'const'
        :param [int] patience: Learning rate scheduler patience.
        :param [float] factor: Learning rate scheduler factor.
        :param [float] weight_decay: Optimization algorithm weight decay factor.
        :param [int] total_iters: Total training epochs, used by learning rate scheduler.
        :param [int] in_dim: Input timeseries dimensions (number of features).
        :param [int] tar_dim: Target dimensions (number of features).
        :param [stintr] pos_dim: Number of position dimensions (number of coordinates, generally 3).
        :param [dict] in_norm: Dictionary of input features and their normalization factors.
        :param [dict] tar_norm: Dictionary of target features and their normalization factors.
        :param [dict] pos_norm: Dictionary of position features and their normalization factors.
        :param [int] window: Timeseries window size.
        :param [int] stride: Prediction lead time.
        :param [float] interp_frac: Percent permissible interpolated data in each input timeseries.
        :param [str] decoder_type: Decoder architecture. Options are 'linear' or 'prob_linear'.
        :param [str] encoder_type: Encoder architecture. Options are 'linear' or 'rnn'.
        :param [list of int] decoder_hidden_layers: Size of each of the hidden layers in decoder.
        :param [int] encoder_hidden_dim: Dimension of hidden layers in encoder.
        :param [int] encoder_num_layers: Number of layers in the encoder.
        :param [float] p_drop: Dropout rate for model during training.
        :param [int] pos_encoding_size: Order of random Fourier features applied to position data.
        :param [str] loss: Loss function for model, options are 'mae' or 'crps'. 'crps' only usable for 'prob_linear' decoders.
        :param [bool] save_debug_ckpt: Whether to dump a debug packet on each validation epoch end.
        '''
        super().__init__(*args, **kwargs) # Pass bonus arguments to the LightningModule
        self.save_hyperparameters() #inherited method from LightningModule
        self.save_debug_ckpt = save_debug_ckpt # Controls whether validation set/predictions and model is saved in on_validation_epoch_end()

        # Optimiser Parameters
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_iters = total_iters # Used for certain LR schedulers
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.factor = factor

        # Model Parameters
        self.in_dim = in_dim
        self.tar_dim = tar_dim
        self.pos_dim = pos_dim
        self.in_norm = in_norm
        self.tar_norm = tar_norm
        self.pos_norm = pos_norm
        self.window = window
        self.stride = stride # Only included so that it's saved as a hyperparameter
        self.interp_frac = interp_frac # Same as above
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.decoder_hidden_layers = decoder_hidden_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_num_layers = encoder_num_layers
        self.p_drop = p_drop
        self.pos_encoding_size = pos_encoding_size

        # Loss parameters
        self.loss = loss
        # if self.loss == 'mae': #NOTE: I don't think we need any additional scalars logged for MAE-trained models? Besides the loss, which is handled later.
            # self.trn_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
            # self.val_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
            # self.tst_mae = torchmetrics.MeanAbsoluteError(num_outputs = self.tar_dim)
        if self.loss == 'crps':
            self.trn_mae = ProbabilisticMeanAbsoluteError()
            self.val_mae = ProbabilisticMeanAbsoluteError()
            self.tst_mae = ProbabilisticMeanAbsoluteError()

        # Initialize the encoder
        match self.encoder_type:
            case "rnn":
                self.encoder = RecurrentEncoder(
                    in_dim = self.in_dim,
                    encoding_size = self.encoder_hidden_dim,
                    num_layers = self.encoder_num_layers,
                    p_drop = self.p_drop,
                )
                decoder_in_dim = self.encoder_hidden_dim
            case "linear":
                self.encoder = TSPassthroughEncoder(
                    in_dim = self.in_dim * self.window,
                )
                decoder_in_dim = self.in_dim * self.window
            case _:
                raise ValueError(f"Invalid encoder type {self.encoder_type}")
        
        # Initialize the decoder
        match self.decoder_type:
            case "linear":
                self.decoder = LinearDecoder(
                    in_dim = decoder_in_dim,
                    tar_dim = self.tar_dim,
                    pos_dim = self.pos_dim,
                    pos_encoding_size = self.pos_encoding_size,
                    hidden_layers = self.decoder_hidden_layers,
                    p_drop = self.p_drop,
                )
            case "prob_linear": 
                # This is a special case of linear that outputs two values for each target feature.
                # NOTE: Compatible with loss = 'crps' ONLY!
                self.decoder = LinearDecoder(
                    in_dim = decoder_in_dim,
                    tar_dim = self.tar_dim * 2,
                    pos_dim = self.pos_dim,
                    pos_encoding_size = self.pos_encoding_size,
                    hidden_layers = self.decoder_hidden_layers,
                    p_drop = self.p_drop,
                )
            case _:
                raise ValueError(f"Invalid decoder type {self.decoder_type}")
        
        # Handle the loss type
        match self.loss:
            case "mae":
                self.loss_fn = torch.nn.L1Loss()
            case "crps":
                self.loss_fn = lambda outputs, targets: crps(
                    outputs,
                    targets,
                ).mean()
            case _:
                raise ValueError(f"Invalid loss type {self.loss}")
            
        # Define things we keep around for validation purposes, not passed to the model
        self.val_predictions = []
        self.val_targets = []
        self.val_times = []

        # Define an example input pair for generating the model graph
        self.example_input_array = (torch.rand(50, self.window, self.in_dim, device = self.device), torch.rand(50, self.pos_dim, device = self.device)) # (x, position)
    
    def forward(self, x, position):
        '''
        Model forward pass.

        :meta private:
        '''
        out, h = self.encoder.forward(x)
        y_hat = self.decoder.forward(out, position)
        return y_hat

    #########################
    # User-facing functions #
    #########################

    def predict_ts(self, start = None, stop = None, in_data = None, pos = np.array([13.25, 0, 0])):
        '''
        Generate predictions from the model. Specify either a `start` and `stop` time or supply input data (`in_data`) from an L1 monitor.
        Predictions are made at a static position (13.25 RE upstream on GSE X axis), but a different or moving position can be specified with `pos`.
        
        :param [str] start: Start time of desired prediction in format `'YYYY-MM-DD HH:MM:SS'`.
        :param [str] stop: Stop time of desired prediction in format `'YYYY-MM-DD HH:MM:SS'`.
        :param [DataFrame or ndarray] in_data: Input data from L1 monitor. If DataFrame, must contain the keys the model expects (check `model.in_norm.keys()`). If Numpy array, must be in the order expected by the model. Overrides `start` and `stop` if specified.
        :param [DataFrame or ndarray] pos: Position(s) of desired prediction. For a static position, specify a 1D vector with three inputs corresponding to GSE X/Y/Z position. For a moving target, specify a 2D array or DataFrame with one entry per timestep in the timeseries. If DataFrame, must contain the keys the model expects (check `model.pos_norm.keys()`). 
        :returns: [DataFrame] output: Model output for the given timerange or input data. Includes means and standard deviations at each timestep.
        '''
        pos_keys = list(self.pos_norm.keys())
        if in_data is None:
            if (start is not None)&(stop is not None):
                new_start = (pd.to_datetime(start, utc = True) - pd.Timedelta(seconds = int(100 * (self.window + self.stride + 1)))).strftime('%Y-%m-%d %H:%M:%S')
                new_stop = (pd.to_datetime(stop, utc = True) - pd.Timedelta(seconds = int(100 * (self.stride)))).strftime('%Y-%m-%d %H:%M:%S')
                timeseries = self.build_real_input(start = new_start, stop = new_stop)
            else:
                raise RuntimeWarning('Must specify either input or (start and stop).')
                return None
        elif isinstance(in_data, np.ndarray): # If input is an array
            timeseries = pd.DataFrame(in_data, columns = self.in_norm.keys()) # Make it into a dataframe
            if (start is not None)&(stop is not None): # Start and stop time can be specified to make the output have times associate with it
                timeseries['Epoch'] = pd.date_range(pd.to_datetime(start), pd.to_datetime(stop), periods = len(timeseries))
        else: # Generally this happens if input is a dataframe
            timeseries = in_data.copy()
        # Do position related biz
        #First, array-ify the position
        if isinstance(pos, list):
            positions = np.asarray(pos)
        else:
            positions = pos.copy()
        # Now, check if the position is static
        if (positions.ndim == 1): #Were we passed a 1D array corresponding to a static position?
            positions = pd.DataFrame(np.asarray([positions[0] * np.ones(len(timeseries)), positions[1] * np.ones(len(timeseries)), positions[2] * np.ones(len(timeseries))]).T, columns = [pos_keys[0], pos_keys[1], pos_keys[2]])
        elif len(pos)!=len(timeseries): # Were we passed a 2D array of the wrong shape?
            raise RuntimeWarning('Positions array must match length of input timeseries (check with the build_real_input() method).')
        elif isinstance(positions, np.ndarray): # Were we passed an array we need to turn into a dataframe?
            positions = pd.DataFrame(positions, columns = self.pos_norm.keys())
        elif ~isinstance(positions, pd.DataFrame): #Okay fr you better have passed a DataFrame, anything else would be foolish
            raise RuntimeWarning('Positions must be supplied as list, array, or DataFrame.')
        output = self.predict_df(timeseries, positions)
        return output

    def predict_df(self, timeseries, position):
        '''
        Generate predictions from the model using DataFrames. In general it is recommended to use `predict_ts()` which is a more flexible wrapper of this method.
        
        :param [DataFrame] timeseries: Input data from L1 monitor. Must contain the keys the model expects (check `model.in_norm.keys()`).
        :param [DataFrame] position: Position(s) of desired prediction. Must contain the keys the model expects (check `model.pos_norm.keys()`). 
        :returns [DataFrame] tar_scaled: Model output for the given timerange or input data. Includes means and standard deviations at each timestep.
        '''
        in_scaled = timeseries.loc[:, self.in_norm.keys()].copy() # Get just the keys used for prediction
        for feature in self.in_norm.keys(): # Scale each input feature DOWN
            in_scaled[feature] = (in_scaled[feature] - self.in_norm[feature][0])/self.in_norm[feature][1]

        # Turn in_scaled into a numpy array of the correct shape
        in_arr = np.zeros((len(position) - self.window, self.window, len(self.in_norm.keys())))
        for i, idx in enumerate(in_scaled.index):
            if i < self.window:
                continue
            in_arr[i - self.window, :, :] = in_scaled.loc[(idx - self.window):(idx - 1), :]

        pos_scaled = position.iloc[self.window:].loc[:, self.pos_norm.keys()].copy() # Get just the position elements
        for feature in self.pos_norm.keys(): # Scale each position DOWN
            pos_scaled[feature] = (pos_scaled[feature] - self.pos_norm[feature][0])/self.pos_norm[feature][1]
        
        # Tensor-ify the inputs from pandas dataframes
        in_tensor = torch.from_numpy(in_arr.astype(np.float32)).to(self.device)
        pos_tensor = torch.from_numpy(pos_scaled.to_numpy().astype(np.float32)).to(self.device)

        y_hat = self(in_tensor, pos_tensor) # Run an actual forward pass
        y_hat = y_hat.detach().cpu().numpy()

        # Try to initialize the return dataframe
        try:
            tar_scaled = pd.DataFrame(timeseries['Epoch'].iloc[self.window:] + pd.Timedelta(seconds = self.stride * 100), columns = ['Epoch'])
        except KeyError: # If there is no 'Epoch' in the supplied dataframe
            warnings.warn('timeseries DataFrame does not have Epoch key, no time data will be returned')
            tar_scaled = pd.DataFrame([], index = timeseries.index[self.window:])
        
        if y_hat.shape[1] == 2*len(self.tar_norm.keys()): # If the output is means + stdevs
            for i, feature in enumerate(self.tar_norm.keys()):
                tar_scaled[feature] = (y_hat[:, i*2] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]
                tar_scaled[feature + '_std'] = ((y_hat[:, i*2] + y_hat[:, i*2 + 1]) * self.tar_norm[feature][1]) + self.tar_norm[feature][0] - tar_scaled[feature]
        else:
            for i, feature in enumerate(self.tar_norm.keys()):
                tar_scaled[feature] = (y_hat[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]
        
        return tar_scaled
    
    def predict_grid(
        self,
        ts,
        gridsize,
        x_extent,
        y_extent=None,
        z_extent=None,
        y = 0,
        z = 0,
        loc_mask=None,
        subtract_ecliptic=False,
    ):
        """
        Generate predictions efficiently on a grid of points. Timeseries data can be generated with methods `build_synth_input()` or `build_real_input()`, depending on desired data source.

        :param [DataFrame] ts: Timeseries of data at L1 (Can be synthetic)
        :param [float] gridsize: Spacing of grid points
        :param [list] x_extent: Range of x values to calculate on
        :param [list] y_extent: Range of y values to calculate on. If None, z_extent must be specified.
        :param [list] z_extent: Range of z values to calculate on. If None, y_extent must be specified.
        :param [float, array-like] y: Y position that is held constant if y_extent is not specified. Default 0.
        :param [float, array-like] z: Z position that is held constant if z_extent is not specified. Default 0.
        :param [float, optional] loc_mask: RE from Earth to occlude (masking the magnetopause/magnetosheath)
        :returns [ndarray] output_grid: Array of predicted values on the grid. Shape (timestamps, x_extent/gridsize, y_extent/gridsize, z_extent/gridsize, features * 2)
        :returns [Series] timestamps: Series of datetimes corresponding to each grid's time
        """
        # Generate the embeddings from the timeseries
        in_scaled = ts.loc[:, self.in_norm.keys()].copy() # Get just the keys used for prediction
        for feature in self.in_norm.keys(): # Scale each input feature DOWN
            in_scaled[feature] = (in_scaled[feature] - self.in_norm[feature][0])/self.in_norm[feature][1]
        in_arr = np.zeros((len(in_scaled) - self.window, self.window, len(self.in_norm.keys()))) # Prepare the input array in the shape the encoder expects
        for i, idx in enumerate(in_scaled.index): # Fill each segment with input data
            if i < self.window:
                continue
            in_arr[i - self.window, :, :] = in_scaled.loc[(idx - self.window - self.stride):(idx - self.stride - 1), :]
        in_tensor = torch.from_numpy(in_arr.astype(np.float32)).to(self.device) # Make sure the inputs are on the same device as the model
        embeddings, h = self.encoder.forward(in_tensor) # Generate the embeddings (1 forward pass)

        # Create the grid based on the extent and gridsize specs
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
        
        # Make the ultralong position tensor
        steps = len(embeddings)
        pos_arr = np.zeros((len(x_grid.flatten()) * steps, 3))  # Initialize array to hold the position data
        pos_arr[:, 0] = np.tile(x_grid.flatten(), steps)
        pos_arr[:, 1] = np.tile(y_grid.flatten(), steps)
        pos_arr[:, 2] = np.tile(z_grid.flatten(), steps)
        for i, feature in enumerate(self.pos_norm.keys()): # Scale each position feature DOWN
            pos_arr[:, i] = (pos_arr[:, i] - self.pos_norm[feature][0])/self.pos_norm[feature][1]
        pos_tensor = torch.from_numpy(pos_arr.astype(np.float32)).to(self.device) # Make sure the positions are on the same device as the model

        # Extend embeddings to match the size of the position tensor
        embeddings = embeddings.repeat_interleave(len(x_grid.flatten()), dim = 0) #NOTE: Torch repeat_interleave() works like numpy repeat()
        
        # Run a forward pass on the ultralong tensors and rescale the outputs to human units
        y_hat = self.decoder(embeddings, pos_tensor) # Run an actual forward pass
        y_hat = y_hat.detach().cpu().numpy()
        output_raveled = np.empty(y_hat.shape)
        if y_hat.shape[1] == 2*len(self.tar_norm.keys()): # If the output is means + stdevs
            for i, feature in enumerate(self.tar_norm.keys()):
                output_raveled[:, i*2] = (y_hat[:, i*2] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]
                output_raveled[:, i*2 +1 ] = ((y_hat[:, i*2] + y_hat[:, i*2 + 1]) * self.tar_norm[feature][1]) + self.tar_norm[feature][0] - output_raveled[:, i*2]
        else:
            for i, feature in enumerate(self.tar_norm.keys()):
                output_raveled[:, i]  = (y_hat[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]

        # Reshape the output data to transform it back to the grid
        output_grid = output_raveled.reshape(steps, len(y_arr), len(x_arr), len(z_arr), y_hat.shape[1])  # Reshape the output data into the correct shape
        output_grid = np.swapaxes(output_grid, 1, 2)  # Move the y axis to the second axis (new order is frame, x, y, z, param)
        if loc_mask is not None:
            r_grid = np.swapaxes(np.sqrt(x_grid**2 + y_grid**2 + z_grid**2), 0, 1) # radial distance to origin at all grid points
            output_mask = np.zeros(output_grid.shape, dtype=bool)  # Initialize array to hold the frame mask
            # # Make a mask for all points outside the bow shock or inside the magnetopause
            for i in np.arange(steps):
                for j in np.arange(output_grid.shape[-1]):
                    output_mask[i, :, : , :, j] = (r_grid < loc_mask)
            # Make a masked version of the output grid
            output_grid = np.ma.masked_array(output_grid, mask=output_mask)
        return output_grid

    def build_synth_input(
            self,
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
            sme = None,
            smr = None,
            tilt = None,
            ):
        '''
        Builds a synthetic input array from user-specified quantities at L1.
        For input arrays made from measured data at L1, see `SWRegressor.build_real_input`.
        
        :param [float, array-like] x: IMF Bx value (nT).
        :param [float, array-like] by: IMF By value (nT).
        :param [float, array-like] bz: IMF Bz value (nT).
        :param [float, array-like] vx: Solar wind Vx value ().
        :param [float, array-like] vy: Solar wind Vy value.
        :param [float, array-like] vz: Solar wind Vz value.
        :param [float, array-like] ni: Solar wind ion density value.
        :param [float, array-like] vt: Solar wind ion thermal speed value.
        :param [float, array-like] rx: Wind spacecraft position x value.
        :param [float, array-like] ry: Wind spacecraft position y value.
        :param [float, array-like] rz: Wind spacecraft position z value.
        :param [float, array-like] sme: SuperMAG SME index (nT). Only used for plasmasheet model.
        :param [float, array-like] smr: SuperMAG SMR index (nT). Only used for plasmasheet model.
        :param [float, array-like] tilt: Earth dipole tilt angle (degrees). Only used for plasmasheet model.
        :returns [Dataframe] in_df: Input dataframe suitable to predict from with self.predict_ts(). 
        '''
        in_df = pd.DataFrame(columns = self.in_norm.keys()) #Initialize single-point input dataframe
        # NOTE: The following is designed to only work with keys from the Wind key parameters datasets
        for key in self.in_norm.keys():
            if (key == 'BGSE_0')|(key == 'B_xgsm'):
                in_df[key] = bx*np.ones(self.window+1) #SW BX in nT (GSM coordinates)
            if (key == 'BGSE_1')|(key == 'B_ygsm'):
                in_df[key] = by*np.ones(self.window+1) #SW BY in nT (GSM coordinates)
            if (key == 'BGSE_2')|(key == 'B_zgsm'):
                in_df[key] = bz*np.ones(self.window+1) #SW BZ in nT (GSM coordinates)
            if (key == 'V_GSE_0')|(key == 'Vi_xgse'):
                in_df[key] = vx*np.ones(self.window+1) #SW X velocity in km/s (GSE coordinates)
            if (key == 'V_GSE_1')|(key == 'Vi_ygse'):
                in_df[key] = vy*np.ones(self.window+1) #SW Y velocity in km/s (GSE coordinates)
            if (key == 'V_GSE_2')|(key == 'Vi_zgse'):
                in_df[key] = vz*np.ones(self.window+1) #SW Z velocity in km/s (GSE coordinates)
            if (key == 'Np')|(key == 'Ni'):
                in_df[key] = ni*np.ones(self.window+1) #SW density in cm^-3
            if (key == 'THERMAL_SPD')|(key == 'Vth'):
                in_df[key] = vt*np.ones(self.window+1) #SW thermal velocity in km/s
            if (key == 'PGSE_0')|(key == 'R_xgse'):
                in_df[key] = rx*np.ones(self.window+1) #Wind position X GSE (RE)
            if (key == 'PGSE_1')|(key == 'R_ygse'):
                in_df[key] = ry*np.ones(self.window+1) #Wind position Y GSE (RE)
            if (key == 'PGSE_2')|(key == 'R_zgse'):
                in_df[key] = rz*np.ones(self.window+1) #Wind position Z GSE (RE)
            if (key == 'SME'):
                in_df[key] = sme*np.ones(self.window+1) #SuperMAG SMR index (nT)
            if (key == 'SMR'):
                in_df[key] = smr*np.ones(self.window+1) #SuperMAG SME index (nT)
            if (key == 'tilt'):
                in_df[key] = tilt*np.ones(self.window+1) #Earth dipole tilt angle (degrees)
        return in_df
    
    def build_real_input(self, start, stop):
        '''
        Load Wind spacecraft input data in between specified date strings.

        :param [str] start: The start date of the data to load ('YYYY-MM-DD')
        :param [str] stop: The end date of the data to load ('YYYY-MM-DD')
        :returns [DataFrame] in_df: Input dataframe suitable to predict from with self.predict(). 
        '''
        try:
            from cdasws import CdasWs
        except:
            raise RuntimeError('Unable to import CdasWs. Predicting using non-synthetic input requires CdasWs package (see https://cdaweb.gsfc.nasa.gov/WebServices/REST/py/cdasws/).')
        cdas = CdasWs() #Initialize CDAS WS Session
        mfi_df = pd.DataFrame([]) #Staging dataframe for Wind spacecraft Magnetic Field Investigation data
        try:
            data = cdas.get_data('WI_K0_MFI', ['BGSMc', 'PGSE'], start, stop) #Load GSM B field and GSE SC position
            mfi_df['Epoch'] = data[1]['Epoch'] #MFI timestamps
            mfi_df['R_xgse'] = data[1]['PGSE'][:, 0] #Wind SC position
            mfi_df['R_ygse'] = data[1]['PGSE'][:, 1]
            mfi_df['R_zgse'] = data[1]['PGSE'][:, 2]
            mfi_df['B_xgsm'] = data[1]['BGSMc'][:, 0] #GSM B field
            mfi_df['B_ygsm'] = data[1]['BGSMc'][:, 1]
            mfi_df['B_zgsm'] = data[1]['BGSMc'][:, 2]
        except TypeError: #Throws when date range is empty OR too big
            raise RuntimeError('CDASWS failed to load MFI data. Date range ('+start+' to '+stop+') may be too large or data may be missing.')
        mfi_df['Epoch'] = pd.to_datetime(mfi_df['Epoch'], utc=True) #Convert to UTC aware datetime
        #Set B values to nan if they are equal to the fill value of -1e31
        mfi_df['B_xgsm'] = mfi_df['B_xgsm'].where(mfi_df['B_xgsm'] > -1e30, np.nan)
        mfi_df['B_ygsm'] = mfi_df['B_ygsm'].where(mfi_df['B_ygsm'] > -1e30, np.nan)
        mfi_df['B_zgsm'] = mfi_df['B_zgsm'].where(mfi_df['B_zgsm'] > -1e30, np.nan)
        #Set R values to nan if they are equal to the fill value of -1e31
        mfi_df['R_xgse'] = mfi_df['R_xgse'].where(mfi_df['R_xgse'] > -1e30, np.nan)
        mfi_df['R_ygse'] = mfi_df['R_ygse'].where(mfi_df['R_ygse'] > -1e30, np.nan)
        mfi_df['R_zgse'] = mfi_df['R_zgse'].where(mfi_df['R_zgse'] > -1e30, np.nan)
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
        swe_df['Epoch'] = swe_df['Epoch'].where(swe_df['Epoch'] >= pd.to_datetime(start, utc=True), np.nan)
        swe_df['Epoch'] = swe_df['Epoch'].where(swe_df['Epoch'] <= pd.to_datetime(stop, utc=True), np.nan)
        #Remove rows with nan Epochs and reset the index
        swe_df = swe_df.dropna(subset=['Epoch'])
        swe_df = swe_df.reset_index(drop=True)
        #Set Ni values to nan if they are equal to the fill value of -1e31
        swe_df['Ni'] = swe_df['Ni'].where(swe_df['Ni'] > -1e30, np.nan)
        #Set Vi values to nan if they are equal to the fill value of -1e31
        swe_df['Vi_xgse'] = swe_df['Vi_xgse'].where(swe_df['Vi_xgse'] > -1e30, np.nan)
        swe_df['Vi_ygse'] = swe_df['Vi_ygse'].where(swe_df['Vi_ygse'] > -1e30, np.nan)
        swe_df['Vi_zgse'] = swe_df['Vi_zgse'].where(swe_df['Vi_zgse'] > -1e30, np.nan)
        #Set Vth values to nan if they are equal to the fill value of -1e31
        swe_df['Vth'] = swe_df['Vth'].where(swe_df['Vth'] > -1e30, np.nan)
        #Set vflag values to nan if they are equal to the fill value of -2147483648
        swe_df['vflag'] = swe_df['vflag'].where(swe_df['vflag'] > -2147483648, np.nan)
        #Set niflag values to nan if they are equal to the fill value of -2147483648
        swe_df['niflag'] = swe_df['niflag'].where(swe_df['niflag'] > -2147483648, np.nan)
        include_supermag = False # Flag for later merges needing to include supemag
        if ('SME' in self.in_norm.keys())|('SME' in self.in_norm.keys())|('tilt' in self.in_norm.keys()): # Load the supermag data here
            include_supermag = True # Include supermag in later merges
            # TODO: Find out if the supermag_api license permits insertion into this package
            import warnings
            warnings.warn("SuperMAG API not integrated yet. Please manually add SME/SMR to returned dataframe.")
            supermag_df = pd.DataFrame(np.zeros((len(swe_df), 4)), columns = ['Epoch', 'SME', 'SMR', 'tilt'])
            supermag_df['Epoch'] = swe_df['Epoch']
        #Bin the data to 100s bins (default PRIME input cadence)
        bins = pd.date_range(pd.to_datetime(start, utc = True), pd.to_datetime(stop, utc = True), freq='100s')
        bins_index = pd.IntervalIndex.from_arrays(bins[:-1], bins[1:], closed='left') #Make interval index for binning
        swe_group = swe_df.groupby(pd.cut(swe_df['Epoch'], bins_index), observed = False) #Group the SWE and MFI data
        mfi_group = mfi_df.groupby(pd.cut(mfi_df['Epoch'], bins_index), observed = False)
        swe_binned = swe_group.mean() #Take the mean in each group (bin)
        mfi_binned = mfi_group.mean()
        swe_binned = swe_binned.reset_index(drop=True) #Reset index to integers instead of group labels
        mfi_binned = mfi_binned.reset_index(drop=True)
        in_df = pd.merge(swe_binned, mfi_binned, left_index = True, right_index = True) #Combine the SWE and MFI dataframes
        if include_supermag:
            supermag_group = supermag_df.groupby(pd.cut(supermag_df['Epoch'], bins_index), observed = False) #Group the SuperMAG data
            supermag_binned = supermag_group.mean()
            supermag_binned = supermag_binned.reset_index(drop=True)
            in_df = pd.merge(in_df, supermag_binned, left_index = True, right_index = True) #Add in the supermag dataframe
        in_df['Epoch'] = bins[:-1] #The last bin gets dropped.
        in_df['flag'] = in_df.isna().any(axis=1) #Get the rows with NaNs and flag them as interpolated
        rename_dict = {} # Make a dictionary that renames the data columns to the correct keys expected by the model
        for key in self.in_norm.keys():
            if (key == 'BGSE_0')|(key == 'B_xgsm'):
                rename_dict['B_xgsm'] = key #SW BX in nT (GSM coordinates)
            if (key == 'BGSE_1')|(key == 'B_ygsm'):
                rename_dict['B_ygsm'] = key #SW BY in nT (GSM coordinates)
            if (key == 'BGSE_2')|(key == 'B_zgsm'):
                rename_dict['B_zgsm'] = key #SW BZ in nT (GSM coordinates)
            if (key == 'V_GSE_0')|(key == 'Vi_xgse'):
                rename_dict['Vi_xgse'] = key #SW X velocity in km/s (GSE coordinates)
            if (key == 'V_GSE_1')|(key == 'Vi_ygse'):
                rename_dict['Vi_ygse'] = key #SW Y velocity in km/s (GSE coordinates)
            if (key == 'V_GSE_2')|(key == 'Vi_zgse'):
                rename_dict['Vi_zgse'] = key #SW Z velocity in km/s (GSE coordinates)
            if (key == 'Np')|(key == 'Ni'):
                rename_dict['Ni'] = key #SW density in cm^-3
            if (key == 'THERMAL_SPD')|(key == 'Vth'):
                rename_dict['Vth'] = key #SW thermal velocity in km/s
            if (key == 'PGSE_0')|(key == 'R_xgse'):
                rename_dict['R_xgse'] = key #Wind position X GSE (RE)
            if (key == 'PGSE_1')|(key == 'R_ygse'):
                rename_dict['R_ygse'] = key #Wind position Y GSE (RE)
            if (key == 'PGSE_2')|(key == 'R_zgse'):
                rename_dict['R_zgse'] = key #Wind position Z GSE (RE)
            if (key == 'SME'):
                rename_dict['SME'] = key #SuperMAG SMR index (nT)
            if (key == 'SMR'):
                rename_dict['SMR'] = key #SuperMAG SME index (nT)
            if (key == 'tilt'):
                rename_dict['tilt'] = key #Earth dipole tilt angle (degrees)
        in_df = in_df.rename(columns = rename_dict) # Rename according to the above
        for key in self.in_norm.keys(): # Interpolate over nans
            in_df[key] = in_df[key].interpolate(method='linear', axis=0)
        in_df = in_df.dropna() #If a nan snuck in get it outta here! For real, scram!
        in_df = in_df.reset_index(drop=True) #Reset the index
        return in_df
    
    ###################################################################
    # Non-user-facing functions overriding pytorch lightning defaults #
    ###################################################################
    
    def predict_step(self, batch, batch_idx):
        '''
        One prediction step (no gradient updates).

        :meta private:
        '''
        timeseries, position, target, times = batch
        with torch.no_grad():
            y_hat = self(timeseries, position)
            h = self.encoder.forward(timeseries)
        return {
            'inputs': timeseries,
            'positions': position,
            'encodings': h,
            'predictions': y_hat,
            'targets': target,
            'timestamps': times,
        }

    def training_step(self, batch, batch_idx):
        '''
        One training step. Updates train-side metrics.

        :meta private:
        '''
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.trn_mae.update(y_hat, target)

        self.log(
            'Loss/train',
            loss.mean(),
            on_step=True,     # Log every step
            on_epoch=True,    # Log at end of epoch
            prog_bar=True,    # Show in progress bar
            logger=True,
            sync_dist=True
        )
        # Log current learning rate from optimizer
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('Opt/lr', lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        One validation step. Updates validation-side metrics.

        :meta private:
        '''
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        val_loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.val_mae.update(y_hat, target)
        self.log('Loss/val', val_loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Store the batches so we can make a 2D joint distribution at epoch end
        self.val_predictions.append(y_hat.cpu())
        self.val_targets.append(target.cpu())
        self.val_times.append(times)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        '''
        One test step. Updates test-side metrics.

        :meta private:
        '''
        timeseries, position, target, times = batch
        y_hat = self(timeseries, position)
        # Calculate loss
        test_loss = self.loss_fn(y_hat, target)

        # Update the metrics
        if self.loss == 'crps':
            self.tst_mae.update(y_hat, target)
        self.log('Loss/test', test_loss.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return {
            "predictions": y_hat,
            "targets": target,
            "test_loss": test_loss,
            "timestamps": times,
        }
    
    def on_validation_epoch_end(self):
        '''
        Compute and log all accumulated metrics

        :meta private:
        '''
        if self.loss == 'crps':
            self.log('MAE/val', self.val_mae.compute().mean(), on_epoch = True, prog_bar = True, logger = True, sync_dist = True)
            # Clear all the metrics
            self.val_mae.reset()

        val_preds = torch.cat(self.val_predictions, dim = 0).numpy()
        targets = torch.cat(self.val_targets, dim = 0).numpy()
        if val_preds.shape[-1] == (self.tar_dim * 2): # Are we using one that outputs a mean and a standard deviation?
            predictions = val_preds[:, ::2]
            # logger.info(f"Plotting JD of probabilistic predictions of size {predictions.shape}")
        else:
            predictions = val_preds
            # logger.info(f"Plotting JD of deterministic predictions of size {predictions.shape}")
        fig, ax = plt.subplots(nrows = 1, ncols = self.tar_dim, figsize = (6 * self.tar_dim, 6))
        nbins = 50
        if self.tar_dim == 1: # In the case of a single target parameter, the Axes object will not be subscriptable
            ax = [ax] # Increase the dimensions of ax so that the indexing below still works
        for i, feature in enumerate(self.tar_norm.keys()):
            im = ax[i].hexbin(
                (targets[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0],
                (predictions[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0],
                gridsize = nbins,
                norm = LogNorm(1e0, 1e3),
                cmap = 'inferno', # TODO: make a fun new colormap
            )
            # ax[i].set_aspect("equal")
            ax[i].set_xlabel(f"Target {i}")
            ax[i].set_ylabel(f"Predcted {i}")
            ax[i].set_title(f"{feature}")
            lims = [
                np.min([ax[i].get_xlim(), ax[i].get_ylim()]),  # min of both axes
                np.max([ax[i].get_xlim(), ax[i].get_ylim()]),  # max of both axes
            ]
            ax[i].plot(
                lims,
                lims,
                color="k",
                linestyle = "--",
            )
            ax[i].set_aspect('equal')
            ax[i].set_xlim(lims)
            ax[i].set_ylim(lims)
        self.logger.experiment.add_figure(f"JD/val_epoch{self.current_epoch}", fig)
        fig.clear()

        if self.loss == 'crps': # CRPS implies we're training a probabilistic model, plot a reliability diagram
            fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (7, 9))
            phi = np.linspace(0,1,1000) #Observed probabily axis
            cumulative_dist = np.zeros((len(phi), len(self.tar_norm.keys()))) #Cumulative distribution for each parameter
            for i, feature in enumerate(self.tar_norm.keys()):
                standard_err = (
                    (((val_preds[:, i*2] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]) - ((targets[:, i] * self.tar_norm[feature][1]) + self.tar_norm[feature][0])) /
                    (np.sqrt(2) * ((val_preds[:, i*2] * self.tar_norm[feature][1]) + self.tar_norm[feature][0]))
                )
                for j in range(len(val_preds)):
                    cumulative_dist[:,i] += (1/len(standard_err)) * np.heaviside(phi - 0.5*(errorfunc(standard_err[j])+1) , 1) #Calculate the cumulative distribution for each parameter
            
            for i, feature in enumerate(self.tar_norm.keys()):
                ax[0].plot(phi, cumulative_dist[:,i], label = feature)
                ax[1].plot(phi, phi - cumulative_dist[:,i], label = feature)
            #Place legend to the right middle of the figure
            ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
            ax[0].plot(phi, phi, linestyle = '--', color = 'k')
            ax[0].set_ylabel('Observed Frequency')
            ax[0].set_xlim(0,1)
            ax[0].set_ylim(0,1)
            ax[1].plot(phi, np.zeros(len(phi)), linestyle = '--', color = 'k')
            # ax[1].set_ylim(-0.15,0.15)
            ax[1].set_xlabel('Predicted Frequency')
            ax[1].set_ylabel('Under/Over-\nEstimation')
            ax[1].set_aspect('equal')
            # plt.subplots_adjust(hspace = -0.20)
            self.logger.experiment.add_figure(f"RD/val_epoch{self.current_epoch}", fig)
            fig.clear()

        if self.save_debug_ckpt:
            import pickle # For saving the normalizations
            np.save('/glade/u/home/cobrien/data/prime/debug_packet/predictions.npy', predictions)
            np.save('/glade/u/home/cobrien/data/prime/debug_packet/targets.npy', targets)
            with open('/glade/u/home/cobrien/data/prime/debug_packet/val_times.pkl', 'wb') as f:
                pickle.dump(self.val_times, f)
            with open('/glade/u/home/cobrien/data/prime/debug_packet/in_norm.pkl', 'wb') as f:
                pickle.dump(self.in_norm, f)
            with open('/glade/u/home/cobrien/data/prime/debug_packet/tar_norm.pkl', 'wb') as f:
                pickle.dump(self.tar_norm, f)
            with open('/glade/u/home/cobrien/data/prime/debug_packet/pos_norm.pkl', 'wb') as f:
                pickle.dump(self.pos_norm, f)

        self.val_predictions.clear()
        self.val_targets.clear()
        self.val_times.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self):
        '''
        :meta private:
        '''
        gc.collect()
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        '''
        :meta private:
        '''
        if self.loss == 'crps':
            self.log('MAE/test', self.tst_mae.compute(), on_epoch=True, logger=True, sync_dist=True)

    def on_before_optimizer_step(self, optimizer):
        '''
        Computes the 2-norm for each layer. If using mixed precision, the gradients are already unscaled here.

        :meta private:
        '''
        norms = pl.utilities.grad_norm(self.encoder, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        '''
        :meta private:
        '''
        match (self.optimizer):
            case "adam":
                optimizer = torch.optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "sgd":
                optimizer = torch.optim.SGD(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case "adamw":
                optimizer = torch.optim.AdamW(
                    self.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
            case _:
                raise NameError(f"Unknown optimizer {optimizer}")
        # Select LR scheduler
        scheduler_config = None
        match self.lr_scheduler:
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.trainer.max_epochs,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "cosine_warm":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=self.total_iters,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=self.factor, patience=self.patience,
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'monitor': 'Loss/val',  # Add this required parameter!
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "linear":
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1, end_factor=self.factor, total_iters=self.total_iters
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case "const":
                scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=self.factor, total_iters=self.total_iters
                )
                scheduler_config = {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            case _:
                raise ValueError(f"Unsupported scheduler: {self.lr_scheduler}")

        # Return config based on whether a scheduler is used
        if scheduler_config is not None:
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler_config
            }
        else:
            return optimizer

def crps(outputs, targets):
    '''
        This function uses the 1st, 3rd, 5th... neurons in the last layer as the means of the output 
        Gaussian and the 2nd, 4th, 6th... neurons as the variance of the output Gaussians for each
        target parameter. See http://www.dl.begellhouse.com/journals/52034eb04b657aea,3ec0b84376cff3d2,1801e97431c5911b.html
        section 2 (equations 2 and 3) for more info. 
    '''
    if ((outputs.size(-1)%2)!=0):
        raise ValueError(f"CRPS loss function requires even number of outputs from model.")
    if outputs.dim() < 2: #If passed 1D outputs/targets
        outputs = outputs.view(1, outputs.shape[0])
    if targets.dim() < 2:
        targets = targets.view(1, targets.shape[0])
    ep = torch.abs(targets - outputs[:, ::2])
    loss = outputs[:, 1::2] * ((ep/outputs[:, 1::2]) * torch.erf((ep/(np.sqrt(2)*outputs[:, 1::2])))
                                + np.sqrt(2/np.pi) * torch.exp(-ep**2 / (2*outputs[:, 1::2]**2))
                                - 1/np.sqrt(np.pi))
    return loss

class GaussianContinuousRankedProbabilityScore(torchmetrics.Metric):
    '''
        Like torchmetrics.regression.crps.ContinuousRankedProbabilityScore but takes the mean
        and variance of a Gaussian instead of an ensemble of predictions as its input.
        From https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    '''
    is_differentiable = True # Is the metric differentiable? Yes, the CRPS is differentiable.
    higher_is_better = False # Is a higher metric better (e.g. accuracy)? No, CRPS is like MAE where lower is better.
    full_state_update = False # Does .update() need to know the global metric state? No, each score is independent.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default = torch.tensor(0), dist_reduce_fx='mean')
    def update(self, preds, target):
        self.score = crps(preds, target)
    def compute(self):
        return self.score

class ProbabilisticMeanAbsoluteError(torchmetrics.Metric):
    '''
        A version of MAE used as a metric for dual-output models trained with the CRPS.
        Splits out the means of the distributions and uses them to calculate the MAE.
        From https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    '''
    is_differentiable = True # Is the metric differentiable? Yes, the MAE is differentiable.
    higher_is_better = False # Is a higher metric better (e.g. accuracy)? No.
    full_state_update = False # Does .update() need to know the global metric state? No, each score is independent.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("score", default = torch.tensor(0), dist_reduce_fx='mean')
    def update(self, preds, target):
        self.score = torch.nn.functional.l1_loss(preds[:, ::2], target)
    def compute(self):
        return self.score

def load(modelname = None, checkpoint = None):
    '''
        Loads a pretrained PRIME model, either one of the included models (PRIME, PRIME-SH, PRIME-PS, PRIME-PS-GEO, PRIME-PS-MMS) or a user supplied model checkpoint.
        
        :param [str] modelname: Name of included pretrained model to load. Options include 'PRIME' (solar wind model), 'PRIME-SH' (magnetosheath model), or 'PRIME-PS' (plasmasheet model). For 'PRIME-PS', can specify Geotail-trained ('PRIME-PS-GEO') or MMS-trained ('PRIME-PS-MMS') models (if 'PRIME-PS' specified, MMS model loaded by default).
        :param [str or Path] checkpoint: Path to user-supplied torch checkpoint .ckpt file. If specified, must also specify configuration path. Overrides supplied modelname.
        :returns [SWRegressor] model: Pretrained PRIME-like model.
    '''
    if (checkpoint!=None): # Is a checkpoint specified? Fires before modelname case
        model = SWRegressor.load_from_checkpoint(checkpoint)
        model.eval() # Freeze the dropout and weights
        return model
    elif (modelname!=None):
        resource_path = importlib.resources.path('primesw', f'{modelname}.ckpt') # Options include PRIME, PRIME-SH, and PRIME-PS
        with resource_path as checkpoint_file:
            model = SWRegressor.load_from_checkpoint(checkpoint_file)
        model.eval() # Freeze the dropout and weights
        return model
    else:
        raise ValueError("Must specify either modelname or checkpoint")