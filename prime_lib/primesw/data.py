import torch
from torch.utils.data import Dataset
import lightning.pytorch as pl
import numpy as np
import pandas as pd
from loguru  import logger


class SWDataset(Dataset):
    def __init__(
        self,
        target_features,
        input_features,
        position_features,
        interp_flags,
        cadence,
        interpolate = False,
        window = 1,
        stride = 0,
        interp_frac = 0,
        input_normalizations = None,
        target_normalizations = None,
        position_normalizations = None,
        min_time = pd.to_datetime('20150902 00:00:00+0000'), # Earliest MMS timestamp,
        max_time = pd.to_datetime('20250101 00:00:00+0000'), # Latest MMS timestamp,
        input_data = None,
        target_data = None,
        position_data = None,
        datastore = "~/data/prime/sw_data.h5",
        in_key = "wind_1min_complete",
        tar_key = "mms_1min_labeled",
    ):
        super().__init__()

        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Position of the target for encoder
        self.interp_flags = interp_flags # Keys where input data interpolation flags are stored
        self.cadence = cadence # Cadence of data
        self.interpolate = interpolate
        self.window = window
        self.stride = stride
        self.interp_frac = interp_frac
        self.input_normalizations = input_normalizations
        self.target_normalizations = target_normalizations
        self.position_normalizations = position_normalizations
        self.min_time = min_time
        self.max_time = max_time
        self.datastore = datastore
        self.in_key = in_key
        self.tar_key = tar_key

        if input_data is None: #Load the data
            self.input_data = pd.read_hdf(self.datastore, key = self.key, mode = "r")
            #TODO: redo making the input and target data
        else:
            self.input_data = input_data
            self.target_data = target_data
            self.position_data = position_data
        if (max_time > self.target_data['Epoch'].max()):
            logger.warning(f"The max_time passed to SWDataset is larger than the latest entry in target_data")
        if (min_time < self.target_data['Epoch'].min()):
            logger.warning(f"The min_time passed to SWDataset is smaller than the first entry in target_data")
        self.target_data = self.target_data.loc[
            (self.target_data['Epoch'] <= max_time)&
            (self.target_data['Epoch'] >= min_time), :
        ] #Cut time of base data to be between min and max times

        #Normalize the target, input, and position data
        if self.target_normalizations is not None: #Should we do target normalization?
            self.target_scaled = self.target_data.loc[:, self.target_features]
            for feature in self.target_features:
                self.target_scaled[feature] = (self.target_scaled[feature] - self.target_normalizations[feature][0])/self.target_normalizations[feature][1]
        else:
            self.target_scaled = self.target_data.loc[:, self.target_features]
        if self.input_normalizations is not None: #Should we do input normalization?
            self.input_scaled = self.input_data.loc[:, self.input_features] # Here we use the full dataset so that we can 
            for feature in self.input_features:
                self.input_scaled[feature] = (self.input_scaled[feature] - self.input_normalizations[feature][0])/self.input_normalizations[feature][1]
        else:
            self.input_scaled = self.input_data.loc[:, self.input_features]
        if self.interpolate: #Interpolate over nans?
            self.input_scaled = self.input_scaled.interpolate(method='linear')
        if self.position_normalizations is not None: #Should we do target normalization?
            self.position_scaled = self.position_data.loc[:, self.position_features]
            for feature in self.position_features:
                self.position_scaled[feature] = (self.position_scaled[feature] - self.position_normalizations[feature][0])/self.position_normalizations[feature][1]
        else:
            self.position_scaled = self.position_data.loc[:, self.position_features]

        #Split the input data into windows and get the right targets
        # input_arr = np.zeros((len(self.target_data), self.window, len(self.input_features)))
        input_list = []
        # target_arr = np.zeros((len(self.target_data), len(self.target_features)))
        target_list = []
        # position_arr = np.zeros((len(self.target_data), len(self.position_features)))
        position_list = []
        times_list = []
        logger.info(f"Segmenting input data.")
        for i, idx in enumerate(self.target_data.index):
            if (np.isnan(self.target_scaled.loc[idx, :].values).any())|(np.isnan(self.position_scaled.loc[idx, :].values).any())|(np.isnan(self.target_data.loc[idx, 'input_idx'])): # Skip targets that are nans
                continue
            target_time = self.target_data.loc[idx, 'Epoch'].strftime('%Y%m%d %H:%M:%S') # Used to get correct input window
            input_idx = self.target_data.loc[idx, 'input_idx'] # Precomputed version of self.input_data.loc[self.input_data['Epoch'] == self.target_data.loc[idx, 'Epoch'], :].index[0]
            segment = self.input_scaled.loc[(input_idx - self.window - self.stride + 1):(input_idx - self.stride), :]
            interp_arr = self.input_data.loc[(input_idx - self.window - self.stride + 1):(input_idx - self.stride), self.interp_flags]
            interp_lengths = [np.sum(interp_arr[key]) for key in self.interp_flags]
            if len(segment) != self.window: #Skip any intervals that have non-full input windows
                logger.info(f"Non-full interval length {len(segment)} lower bound {self.input_data.loc[segment.index, 'Epoch'].min()}, upper bound {self.input_data.loc[segment.index, 'Epoch'].max()}")
                # raise(TypeError(f"Segment wrong size, goofy: {len(segment)}"))
                continue
            if (np.max(interp_lengths)/len(segment))>interp_frac: # Is more than interp_frac of the input data interpolated?
                continue
            # target_arr[i, :] = self.target_scaled.loc[idx, :].values
            target_list.append(self.target_scaled.loc[idx, :].values)
            times_list.append(target_time)
            # input_arr[i, :, :] = self.input_scaled.loc[input_mask, :]
            input_list.append(segment.values)
            # position_arr[i, :] = self.position_scaled.loc[idx, :].values
            position_list.append(self.position_scaled.loc[idx, :].values)

        # self.input_data = torch.tensor(input_arr, dtype = torch.float32) # Turn numpy arrays into tensors
        self.input_data = torch.tensor(np.array(input_list), dtype = torch.float32)
        # self.target_data = torch.tensor(target_arr, dtype = torch.float32)
        self.target_data = torch.tensor(np.array(target_list), dtype = torch.float32)
        # self.position_data = torch.tensor(position_arr, dtype = torch.float32)
        self.position_data = torch.tensor(np.array(position_list), dtype = torch.float32)
        self.target_timestamps = times_list
        # self.target_timestamps = self.raw_data.iloc[(self.window+self.stride-1):].loc[:,'Epoch'].to_numpy() # Store the times of each target for QA

    def __len__(self): # A torch dataset must have a __len__ method
        return self.input_data.shape[0]
    
    def __getitem__(self, idx): # A torch dataset must have a __getitem__ method
        return self.input_data[idx], self.position_data[idx], self.target_data[idx], self.target_timestamps[idx]

    def __str__(self): # A torch dataset MIGHT need a __str__ method
        output = ""
        for k, v in self.__dict__.items():
            output += f"{k}: {v}\n"
        return output

class SWDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target_features,
        input_features,
        position_features,
        interp_flags,
        cadence,
        interpolate,
        region,
        cuts = None,
        window = None,
        stride = None,
        interp_frac = None,
        trn_bounds = None, #TODO: add more ways to define the train/val/test sets
        val_bounds = None,
        tst_bounds = None,
        batch_size = 32,
        num_workers = 1,
        datastore = "~/data/prime/sw_data.h5",
        in_key = "wind_1min_complete",
        tar_key = "mms_1min_labeled",
        scaler_type = 'STD',
    ):
        super().__init__()
        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Positions of the targets added to the inputs
        self.interp_flags = interp_flags # Keys where input data interpolation flags are stored
        self.cadence = cadence # Cadence of data
        self.interpolate = interpolate # Interpolate over nans?
        self.region = region # Region of space trained to (e.g. 'solar wind', 'magnetosheath')
        self.cuts = cuts # How to cut data (e.g. stability, solar wind table)
        self.batch_size = batch_size # Training batch size
        self.num_workers = num_workers # Number of workers for loading data
        self.scaler_type = scaler_type # Type of scaling to apply to input and target data

        if window is not None:
            self.window = window
        else:
            self.window = 1 # One entry input timeseries (non-recurrent)
        if stride is not None:
            self.stride = stride
        else:
            self.stride = 0 # Same time as input data
        if interp_frac is not None:
            self.interp_frac = interp_frac
        else:
            self.interp_frac = 1 # Accepts all interpolated data

        # Load the data and define normalization terms
        self.datastore = datastore # Open the HDF with combined target and input data
        self.in_key = in_key # Key in HDF with input data
        self.tar_key = tar_key # Key in HDF with target data
        self.raw_in_data = pd.read_hdf(datastore, key = self.in_key, mode = "r") # Load the HDF of data with no cuts
        self.raw_tar_data = pd.read_hdf(datastore, key = self.tar_key, mode = "r")
        
        self.target_data = self.raw_tar_data.loc[(self.raw_tar_data['modified_named_label'] == self.region), :] # Isolate the desired region/type of solar wind and store as targets (LEAVE ALL FEATURES IN FOR MORE CUTS LATER)
        self.position_data = self.raw_tar_data.loc[(self.raw_tar_data['modified_named_label'] == self.region), :] # Isolate the desired region/type of solar wind and store as targets (LEAVE ALL FEATURES IN FOR MORE CUTS LATER)
        if self.cuts is not None: # Are we cutting the dataset for only stable regions, or other cuts?
            for cut in self.cuts:
                if cut == 'stability': # Only train on data where MMS is in same region for 15+ minutes
                    logger.info(f"Dataset cut {cut}")
                    self.target_data = self.target_data.loc[self.target_data['stable'] == 1, :]
                    self.position_data = self.position_data.loc[self.position_data['stable'] == 1, :]
                if cut == 'solar wind table': # Only use data with the solar wind energy-azimuth table
                    logger.info(f"Dataset cut {cut}")
                    self.target_data = self.target_data.loc[self.target_data['SW_table'] == 1, :]
                    self.position_data = self.position_data.loc[self.position_data['SW_table'] == 1, :]
                if cut.startswith('density_despike'): # Developed to remove density spikes (>Ncm-3 for density_despike_N) in Geotail data.
                    logger.info(f"Dataset cut {cut}")
                    threshold = int(cut.split('_')[-1])
                    self.target_data = self.target_data.loc[self.target_data['N'] <= threshold, :]
                    self.position_data = self.position_data.loc[self.position_data['N'] <= threshold, :]

        tar_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std) or (mean, iqr)
        for feature in self.target_features:
            if self.scaler_type == 'STD':
                tar_norm_tup_list.append((self.target_data[feature].mean(), self.target_data[feature].std()))
            if self.scaler_type == 'IQR':
                tar_norm_tup_list.append((np.nanpercentile(self.target_data[feature],50), # Median
                                          np.nanpercentile(self.target_data[feature], 75) - np.nanpercentile(self.target_data[feature], 25))) # Interquartile range
        self.target_normalizations = dict(zip(self.target_features, tar_norm_tup_list)) # Dictionary of information used to do normalization

        in_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std) or (mean, iqr)
        for feature in self.input_features:
            if self.scaler_type == 'STD':
                in_norm_tup_list.append((self.raw_in_data[feature].mean(), self.raw_in_data[feature].std()))
            if self.scaler_type == 'IQR':
                in_norm_tup_list.append((np.nanpercentile(self.raw_in_data[feature],50), # Median
                                         np.nanpercentile(self.raw_in_data[feature], 75) - np.nanpercentile(self.raw_in_data[feature], 25))) # Interquartile range
        self.input_normalizations = dict(zip(self.input_features, in_norm_tup_list)) # Dictionary of information used to do normalization

        pos_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std) or (mean, iqr)
        for feature in self.position_features: #For the purposes of normalization, the position features count as inputs
            if self.scaler_type == 'STD':
                pos_norm_tup_list.append((self.position_data[feature].mean(), self.position_data[feature].std()))
            if self.scaler_type == 'IQR':
                pos_norm_tup_list.append((np.nanpercentile(self.position_data[feature],50), # Median
                                          np.nanpercentile(self.position_data[feature], 75) - np.nanpercentile(self.position_data[feature], 25))) # Interquartile range
        self.position_normalizations = dict(zip(self.position_features, pos_norm_tup_list)) # Dictionary of information used to do normalization
        
        # Bounds of train/test/validation sets
        if trn_bounds is not None:
            self.trn_bounds = [
                pd.to_datetime(trn_bounds[0]),
                pd.to_datetime(trn_bounds[1])
            ]
        else:
            self.trn_bounds = [
                pd.to_datetime('20150902 00:00:00+0000'), # First 60% of MMS dataset by default
                pd.to_datetime('20210411 00:00:00+0000')
            ]
        if val_bounds is not None:
            self.val_bounds = [
                pd.to_datetime(val_bounds[0]),
                pd.to_datetime(val_bounds[1])
            ]
        else:
            self.val_bounds = [
                pd.to_datetime('20210411 00:00:00+0000'), # next 20% of MMS dataset by default
                pd.to_datetime('20230222 00:00:00+0000')
            ]
        if tst_bounds is not None:
            self.tst_bounds = [
                pd.to_datetime(tst_bounds[0]),
                pd.to_datetime(tst_bounds[1])
            ]
        else:
            self.tst_bounds = [
                pd.to_datetime('20230222 00:00:00+0000'), # last 20% of MMS dataset by default
                pd.to_datetime('20250101 00:00:00+0000')
            ]

    def setup(self, stage=None): #Sets up train, validation, test datasets
        self.trn_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
            self.interp_flags,
            self.cadence,
            interpolate = self.interpolate,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.trn_bounds[0],
            max_time = self.trn_bounds[1],
            input_data = self.raw_in_data,
            target_data = self.target_data,
            position_data = self.position_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Train dataloader is ready. Dataset size: {len(self.trn_ds)}")

        self.val_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
            self.interp_flags,
            self.cadence,
            interpolate = self.interpolate,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.val_bounds[0],
            max_time = self.val_bounds[1],
            input_data = self.raw_in_data,
            target_data = self.target_data,
            position_data = self.position_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Validation dataloader is ready. Dataset size: {len(self.val_ds)}")

        self.tst_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
            self.interp_flags,
            self.cadence,
            interpolate = self.interpolate,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.tst_bounds[0],
            max_time = self.tst_bounds[1],
            input_data = self.raw_in_data,
            target_data = self.target_data,
            position_data = self.position_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Test dataloader is ready. Dataset size: {len(self.tst_ds)}")

    def train_dataloader(self): # A torch datamodule must have a train_dataloader method
        return torch.utils.data.DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self): # A torch datamodule must have a val_dataloader method
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )

    def test_dataloader(self): # A torch datamodule must have a test_dataloader method
        return torch.utils.data.DataLoader(
            self.tst_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
        )