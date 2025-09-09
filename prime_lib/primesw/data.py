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
        cadence,
        window = None,
        stride = None,
        interp_frac = None,
        input_normalizations = None,
        target_normalizations = None,
        position_normalizations = None,
        min_time = None,
        max_time = None,
        raw_data = None,
        datastore = "~/data/prime/sw_data.h5",
        key = "mms_wind_combined",
    ):
        super().__init__()

        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Position of the target for encoder
        self.cadence = cadence # Cadence of data
        self.input_normalizations = input_normalizations
        self.target_normalizations = target_normalizations
        self.position_normalizations = position_normalizations

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
        if min_time is not None:
            self.min_time = min_time
        else:
            self.min_time = pd.to_datetime('20150902 00:00:00+0000') # Earliest MMS timestamp
        if min_time is not None:
            self.min_time = min_time
        else:
            self.min_time = pd.to_datetime('20250101 00:00:00+0000') # Latest MMS timestamp

        self.datastore = datastore
        self.key = key
        if raw_data is None: #Load the data
            self.raw_data = pd.read_hdf(self.datastore, key = self.key, mode = "r")
        else:
            self.raw_data = raw_data
        if (max_time > self.raw_data['time'].max()):
            logger.warning(f"The max_time passed to SWDataset is larger than the latest entry in raw_data")
        if (min_time < self.raw_data['time'].min()):
            logger.warning(f"The min_time passed to SWDataset is smaller than the first entry in raw_data")
        self.raw_data = self.raw_data.loc[
            (self.raw_data['time'] <= max_time)&
            (self.raw_data['time'] >= min_time), :
        ] #Cut time of base data to be between min and max times

        #Normalize the target, input, and position data
        if self.target_normalizations is not None: #Should we do target normalization?
            target_scaled = self.raw_data.loc[:, self.target_features]
            for feature in self.target_features:
                target_scaled[feature] = (target_scaled[feature] - self.target_normalizations[feature][0])/self.target_normalizations[feature][1]
        else:
            target_scaled = self.raw_data.loc[:, self.target_features]
        if self.input_normalizations is not None: #Should we do input normalization?
            input_scaled = self.raw_data.loc[:, self.input_features]
            for feature in self.input_features:
                input_scaled[feature] = (input_scaled[feature] - self.input_normalizations[feature][0])/self.input_normalizations[feature][1]
        else:
            input_scaled = self.raw_data.loc[:, self.input_features]
        if self.position_normalizations is not None: #Should we do target normalization?
            position_scaled = self.raw_data.loc[:, self.position_features]
            for feature in self.position_features:
                position_scaled[feature] = (position_scaled[feature] - self.position_normalizations[feature][0])/self.position_normalizations[feature][1]
        else:
            position_scaled = self.raw_data.loc[:, self.position_features]
        
        #Split the input data into windows and get the right targets
        input_arr = np.zeros((len(input_scaled)-(self.window+self.stride-1), self.window, len(self.input_features)))
        target_arr = np.zeros((len(input_scaled)-(self.window+self.stride-1), len(self.target_features)))
        position_arr = np.zeros((len(input_scaled)-(self.window+self.stride-1), len(self.position_features))) # A temporary array to hold the target position in, then add in later
        # logger.info(f"Length of target_arr: {len(target_arr)}")
        for i in np.arange(len(input_scaled)-(self.window+self.stride-1)):
            input_arr[i,:,:] = input_scaled.iloc[i:(i+self.window), :].values # Move the window through the input data
            position_arr[i,:] = position_scaled.iloc[(i+self.window+self.stride-1), :].values # Get the target position associated with the target
            target_arr[i,:] = target_scaled.iloc[(i+self.window+self.stride-1), :].values # Get the target stride away from the last entry in the timeseries
        self.input_data = torch.tensor(input_arr) # Turn numpy arrays into tensors
        self.target_data = torch.tensor(target_arr)
        self.position_data = torch.tensor(position_arr)
        self.target_timestamps = [time.strftime('%Y%m%d %H:%M:%S+0000') for time in self.raw_data.iloc[(self.window+self.stride-1):].loc[:,'time']]
        # self.target_timestamps = self.raw_data.iloc[(self.window+self.stride-1):].loc[:,'time'].to_numpy() # Store the times of each target for QA

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
        cadence,
        window = None,
        stride = None,
        interp_frac = None,
        trn_bounds = None, #TODO: add more ways to define the train/val/test sets
        val_bounds = None,
        tst_bounds = None,
        batch_size = 32,
        num_workers = 1,
        datastore = "~/data/prime/sw_data.h5",
        key = "mms_wind_combined",
    ):
        super().__init__()
        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Positions of the targets added to the inputs
        self.cadence = cadence # Cadence of data
        self.batch_size = batch_size # Training batch size
        self.num_workers = num_workers # Number of workers for loading data

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
        self.key = key # Key in HDF with combined target and input data
        self.raw_data = pd.read_hdf(datastore, key = self.key, mode = "r") # Load the HDF of data with no cuts

        tar_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.target_features:
            tar_norm_tup_list.append((self.raw_data[feature].mean(), self.raw_data[feature].std())) #TODO: change this based on some config (like, the second value could be the IQR)
        self.target_normalizations = dict(zip(self.target_features, tar_norm_tup_list)) # Dictionary of information used to do normalization

        in_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.input_features:
            in_norm_tup_list.append((self.raw_data[feature].mean(), self.raw_data[feature].std())) #TODO: change this based on some config (like, the second value could be the IQR)
        self.input_normalizations = dict(zip(self.input_features, in_norm_tup_list)) # Dictionary of information used to do normalization

        pos_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.position_features: #For the purposes of normalization, the position features count as inputs
            pos_norm_tup_list.append((self.raw_data[feature].mean(), self.raw_data[feature].std()))
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
            self.cadence,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.trn_bounds[0],
            max_time = self.trn_bounds[1],
            raw_data = self.raw_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Train dataloader is ready. Dataset size: {len(self.trn_ds)}")

        self.val_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
            self.cadence,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.val_bounds[0],
            max_time = self.val_bounds[1],
            raw_data = self.raw_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Validation dataloader is ready. Dataset size: {len(self.val_ds)}")

        self.tst_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
            self.cadence,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.tst_bounds[0],
            max_time = self.tst_bounds[1],
            raw_data = self.raw_data,
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