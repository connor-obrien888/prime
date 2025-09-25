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
        interpolate = False,
        window = 1,
        stride = 0,
        interp_frac = 0,
        input_normalizations = None,
        target_normalizations = None,
        position_normalizations = None,
        min_time = pd.to_datetime('20150902 00:00:00+0000'), # Earliest MMS timestamp,
        max_time = pd.to_datetime('20250101 00:00:00+0000'), # Latest MMS timestamp,
        raw_data = None,
        target_data = None,
        position_data = None,
        datastore = "~/data/prime/sw_data.h5",
        key = "mms_wind_combined",
    ):
        super().__init__()

        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Position of the target for encoder
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
        self.key = key

        if raw_data is None: #Load the data
            self.raw_data = pd.read_hdf(self.datastore, key = self.key, mode = "r")
            #TODO: redo making the input and target data
        else:
            self.raw_data = raw_data
            self.target_data = target_data
            self.position_data = position_data
        if (max_time > self.raw_data['Epoch'].max()):
            logger.warning(f"The max_time passed to SWDataset is larger than the latest entry in raw_data")
        if (min_time < self.raw_data['Epoch'].min()):
            logger.warning(f"The min_time passed to SWDataset is smaller than the first entry in raw_data")
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
            self.input_scaled = self.raw_data.loc[:, self.input_features] # Here we use the full dataset so that we can 
            for feature in self.input_features:
                self.input_scaled[feature] = (self.input_scaled[feature] - self.input_normalizations[feature][0])/self.input_normalizations[feature][1]
        else:
            self.input_scaled = self.raw_data.loc[:, self.input_features]
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
        for i, idx in enumerate(self.target_data.index):
            if np.isnan(self.target_scaled.loc[idx, :].values).any(): # Skip targets that are nans
                continue
            target_time = self.target_data.loc[idx, 'Epoch'] # Used to get correct input window
            input_mask = (
                (self.raw_data['Epoch'] > (target_time - pd.Timedelta(self.window, unit = 'minutes') - pd.Timedelta(self.stride, unit = 'minutes'))) &
                (self.raw_data['Epoch'] <= (target_time - pd.Timedelta(self.stride, unit = 'minutes')))
            )
            if ((self.raw_data.loc[input_mask, 'interped_swe'].sum()/self.window < self.interp_frac)& # Do not store 
                (self.raw_data.loc[input_mask, 'interped_mfi'].sum()/self.window < self.interp_frac)):
                continue
            segment = self.input_scaled.loc[input_mask, :]
            if len(segment) != self.window: #Skip any intervals that have non-full input windows
                logger.info(f"Non-full interval lower bound {self.raw_data.loc[input_mask, 'Epoch'].min()}, upper bound {self.raw_data.loc[input_mask, 'Epoch'].max()}")
                continue
            # target_arr[i, :] = self.target_scaled.loc[idx, :].values
            target_list.append(self.target_scaled.loc[idx, :].values)
            times_list.append(target_time)
            # input_arr[i, :, :] = self.input_scaled.loc[input_mask, :]
            input_list.append(segment)
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
        key = "mms_wind_combined",
    ):
        super().__init__()
        self.target_features = target_features # Features model uses as targets
        self.input_features = input_features # Features model uses as input
        self.position_features = position_features # Positions of the targets added to the inputs
        self.cadence = cadence # Cadence of data
        self.interpolate = interpolate # Interpolate over nans?
        self.region = region # Region of space trained to (e.g. 'solar wind', 'magnetosheath')
        self.cuts = cuts # How to cut data (e.g. stability, solar wind table)
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
        #NOTE: raw_data is essentially input_data. In SWDataset, when scaling the data we downselect to just the input features.
        self.target_data = self.raw_data.loc[(self.raw_data['modified_named_label'] == self.region), :] # Isolate the desired region/type of solar wind and store as targets (LEAVE ALL FEATURES IN FOR MORE CUTS LATER)
        self.position_data = self.raw_data.loc[(self.raw_data['modified_named_label'] == self.region), :] # Isolate the desired region/type of solar wind and store as targets (LEAVE ALL FEATURES IN FOR MORE CUTS LATER)
        if self.cuts is not None: # Are we cutting the dataset for only stable regions, or other cuts?
            for cut in self.cuts:
                if cut == 'stability': # Only train on data where MMS is in same region for 15+ minutes
                    self.target_data = self.target_data[self.target_data['stable'] == 1, :]
                    self.position_data = self.position_data[self.position_data['stable'] == 1, :]
                if cut == 'solar wind table': # Only use data with the solar wind energy-azimuth table
                    self.target_data = self.target_data[self.target_data['SW_table'] == 1, :]
                    self.position_data = self.position_data[self.position_data['SW_table'] == 1, :]

        tar_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.target_features:
            tar_norm_tup_list.append((self.target_data[feature].mean(), self.target_data[feature].std())) #TODO: change this based on some config (like, the second value could be the IQR)
        self.target_normalizations = dict(zip(self.target_features, tar_norm_tup_list)) # Dictionary of information used to do normalization

        in_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.input_features:
            in_norm_tup_list.append((self.raw_data[feature].mean(), self.raw_data[feature].std())) #TODO: change this based on some config (like, the second value could be the IQR)
        self.input_normalizations = dict(zip(self.input_features, in_norm_tup_list)) # Dictionary of information used to do normalization

        pos_norm_tup_list = [] #List of tuples used to store normalization values. Typically this is (mean, std)
        for feature in self.position_features: #For the purposes of normalization, the position features count as inputs
            pos_norm_tup_list.append((self.position_data[feature].mean(), self.position_data[feature].std()))
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
            interpolate = self.interpolate,
            window = self.window,
            stride = self.stride,
            interp_frac = self.interp_frac,
            target_normalizations = self.target_normalizations,
            input_normalizations = self.input_normalizations,
            position_normalizations = self.position_normalizations,
            min_time = self.trn_bounds[0],
            max_time = self.trn_bounds[1],
            raw_data = self.raw_data,
            target_data = self.target_data,
            position_data = self.position_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Train dataloader is ready. Dataset size: {len(self.trn_ds)}")

        self.val_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
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
            raw_data = self.raw_data,
            target_data = self.target_data,
            position_data = self.position_data,
        )
        if stage == 'fit' or stage is None:
            logger.info(f"Validation dataloader is ready. Dataset size: {len(self.val_ds)}")

        self.tst_ds = SWDataset(
            self.target_features,
            self.input_features,
            self.position_features,
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
            raw_data = self.raw_data,
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