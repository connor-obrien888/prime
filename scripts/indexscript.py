
import pandas as pd
import numpy as np

input_data = pd.read_hdf("/glade/u/home/cobrien/data/combined_data.h5", key = "wind_1min_complete")
target_data = pd.read_hdf("/glade/u/home/cobrien/data/combined_data.h5", key = "mms_1min_labeled")

target_data['input_idx'] = np.nan
for i, idx in enumerate(target_data.index):
    target_time = target_data.loc[idx, 'Epoch'].strftime('%Y%m%d %H:%M:%S') # Used to get correct input window
    input_idx = input_data.loc[input_data['Epoch'] == target_data.loc[idx, 'Epoch'], :].index[0]
    target_data.loc[idx, 'input_idx'] = input_idx

target_data.to_hdf("/glade/u/home/cobrien/data/combined_data.h5", key = "mms_1min_labeled_indexed")