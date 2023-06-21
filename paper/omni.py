
# Do proper OMNI propagation to MMS dataset

import numpy as np
import pandas as pd

EARTH_RADIUS = 6378 #Re in km

omni_data = pd.read_hdf('sw_data.h5', key = 'wind_kp_omni')
mms_data = pd.read_hdf('sw_data.h5', key = 'mms_target')

omni_ind = np.zeros(len(mms_data))
shift = np.zeros(len(mms_data))

for i in np.arange(len(mms_data)):
    if (mms_data['regid'][i]==2):
        nx = omni_data['Phase_n_x']
        ny = omni_data['Phase_n_y']
        nz = omni_data['Phase_n_z']

        Rox = omni_data['BSN_X'] * EARTH_RADIUS #Convert from Re to km
        Roy = omni_data['BSN_Y'] * EARTH_RADIUS
        Roz = omni_data['BSN_Z'] * EARTH_RADIUS

        Vx = omni_data['VX_GSE']
        Vy = omni_data['VY_GSE']
        Vz = omni_data['VZ_GSE']

        Rdx = mms_data['Px_gse'][i]
        Rdy = mms_data['Py_gse'][i]
        Rdz = mms_data['Pz_gse'][i]

        mms_time = mms_data['time'][i]
        omni_time = omni_data['time']

        delta_t = nx * (Rdx - Rox) / (nx * Vx) + ny * (Rdy - Roy) / (ny * Vy) + nz * (Rdz - Roz) / (nz * Vz)

        omni_ind[i] = np.argmin(np.abs(omni_time + delta_t - mms_time))
        shift[i] = np.abs(omni_time[i] + delta_t[i] - mms_time)
    else:
        omni_ind[i] = 0
        shift[i] = -1
    print('Processing '+str(100*i/len(mms_data))[0:5]+'% Complete', end='\r')
omni_data_shift = omni_data[omni_ind]
omni_data_shift['timeshift'] = shift
omni_data_shift.to_hdf('sw_data.h5', key = 'wind_kp_shift')
