import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

OD600_data = pd.read_excel('Experimental_Data/Blank_OD600_GFP_4-2-2026.xlsx')
OD600_plots = OD600_data.columns.difference(['Kinetic read'])
OD600_data['Kinetic read'] = OD600_data['Kinetic read'].apply(
    lambda t: t.hour + t.minute / 60 + t.second / 3600)

OD600_data.plot('Kinetic read', OD600_plots)
plt.xlabel('Time (hours)')
plt.ylabel('OD600')
plt.legend(['A1', 'A2', 'A3'])
plt.show()



GFP_data = pd.read_excel('Experimental_Data/BlankGFP_4-2-2026.xlsx')
GFP_plots = GFP_data.columns.difference(['Kinetic read'])
GFP_data['Kinetic read'] = GFP_data['Kinetic read'].apply(
    lambda t: t.hour + t.minute / 60 + t.second / 3600)

GFP_data.plot('Kinetic read', GFP_plots)
plt.xlabel('Time (hours)')
plt.ylabel('GFP')
plt.legend(['A1', 'A2', 'A3'])
plt.show()



dts = np.diff(GFP_data['Kinetic read'])
dt = np.mean(dts)
fs = 1 / dt
b, a = butter(2, 0.5, fs=fs)

filtered_GFP = GFP_data.copy()
for col in GFP_plots:
    filtered_GFP[col] = filtfilt(b, a, GFP_data[col])

filtered_GFP[GFP_plots] = filtered_GFP[GFP_plots].where(filtered_GFP[GFP_plots] > 0, 0)

filtered_GFP.plot('Kinetic read', GFP_plots)
plt.xlabel('Time (hours)')
plt.ylabel('GFP')
plt.legend(['A1', 'A2', 'A3'])
plt.show()

filtered_GFP_time = filtered_GFP['Kinetic read'].to_numpy()
GFP_values = filtered_GFP[GFP_plots].to_numpy()

np.save('Experimental_Data/filtered_GFP_time.npy', filtered_GFP_time)
np.save('Experimental_Data/filtered_GFP_values.npy', GFP_values)
