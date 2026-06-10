'''

E. Coli (cells/mL)/OD600 -> (0.6-2) e9 -> https://book.bionumbers.org/what-is-the-concentration-of-bacterial-cells-in-a-saturated-culture/

'''

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit

def logistic_growth(t, x_0, k_g, K, t_lag):
    t = np.asarray(t)  # Ensure t is an array for vectorized operations
    return np.where(
        t < t_lag, 
        x_0,  # Before t_lag, return the initial value x_0
        K / (1 + ((K - x_0) / x_0) * np.exp(-k_g * (t - t_lag)))  # Logistic growth after t_lag
    )

OD600_data = pd.read_excel('Experimental_Data/Blank_OD600_GFP_4-2-2026.xlsx')
OD600_plots = OD600_data.columns.difference(['Kinetic read'])
OD600_data['Kinetic read'] = OD600_data['Kinetic read'].apply(
    lambda t: t.hour + t.minute / 60 + t.second / 3600)

OD600_data.plot('Kinetic read', OD600_plots)
plt.xlabel('Time (hours)')
plt.ylabel('OD600')
plt.legend(['A1', 'A2', 'A3'])
plt.show()

cell_counts = OD600_data.copy()
cell_counts[OD600_plots] = cell_counts[OD600_plots].apply(lambda unit: unit * 1e9)
cell_counts[OD600_plots] = cell_counts[OD600_plots].apply(lambda unit: unit.where(unit > 0, 0))

cell_counts.plot('Kinetic read', OD600_plots)
plt.xlabel('Time (hours)')
plt.ylabel('Cell Count (cells/mL)')
plt.legend(['A1', 'A2', 'A3'])
plt.show()

plt.figure()
df = pd.DataFrame()
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

for i in range(5):#len(OD600_plots)):

    x_0_initial = cell_counts[OD600_plots[i]].iloc[0]
    p0 = [x_0_initial, 0.01, 0.5e9, 6]  
    bounds = ([0, 1e-10, 1e6, 0], [1e10, 1, 1e12, 12])

    popt, pcov = curve_fit(logistic_growth, cell_counts['Kinetic read'], cell_counts[OD600_plots[i]], p0=p0, bounds=bounds)

    t = np.linspace(0, 20, 1000)
    y_fit = logistic_growth(t, *popt)

    # Calculating R^2
    y_pred = logistic_growth(cell_counts['Kinetic read'], *popt)
    ss_res = np.sum( (cell_counts[OD600_plots[i]] - y_pred)**2 )
    ss_tot = np.sum( (cell_counts[OD600_plots[i]] - np.mean(cell_counts[OD600_plots[i]]))**2 )
    r_squared = 1 - (ss_res/ss_tot)

    color = color_cycle[i % len(color_cycle)]        
    plt.plot(cell_counts['Kinetic read'], cell_counts[OD600_plots[i]], label='Data', color = color, alpha=0.7)
    plt.plot(t, y_fit, '--', label='Fit', color= color, alpha=0.7)

    # print(f'Fitted parameters: x_0={popt[0]} cells/mL, k_g={popt[1]} h^-1, K={popt[2]} cells/mL, t_lag={popt[3]} hours')
    temp_df = pd.DataFrame({'x_0': [popt[0]], 'k_g': [popt[1]], 'K': [popt[2]], 't_lag': [popt[3]], 'R_squared': r_squared})
    df = pd.concat([df, temp_df], ignore_index=True)


plt.xlabel('Time (hours)')
plt.ylabel('Cell Count (cells/mL)')
plt.title(f'Logistic Fit (R^2={r_squared:.3f})')
# plt.legend()
plt.show()
print(df)
print(f'Average fitted parameters: x_0 = {df["x_0"].mean():.4e} cells/mL, k_g = {df["k_g"].mean():.4f} h^-1, K = {df["K"].mean():.4e} cells/mL, t_lag = {df["t_lag"].mean():.4f} hours')