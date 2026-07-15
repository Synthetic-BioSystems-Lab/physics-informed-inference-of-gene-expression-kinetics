import numpy as np
import matplotlib.pyplot as plt
import torch

time = np.load('Experimental_Data/filtered_GFP_time.npy')
GFP = np.load('Experimental_Data/filtered_GFP_values.npy')

sim_time = np.load('Simulations/sim_TU_data/time_culture.npy')
sim_yfp = np.load('Simulations/sim_TU_data/yfp_culture.npy')

print(sim_time.shape, sim_yfp[0].shape)
# print(sim_time)

# plt.plot(sim_time, sim_yfp[0])
# plt.show()

# for i in range(3):
#     dy = np.gradient(sim_yfp[i], sim_time)

#     plt.figure()
#     plt.plot(sim_time, sim_yfp[i])
#     plt.show()

#     plt.plot(sim_time, dy)
#     plt.show()

for i in range(3):
    dy = np.gradient(GFP[:, i], time)

    plt.figure()
    plt.plot(time, GFP[:, i])

    plt.plot(time, dy)
    plt.show()