import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/zacha/1_Projects/DL')
from PINN.PINN_v2 import PINN, PINN_module


filtered_GFP_time = np.load('Experimental_Data/filtered_GFP_time.npy')
filtered_GFP_values = np.load('Experimental_Data/filtered_GFP_values.npy')

for i in range(filtered_GFP_values.shape[1]):
    plt.plot(filtered_GFP_time, filtered_GFP_values[:, i])

plt.xlabel('Time (hours)')
plt.ylabel('GFP')
plt.show()

indices = np.linspace(0, filtered_GFP_values.shape[0] - 1, 51)
indices = np.round(indices).astype(int)


indices_time = np.linspace(0, filtered_GFP_time.shape[0] - 1, 51)
indices_time = np.round(indices_time).astype(int)

filtered_GFP_values = filtered_GFP_values[indices]
filtered_GFP_time = filtered_GFP_time[indices_time]

plt.figure()

for i in range(filtered_GFP_values.shape[1]):
    plt.plot(filtered_GFP_time, filtered_GFP_values[:, i])

plt.xlabel('Time (hours)')
plt.ylabel('GFP')
plt.show()

# Convert to PyTorch tensor
filtered_GFP_values = torch.tensor(filtered_GFP_values, dtype=torch.float32)

# Reshape to (36, 121, 1) assuming batch_size = 36 for 36 independent time series
filtered_GFP_values = filtered_GFP_values.T  # Transpose -> (36, 121) -> Add dim -> (36, 121, 1)

print(filtered_GFP_values.shape)

model = torch.load('Experiments/Simulated_Cultures/models/PINN_v2_model.pth', weights_only=False) 
ktl, kdil, mrna = model.run_NN(filtered_GFP_values)  

print(ktl, kdil, mrna)

# for i in range(filtered_GFP_values.shape[1]):

# import sys
# sys.path.insert(0, '/home/zacha/1_Projects/DL')

# import PINN.PINN_v2 as NN
# from pathlib import Path
# import shutil

# print(Path.cwd())

# save_direct ="Experiments/Simulated_Cultures/plots/5-30-2026"

# plots = Path(save_direct)
# plots.mkdir(exist_ok=True)

# for p in plots.iterdir():
#     if p.is_dir():
#         shutil.rmtree(p)
#     else:
#         p.unlink()

# start = time.time()

# X_lst = np.load("Simulations/sim_TU_data/yfp_culture.npy")
# Y_lst = np.load("Simulations/sim_TU_data/param_labels_culture.npy")

# accuracy_lst = []

# torch.manual_seed(308380)

# model = NN.PINN(save_direct, n_epochs=1001, p_epoch=250, lr=0.0063, weight_decay=0, 
#                     lambda_phys=0.001, hidden_dim=64, phys_start_epoch=100)
# model.fit(X_lst, Y_lst, batch_size=50)
# model.plot_loss()
# model.plot_accuracy()   
# acc = model.predict()
    
# accuracy_lst.append(acc)
        
# print('lambda physics: ', 0.001)
# print('Accuracy: ', accuracy_lst)

# end = time.time()
# print(f"Total time: {(end - start)/60:.2f} minutes")
