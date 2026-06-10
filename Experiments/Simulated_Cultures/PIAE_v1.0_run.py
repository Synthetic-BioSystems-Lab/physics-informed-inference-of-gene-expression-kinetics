import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/home/zacha/1_Projects/DL')

import PINN.PIAE_v10 as NN
from pathlib import Path
import shutil

print(Path.cwd())

save_direct ="Experiments/Simulated_Cultures/plots/PIAE_v1.0_r1"

plots = Path(save_direct)
plots.mkdir(exist_ok=True)

for p in plots.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()

start = time.time()

timepoints = np.load("Simulations/sim_TU_data/time_culture.npy")
cell_conc = np.load("Simulations/sim_TU_data/cell_conc_culture.npy")
yfp = np.load("Simulations/sim_TU_data/yfp_culture.npy")
labels = np.load("Simulations/sim_TU_data/param_labels_culture_PIAE.npy")

print(yfp)

lambda_phys_lst = [0, 6e-5, 0.05]
accuracy_lst = []

for i in range(len(lambda_phys_lst)):

    torch.manual_seed(308380)

    model = NN.PIAE(save_direct, n_epochs=2001, p_epoch=250, lr=0.009, weight_decay=0, 
                     lambda_phys=lambda_phys_lst[i], hidden_dim=64, phys_start_epoch=0, 
                     batch_size=20)
    model.fit(cell_conc, yfp, timepoints)
    model.plot_loss()
    model.plot_accuracy()   
    acc = model.predict(labels)
        
    accuracy_lst.append(acc)
        
print('lambda physics: ', lambda_phys_lst)
print('Accuracy: ', accuracy_lst)

torch.save(model, 'Experiments/Simulated_Cultures/models/PIAE_v1.0_model.pth')

end = time.time()
print(f"Total time: {(end - start)/60:.2f} minutes")