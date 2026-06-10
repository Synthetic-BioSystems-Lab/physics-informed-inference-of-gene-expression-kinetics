import time
import numpy as np
import torch

import sys
sys.path.insert(0, '/home/zacha/1_Projects/DL')

import PINN.PINN_v2 as NN
from pathlib import Path
import shutil

print(Path.cwd())

save_direct ="Experiments/Simulated_Cultures/plots/5-30-2026"

plots = Path(save_direct)
plots.mkdir(exist_ok=True)

for p in plots.iterdir():
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()

start = time.time()

X_lst = np.load("Simulations/sim_TU_data/yfp_culture.npy")
Y_lst = np.load("Simulations/sim_TU_data/param_labels_culture.npy")

lambda_phys_lst = [0, 0.05]
accuracy_lst = []

for i in range(len(lambda_phys_lst)):

    torch.manual_seed(308380)

    model = NN.PINN(save_direct, n_epochs=2001, p_epoch=250, lr=0.009, weight_decay=0, 
                     lambda_phys=lambda_phys_lst[i], hidden_dim=64, phys_start_epoch=0)
    model.fit(X_lst, Y_lst, batch_size=20)
    model.plot_loss()
    model.plot_accuracy()   
    acc = model.predict()
        
    accuracy_lst.append(acc)
        
print('lambda physics: ', lambda_phys_lst)
print('Accuracy: ', accuracy_lst)

torch.save(model, 'Experiments/Simulated_Cultures/models/PINN_v2_model.pth')

end = time.time()
print(f"Total time: {(end - start)/60:.2f} minutes")
