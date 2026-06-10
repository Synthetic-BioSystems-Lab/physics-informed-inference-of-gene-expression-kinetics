from ray import tune
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

def get_accuracy(y_true, y_pred):
    err = y_pred - y_true

    # accuracy within 5%
    acc_within_5 = ((err.abs() / (y_true.abs())) <= 0.05).float().mean().item() * 100.0
    
    return acc_within_5

def central_difference(y, timepoints):
    dt = timepoints[1] - timepoints[0]
    dy_dt = torch.zeros_like(y)

    # Central difference for interior points
    dy_dt[1:-1] = (y[2:] - y[:-2]) / (2 * dt)

    return dy_dt

class PIAE_module(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.copynum = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2)
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 2)
        )

        self.decoder_2 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, cell_conc):

        DNA = self.copynum*cell_conc
        ktx_kdeg_m = self.encoder_1(DNA)
        mRNA = self.decoder_1(ktx_kdeg_m)

        ktl_kdeg_p = self.encoder_2(mRNA)
        protein = self.decoder_2(ktl_kdeg_p)

        return DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein
        
class PIAE():

    def __init__(self, n_epochs=2001, p_epoch=100, lr=1e-3, weight_decay=0, lambda_phys=0.02,
                 hidden_dim=64, phys_start_epoch=100, batch_size=32):
        self.n_epochs = n_epochs
        self.p_epoch = p_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.hidden_dim = hidden_dim
        self.phys_start_epoch = phys_start_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

    def fit(self, cell_conc, FP, timepoints, labels):

        cell_conc = torch.tensor(cell_conc, dtype=torch.float32)
        FP = torch.tensor(FP, dtype=torch.float32)
        timepoints = torch.tensor(timepoints, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

        # Splitting data into train and test sets
        self.dataset = TensorDataset(cell_conc, FP, labels)

        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

        self.module = PIAE_module(input_dim=cell_conc.shape[-1], hidden_dim=self.hidden_dim, output_dim=FP.shape[-1]).to(self.device)
        self.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_phys = self.lambda_phys

        self.train_loss_lst, self.test_loss_lst, self.epochs_lst = [], [], []
        self.train_phys_loss_lst, self.acc_lst = [], []

        for self.epoch in range(self.n_epochs):
            self.module.train()

            sum_data = sum_phys = sum_total = sum_acc = 0.0
            n_batches = 0

            for cell_conc_batch, FP_batch, batch_labels in train_loader:
                cell_conc_batch, FP_batch = cell_conc_batch.to(self.device), FP_batch.to(self.device)

                DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein = self.module(cell_conc_batch)

                loss_data = self.loss_fn(protein, FP_batch)

                if self.epoch >= self.phys_start_epoch:
                    ktx = ktx_kdeg_m[:, 0]
                    kdeg_m = ktx_kdeg_m[:, 1]
                    ktl = ktl_kdeg_p[:, 0]
                    kdeg_p = ktl_kdeg_p[:, 1]

                    ktx = ktx.unsqueeze(-1)
                    kdeg_m = kdeg_m.unsqueeze(-1)
                    ktl = ktl.unsqueeze(-1)
                    kdeg_p = kdeg_p.unsqueeze(-1)

                    d_mRNA_dt = central_difference(mRNA, timepoints)
                    d_protein_dt = central_difference(protein, timepoints)

                    #print(d_mRNA_dt.shape, DNA.shape, ktx.shape, kdeg_m.shape, mRNA.shape, d_protein_dt.shape, ktl.shape, kdeg_p.shape)

                    phys_loss = torch.mean(d_mRNA_dt - (ktx*DNA - kdeg_m*mRNA)) + torch.mean(d_protein_dt - (ktl*mRNA - kdeg_p))
                else:
                    phys_loss = torch.tensor(0.0)

                loss = loss_data + lambda_phys * phys_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sum_data += loss_data.item()
                sum_phys += phys_loss.item()
                sum_total += loss.item()
                n_batches += 1
            
                self.train_loss_lst.append(sum_data / n_batches)
                self.train_phys_loss_lst.append(sum_phys / n_batches)
                self.epochs_lst.append(self.epoch)

                ktx_pred = ktx_kdeg_m[:, 0]
                kdeg_m_pred = ktx_kdeg_m[:, 1]
                ktl_pred = ktl_kdeg_p[:, 0]
                kdeg_p_pred = ktl_kdeg_p[:, 1]

                ktx_true = batch_labels[:, 0]
                kdeg_m_true = batch_labels[:, 1]
                ktl_true = batch_labels[:, 2]
                kdeg_p_true = batch_labels[:, 3]

                ktx_5 = get_accuracy(ktx_true,  ktx_pred)
                kdeg_m_5 = get_accuracy(kdeg_m_true, kdeg_m_pred)
                ktl_5 = get_accuracy(ktl_true,  ktl_pred)
                kdeg_p_5 = get_accuracy(kdeg_p_true, kdeg_p_pred)

                batch_accuracy = (ktx_5 + kdeg_m_5 + ktl_5 + kdeg_p_5) / 4
                sum_acc += batch_accuracy

            tune.report({"loss": loss.item()})



def main():

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        # "weight_decay": tune.choice([0, 1e-5, 1e-3, 1e-2]),
        "lambda_phys": tune.loguniform(0.00001, 10),
        "hidden_dim": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([25, 50]),
        "phys_start_epoch": tune.choice([0])
    }

    timepoints = np.load("Simulations/sim_TU_data/time_culture.npy")
    cell_conc = np.load("Simulations/sim_TU_data/cell_conc_culture.npy")
    yfp = np.load("Simulations/sim_TU_data/yfp_culture.npy")
    labels = np.load("Simulations/sim_TU_data/param_labels_culture_PIAE.npy")

    def train_pinn(config):
        piae = PIAE(n_epochs=501, lr=config["lr"], weight_decay=0, 
                    lambda_phys=config["lambda_phys"], hidden_dim=config["hidden_dim"],
                    phys_start_epoch=config["phys_start_epoch"], batch_size=config["batch_size"])
        piae.fit(cell_conc, yfp, timepoints, labels)

    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    import os
    import ray

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    ray.init(runtime_env={"working_dir": "."})

    scheduler = ASHAScheduler(metric="loss", mode="min", 
                              max_t=2001, grace_period=50, reduction_factor=4)
    optuna_search = OptunaSearch(metric="loss", mode="min")

    tuner = tune.Tuner(train_pinn, param_space=config, 
                       tune_config=tune.TuneConfig(num_samples=50, 
                                                   scheduler=scheduler, 
                                                   search_alg=optuna_search))
    
    results = tuner.fit()
    print("Best config:", results.get_best_result(metric="loss", mode="min").config)

    
if __name__ == "__main__":
    main()               