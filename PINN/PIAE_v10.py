import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math

def print_accuracy(y_true, y_pred, name):
    err = y_pred - y_true

    # accuracy within 5%
    acc_within_5 = ((err.abs() / (y_true.abs())) <= 0.05).float().mean().item() * 100.0

    print(f"\n{name}: within 5% accuracy ={acc_within_5:.2f}%")
    
    # "accuracy within 10%" (custom, intuitive)
    acc_within_10 = ((err.abs() / (y_true.abs())) <= 0.10).float().mean().item() * 100.0

    print(f"{name}: within 10% accuracy ={acc_within_10:.2f}%")
    
    # "accuracy within 25%" (custom, intuitive)
    acc_within_25 = ((err.abs() / (y_true.abs())) <= 0.25).float().mean().item() * 100.0

    print(f"{name}: within 25% accuracy ={acc_within_25:.2f}%")
    
    return acc_within_5

def central_difference(y, timepoints):
    dt = timepoints[1] - timepoints[0]
    dy_dt = torch.zeros_like(y)

    # Central difference for interior points
    dy_dt[1:-1] = (y[2:] - y[:-2]) / (2 * dt)

    dy_dt[0]  = (y[1]  - y[0])   / dt
    dy_dt[-1] = (y[-1] - y[-2])  / dt

    return dy_dt

def inv_minmax(x, X_min, X_max):
    return x * (X_max - X_min) + X_min

class PIAE_module(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()
        self.copynum = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.encoder_1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus()
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.encoder_2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softplus()
        )

        self.decoder_2 = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, cell_conc):

        DNA = self.copynum*cell_conc
        print(DNA.shape)
        ktx_kdeg_m = (self.encoder_1(DNA))
        mRNA = self.decoder_1(ktx_kdeg_m)

        ktl_kdeg_p = self.encoder_2(mRNA)
        protein = self.decoder_2(ktl_kdeg_p)

        return DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein
        
class PIAE():

    def __init__(self, save_direct, n_epochs=2001, p_epoch=100, lr=1e-3, weight_decay=0, lambda_phys=0.02,
                 hidden_dim=64, phys_start_epoch=100, batch_size=32):
        self.n_epochs = n_epochs
        self.p_epoch = p_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.hidden_dim = hidden_dim
        self.phys_start_epoch = phys_start_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_direct = save_direct
        self.batch_size = batch_size

    def fit(self, cell_conc, FP, timepoints):

        cell_conc = torch.tensor(cell_conc, dtype=torch.float32)
        FP = torch.tensor(FP, dtype=torch.float32)
        timepoints = torch.tensor(timepoints, dtype=torch.float32, requires_grad=True)

        if cell_conc.ndim == 2:
            cell_conc = cell_conc.unsqueeze(-1)
        if FP.ndim == 2:
            FP = FP.unsqueeze(-1)

        self.cell_conc_min, self.cell_conc_max = cell_conc.min(), cell_conc.max()
        cell_conc = (cell_conc - self.cell_conc_min) / (self.cell_conc_max - self.cell_conc_min)

        self.FP_min, self.FP_max = FP.min(), FP.max()
        FP = (FP - self.FP_min) / (self.FP_max - self.FP_min)

        # Splitting data into train and test sets
        self.dataset = TensorDataset(cell_conc, FP)

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

            sum_data = sum_phys = sum_total = 0.0
            n_batches = 0

            for cell_conc_batch, FP_batch in train_loader:
                cell_conc_batch, FP_batch = cell_conc_batch.to(self.device), FP_batch.to(self.device)

                DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein = self.module(cell_conc_batch)
                print(DNA.shape)
                loss_data = self.loss_fn(protein, FP_batch)

                if self.epoch >= self.phys_start_epoch:
                    #print(ktx_kdeg_m.shape, ktl_kdeg_p.shape)
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

                    print(d_mRNA_dt.shape, DNA.shape, ktx.shape, kdeg_m.shape, mRNA.shape, d_protein_dt.shape, ktl.shape, kdeg_p.shape)

                    phys_loss = torch.mean(d_mRNA_dt - (ktx*DNA - kdeg_m*mRNA)) + torch.mean(d_protein_dt - (ktl*mRNA - kdeg_p*protein))
                else:
                    phys_loss = torch.tensor(0.0)

                loss = loss_data + lambda_phys * phys_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
                optimizer.step()

                sum_data += loss_data.item()
                sum_phys += phys_loss.item()
                sum_total += loss.item()
                n_batches += 1
            
            self.train_loss_lst.append(sum_data / n_batches)
            self.train_phys_loss_lst.append(sum_phys / n_batches)
            self.epochs_lst.append(self.epoch)

            with torch.inference_mode():
                sum_test_data, n_test_batches = 0.0, 0
                for cell_conc_batch, FP_batch in test_loader:
                    cell_conc_batch, FP_batch = cell_conc_batch.to(self.device), FP_batch.to(self.device)

                    DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein = self.module(cell_conc_batch)
                    loss_data = self.loss_fn(protein, FP_batch)

                    sum_test_data += loss_data.item()
                    n_test_batches += 1

                self.test_loss_lst.append(sum_test_data / n_test_batches)

            if self.epoch % self.p_epoch == 0:
                print(f"Epoch {self.epoch}: Train Loss = {self.train_loss_lst[-1]:.4f} | Phys Loss = {self.train_phys_loss_lst[-1]:.4f} "
                      f"| Test Loss = {self.test_loss_lst[-1]:.4f}")

    def predict(self, labels):
        self.module.eval()
        with torch.inference_mode():
            cell_conc_batch, FP_batch = self.test_set[:][0].to(self.device), self.test_set[:][1].to(self.device)
            DNA, ktx_kdeg_m, mRNA, ktl_kdeg_p, protein = self.module(cell_conc_batch)

            ktx_pred = ktx_kdeg_m[:, 0]
            kdeg_m_pred = ktx_kdeg_m[:, 1]
            ktl_pred = ktl_kdeg_p[:, 0]
            kdeg_p_pred = ktl_kdeg_p[:, 1]

            labels = torch.tensor(labels, dtype=torch.float32).to(self.device)

            ktx_true = labels[:, 0]
            kdeg_m_true = labels[:, 1]
            ktl_true = labels[:, 2]
            kdeg_p_true = labels[:, 3]

            ktx_5 = print_accuracy(ktx_true,  ktx_pred,  "ktx")
            kdeg_m_5 = print_accuracy(kdeg_m_true, kdeg_m_pred, "kdeg_m")
            ktl_5 = print_accuracy(ktl_true,  ktl_pred,  "ktl")
            kdeg_p_5 = print_accuracy(kdeg_p_true, kdeg_p_pred, "kdeg_p")

            overall_accuracy = (ktx_5 + kdeg_m_5 + ktl_5 + kdeg_p_5) / 4
            print(f'\nOverall Accuracy within 5%: {overall_accuracy:.2f}%')

        return overall_accuracy

    def plot_loss(self):
            
            plt.figure()
            
            plt.plot(self.epochs_lst, self.train_loss_lst, label='train loss')
            plt.plot(self.epochs_lst, self.test_loss_lst, label='test loss')
            plt.plot(self.epochs_lst, self.train_phys_loss_lst, label='physics loss')
            
            plt.xlabel('Epochs')
            plt.legend()
            plt.savefig(f"{self.save_direct}/PINN_v2_loss_{self.epoch}_{self.lambda_phys}.png")
            plt.close()

    def plot_accuracy(self):
        
        plt.figure()
        
        plt.plot(self.epochs_lst, self.acc_lst)

        plt.ylabel('Accuracy within 5% (%)')
        plt.xlabel('Epochs')
        plt.savefig(f"{self.save_direct}/PINN_v2_accuracy_{self.epoch}_{self.lambda_phys}.svg")
        plt.close()
                