import numpy as np
import torch
from torch import nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math

mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

def inv_minmax(x, X_min, X_max):
    return x * (X_max - X_min) + X_min

def plot_predictions(save_direct, x_train, y_train, x_test, y_test, y_pred, title=""):
    x_train, x_test = np.asarray(x_train), np.asarray(x_test)
    y_train, y_test, y_pred = np.asarray(y_train), np.asarray(y_test), np.asarray(y_pred)   

    k_lst = ['ktl', 'kdil', 'Final [mRNA]']

    n_out = y_train.shape[1]
    
    cols = math.ceil(math.sqrt(n_out))
    rows = math.ceil(n_out/cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    
    for k in range(n_out):
        ax = axes[k]
        ax.scatter(x_train, y_train[:, k], c="b", s=10, alpha=0.6, label="Train")
        ax.scatter(x_test, y_test[:, k],  c="g", s=10, alpha=0.6, label="Test")
        ax.scatter(x_test, y_pred[:, k],  c="r", s=10, alpha=0.6, label="Pred")
        ax.set_xlabel("Final [FP]")
        ax.set_ylabel(f"{k_lst[k]}")
        ax.grid(alpha=0.2)

    axes[0].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{save_direct}/{title.replace(' ', '_')}_v2.svg")
    plt.close()

    for k in range(n_out):
        plt.figure()

        plt.scatter(x_train, y_train[:, k], c="b", s=10, alpha=0.6, label="Train")
        plt.scatter(x_test, y_test[:, k],  c="g", s=10, alpha=0.6, label="Test")
        plt.scatter(x_test, y_pred[:, k],  c="r", s=10, alpha=0.6, label="Pred")
        plt.xlabel("Final [FP]")
        plt.ylabel(f"{k_lst[k]}")

        plt.tight_layout()
        plt.savefig(f"{save_direct}/{title.replace(' ', '_')}_{k_lst[k]}_v2.svg")
        plt.close()

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

class PINN_module(nn.Module):
   
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc3 = nn.Linear(int(hidden_dim/2), output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):

        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)

        return out

class PINN():
    def __init__(self, save_direct, n_epochs=2001, p_epoch=100, lr=1e-3, weight_decay=0, lambda_phys=0.02, hidden_dim=64,
                 phys_start_epoch=100):
        self.save_direct = save_direct
        self.n_epochs = n_epochs
        self.p_epoch = p_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.hidden_dim = hidden_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phys_start_epoch = phys_start_epoch
        
    def fit(self, X, Y, batch_size=32):
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        
        self.X_min, self.X_max = X.min(), X.max()
        X = (X - self.X_min) / (self.X_max - self.X_min)
        
        self.Y0_min, self.Y0_max = Y[:, 0].min(), Y[:, 0].max()
        Y[:, 0] = (Y[:, 0] - self.Y0_min) / (self.Y0_max - self.Y0_min)
        
        self.Y1_min, self.Y1_max = Y[:, 1].min(), Y[:, 1].max()
        Y[:, 1] = (Y[:, 1] - self.Y1_min) / (self.Y1_max - self.Y1_min)
        
        self.Y2_min, self.Y2_max = Y[:, 2].min(), Y[:, 2].max()
        Y[:, 2] = (Y[:, 2] - self.Y2_min) / (self.Y2_max - self.Y2_min)
        
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        self.dataset = TensorDataset(X, Y)

        # Split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(Y) - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)

        # Model
        self.module = PINN_module(input_dim=X.shape[-1], hidden_dim=self.hidden_dim, output_dim=Y.shape[-1]).to(self.device)
        self.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_phys = self.lambda_phys
        
        self.train_loss_lst, self.test_loss_lst, self.epochs_lst = [], [], []
        self.train_phys_loss_lst, self.acc_lst = [], []

        for self.epoch in range(self.n_epochs):
            self.module.train()
            
            sum_data = sum_phys = sum_total = 0.0
            n_batches = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_pred = self.module(x_batch)

                loss_data = self.loss_fn(y_pred, y_batch)
                
                ktl = inv_minmax(y_pred[:, 0], self.Y0_min, self.Y0_max)
                kdil = inv_minmax(y_pred[:, 1], self.Y1_min, self.Y1_max)
                mrna = inv_minmax(y_pred[:, 2], self.Y2_min, self.Y2_max)
                
                yfp_final = inv_minmax(x_batch[:, -1], self.X_min, self.X_max)
                yfp_penult = inv_minmax(x_batch[:, -2], self.X_min, self.X_max)

                time_lst =np.load("Simulations/sim_TU_data/time_culture.npy")
                dt = time_lst[-1] - time_lst[-2]

                # 0 = (ktl * M) - (kdil * A) - dAdt
                eps = 1e-6
                res = ktl * mrna - kdil * yfp_final - ((yfp_final - yfp_penult) / dt)
                scale1 = (ktl * mrna).abs() + (kdil * yfp_final).abs() + ((yfp_final - yfp_penult) / dt).abs() + eps
                loss_phys = (res.abs() / scale1).mean()
                

                if self.epoch <= self.phys_start_epoch:
                    loss = loss_data
                else:
                    loss = loss_data + lambda_phys * loss_phys

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
                optimizer.step()
                
                sum_data += loss_data.item()
                sum_phys += loss_phys.item()
                sum_total += loss.item()
                n_batches += 1

            with torch.inference_mode():
                X_all, Y_all = self.dataset.tensors
                train_idx = self.train_set.indices
                test_idx = self.test_set.indices
                
                x_train = X_all[train_idx].to(self.device)
                y_train = Y_all[train_idx].to(self.device)
                x_test  = X_all[test_idx].to(self.device)
                y_test  = Y_all[test_idx].to(self.device)

                test_pred = self.module(x_test)
                test_loss = self.loss_fn(test_pred, y_test)
                
                train_data_epoch = sum_data / n_batches
                train_phys_epoch = sum_phys / n_batches
                
                self.train_loss_lst.append(train_data_epoch)
                self.test_loss_lst.append(test_loss.item())
                self.train_phys_loss_lst.append(train_phys_epoch)
                self.epochs_lst.append(self.epoch)

            x_train_plot = inv_minmax(x_train, self.X_min, self.X_max)
            x_test_plot = inv_minmax(x_test, self.X_min, self.X_max)
            
            y_train_plot = y_train.clone()
            y_train_plot[:, 0] = inv_minmax(y_train_plot[:, 0], self.Y0_min, self.Y0_max)
            y_train_plot[:, 1] = inv_minmax(y_train_plot[:, 1], self.Y1_min, self.Y1_max)
            y_train_plot[:, 2] = inv_minmax(y_train_plot[:, 2], self.Y2_min, self.Y2_max)
            
            y_test_plot = y_test.clone()
            y_test_plot[:, 0] = inv_minmax(y_test_plot[:, 0], self.Y0_min, self.Y0_max)
            y_test_plot[:, 1] = inv_minmax(y_test_plot[:, 1], self.Y1_min, self.Y1_max)
            y_test_plot[:, 2] = inv_minmax(y_test_plot[:, 2], self.Y2_min, self.Y2_max)
            
            test_pred_plot = test_pred.clone()
            test_pred_plot[:, 0] = inv_minmax(test_pred_plot[:, 0], self.Y0_min, self.Y0_max)
            test_pred_plot[:, 1] = inv_minmax(test_pred_plot[:, 1], self.Y1_min, self.Y1_max)
            test_pred_plot[:, 2] = inv_minmax(test_pred_plot[:, 2], self.Y2_min, self.Y2_max)
            
            err = test_pred_plot - y_test_plot
            
            # accuracy within 5%
            acc_within_5 = ((err.abs() / (y_test_plot.abs())) <= 0.05).float().mean().item() * 100.0
            self.acc_lst.append(acc_within_5)

            if self.epoch % self.p_epoch == 0 or self.epoch == self.n_epochs - 1:
                self.module.eval()

                print(f"{self.epoch:04d} | train {train_data_epoch:.4f} | test {test_loss.item():.4f} "
                      f"| phys {train_phys_epoch:.4f} ")
                
                plot_predictions(self.save_direct, x_train_plot[:, -1].cpu().numpy(),   
                                 y_train_plot.cpu().numpy(),             
                                 x_test_plot[:, -1].cpu().numpy(),    
                                 y_test_plot.cpu().numpy(),              
                                 test_pred_plot.cpu().numpy(),       
                                 title=f"PINN Epoch {self.epoch} lambda physics {lambda_phys:.4f}")

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

        return self.epochs_lst, self.acc_lst
    
    def predict(self):
        
        self.module.eval()
        with torch.inference_mode():
            X_all, Y_all = self.dataset.tensors
            train_idx = self.train_set.indices
            test_idx = self.test_set.indices
            
            x_train = X_all[train_idx].to(self.device)
            y_train = Y_all[train_idx].to(self.device)
            x_test  = X_all[test_idx].to(self.device)
            y_test  = Y_all[test_idx].to(self.device)

            test_pred = self.module(x_test)
            test_loss = self.loss_fn(test_pred, y_test)

        # true/pred for test split
    
        ktl_true  = inv_minmax(y_test[:, 0], self.Y0_min, self.Y0_max)
        kdil_true = inv_minmax(y_test[:, 1], self.Y1_min, self.Y1_max)
        mRNA_true = inv_minmax(y_test[:, 2], self.Y2_min, self.Y2_max)
        DNA_true = torch.ones_like(test_pred[:, 0])  #y_test[:, 3]
        
        ktl_pred  = inv_minmax(test_pred[:, 0], self.Y0_min, self.Y0_max)
        kdil_pred = inv_minmax(test_pred[:, 1], self.Y1_min, self.Y1_max)
        mRNA_pred = inv_minmax(test_pred[:, 2], self.Y2_min, self.Y2_max)
        DNA_pred = torch.ones_like(test_pred[:, 0])  #test_pred[:, 3]
        
        ktl_5 = print_accuracy(ktl_true,  ktl_pred,  "ktl")
        kdil_5 = print_accuracy(kdil_true, kdil_pred, "kdil")
        mRNA_5 = print_accuracy(mRNA_true,  mRNA_pred,  "mRNA")
        
        overall_accuracy = (ktl_5 + kdil_5 + mRNA_5) / 3
        print(f'\nOverall Accuracy within 5%: {overall_accuracy:.2f}%')
        
        return overall_accuracy
    
    def run_NN(self, X_input):

            self.X_min, self.X_max = X_input.min(), X_input.max()
            X_input = (X_input - self.X_min) / (self.X_max - self.X_min)
            X_input = torch.tensor(X_input, dtype=torch.float32)
            X_input = X_input.to(self.device)

            with torch.inference_mode():
                y_pred = self.module(X_input)
                ktl = inv_minmax(y_pred[:, 0], self.Y0_min, self.Y0_max)
                kdil = inv_minmax(y_pred[:, 1], self.Y1_min, self.Y1_max)
                mrna = inv_minmax(y_pred[:, 2], self.Y2_min, self.Y2_max)
        

            return ktl, kdil, mrna