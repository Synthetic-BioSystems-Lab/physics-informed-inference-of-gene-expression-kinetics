import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math

def inv_minmax(x, X_min, X_max):
    return x * (X_max - X_min) + X_min


def plot_predictions(x_train, y_train, x_test, y_test, y_pred, title=""):
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
    plt.savefig(f"plots/{title.replace(' ', '_')}.pdf")
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

class PIAE_module(nn.Module):
   
    def __init__(self, input_dim=2, activation=nn.ReLU):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 80),
            activation(),
            nn.Linear(80, 40),
            activation(),
            nn.Linear(40, 20),
            activation(),
            nn.Linear(20, 10),
            activation()
            )
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            activation(),
            nn.Linear(20, 40),
            activation(),
            nn.Linear(40, 80),
            activation(),
            nn.Linear(80, input_dim)
            )

    def forward(self, x):

        latent = self.encoder(x)
        out = self.decoder(latent)

        return latent, out

class PIAE():
    def __init__(self, n_epochs=2001, p_epoch=1000, lr=1e-3, weight_decay=0, lambda_phys=0.02, activation=nn.ReLU):
        self.n_epochs = n_epochs
        self.p_epoch = p_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.activation = activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, Y, batch_size=32):
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        
        # self.X_min, self.X_max = X.min(), X.max()
        # X = (X - self.X_min) / (self.X_max - self.X_min)
        
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        self.dataset = TensorDataset(X, Y)

        # Split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(X) - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        # Model
        self.module = PIAE_module(input_dim=X.shape[-1], activation=self.activation).to(self.device)
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
                latent, x_pred = self.module(x_batch)

                loss_data = self.loss_fn(x_pred, x_batch)
                
                ktl = latent[:, 0]
                kdil = latent[:, 1]
                mrna = latent[:, 2]
                yfp_final = x_batch[:, -1]
                
                # dAdt = (ktl * M) - (kdil * A)
                eps = 1e-6
                res = ktl * mrna - kdil * yfp_final
                scale1 = (ktl * mrna).abs() + (kdil * yfp_final).abs() + eps
                loss_phys = (res.abs() / scale1).mean()
                
                # loss_phys = 0
                              
                loss = loss_data + lambda_phys * loss_phys

                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
                optimizer.step()
                
                sum_data += loss_data.item()
                sum_phys += loss_phys.item()
                sum_total += loss.item()
                n_batches += 1

            if self.epoch % self.p_epoch == 0 or self.epoch == self.n_epochs - 1:
                self.module.eval()
                with torch.inference_mode():
                    X_all, Y_all = self.dataset.tensors
                    train_idx = self.train_set.indices
                    test_idx = self.test_set.indices
                    
                    x_train = X_all[train_idx].to(self.device)
                    y_train = Y_all[train_idx].to(self.device)
                    x_test  = X_all[test_idx].to(self.device)
                    y_test  = Y_all[test_idx].to(self.device)

                    latent, test_pred = self.module(x_test)
                    test_loss = self.loss_fn(test_pred, x_test)
                    
                    train_data_epoch = sum_data / n_batches
                    train_phys_epoch = sum_phys / n_batches
                    
                    self.train_loss_lst.append(train_data_epoch)
                    self.test_loss_lst.append(test_loss.item())
                    self.train_phys_loss_lst.append(train_phys_epoch)
                    self.epochs_lst.append(self.epoch)

                print(f"{self.epoch:04d} | train {train_data_epoch:.4f} | test {test_loss.item():.4f} "
                      f"| phys {train_phys_epoch:.4f} ")
                
                err = latent[:,:3] - y_test

                # accuracy within 5%
                acc_within_5 = ((err.abs() / (y_test.abs())) <= 0.05).float().mean().item() * 100.0
                self.acc_lst.append(acc_within_5)

                plot_predictions(x_train[:, -1].cpu().numpy(),   
                                 y_train.cpu().numpy(),             
                                 x_test[:, -1].cpu().numpy(),    
                                 y_test.cpu().numpy(),              
                                 test_pred.cpu().numpy(),       
                                 title=f"PIAE Epoch {self.epoch} lambda physics {lambda_phys:.4f}")

    def plot_loss(self):
        
        plt.figure()
        
        plt.plot(self.epochs_lst, self.train_loss_lst, label='train loss')
        plt.plot(self.epochs_lst, self.test_loss_lst, label='test loss')
        plt.plot(self.epochs_lst, self.train_phys_loss_lst, label='physics loss')
        
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(f"plots/loss_{self.epoch}_{self.lambda_phys}.pdf")
        plt.close()

    def plot_accuracy(self):
        
        plt.figure()
        
        plt.plot(self.epochs_lst, self.acc_lst)

        plt.ylabel('Accuracy within 5% (%)')
        plt.xlabel('Epochs')
        plt.savefig(f"plots/accuracy_{self.epoch}_{self.lambda_phys}.pdf")
        plt.close()
    
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

            latent, test_pred = self.module(x_test)
            test_loss = self.loss_fn(test_pred, x_test)

        # true/pred for test split
    
        ktl_true  = y_test[:, 0]
        kdil_true = y_test[:, 1]
        mRNA_true = y_test[:, 2]
        
        ktl_pred  = latent[:, 0]
        kdil_pred = latent[:, 1]
        mRNA_pred = latent[:, 2]    
        
        ktl_5 = print_accuracy(ktl_true,  ktl_pred,  "ktl")
        kdil_5 = print_accuracy(kdil_true, kdil_pred, "kdil")
        mRNA_5 = print_accuracy(mRNA_true,  mRNA_pred,  "mRNA")
        
        overall_accuracy = (ktl_5 + kdil_5 + mRNA_5) / 3
        print(f'\nOverall Accuracy within 5%: {overall_accuracy:.2f}%')
        
        return overall_accuracy
    

def main():

    from pathlib import Path
    import shutil

    plots = Path("plots")
    plots.mkdir(exist_ok=True)

    for p in plots.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()

    import time

    start = time.time()
        
    X_lst = np.load('sim_TU_data/yfp.npy')
    Y_lst = np.load('sim_TU_data/param_labels.npy')
    
    lambda_phys_lst = [0, 0.01]
    accuracy_lst = []
    
    for i in range(len(lambda_phys_lst)):
        
        torch.manual_seed(308380)
    
        model = PIAE(n_epochs=2001, p_epoch=1000, lr=1e-1, weight_decay=0, 
                     lambda_phys=lambda_phys_lst[i])
        model.fit(X_lst, Y_lst, batch_size=32)
        model.plot_loss()
        model.plot_accuracy()   
        acc = model.predict()
        
        accuracy_lst.append(acc)
        
    print('lambda physics: ', lambda_phys_lst)
    print('Accuracy: ', accuracy_lst)

    end = time.time()
    print(f"Total time: {(end - start)/60:.2f} minutes")
    
if __name__ == "__main__":
    main()