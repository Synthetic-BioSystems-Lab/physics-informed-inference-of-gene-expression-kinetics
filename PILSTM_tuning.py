from ray import tune
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

class LSTM_module(nn.Module):
   
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Softplus()

    def forward(self, x):

        out, (hn, cn) = self.lstm(x)
        out = self.activation(out)
        out = self.fc(out[:, -1, :])

        return out

class LSTM():
    def __init__(self, n_epochs=2001, p_epoch=100, lr=1e-3, weight_decay=0, lambda_phys=0.02):
        self.n_epochs = n_epochs
        self.p_epoch = p_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys

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
        
        X = X.unsqueeze(-1)
        
        self.dataset = TensorDataset(X, Y)

        # Split
        train_size = int(0.8 * len(self.dataset))
        test_size = len(Y) - train_size
        self.train_set, self.test_set = random_split(self.dataset, [train_size, test_size])

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        # Model
        self.module = LSTM_module(input_dim=X.shape[-1], hidden_dim=64, output_dim=Y.shape[-1]).to(device)
        self.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_phys = self.lambda_phys
        
        self.train_loss_lst, self.test_loss_lst, self.epochs_lst = [], [], []
        self.train_phys_loss_lst = []

        for self.epoch in range(self.n_epochs):
            self.module.train()
            
            sum_data = sum_phys = sum_total = 0.0
            n_batches = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self.module(x_batch)

                loss_data = self.loss_fn(y_pred, y_batch)
                
                ktl = inv_minmax(y_pred[:, 0], self.Y0_min, self.Y0_max)
                kdil = inv_minmax(y_pred[:, 1], self.Y1_min, self.Y1_max)
                mrna = inv_minmax(y_pred[:, 2], self.Y2_min, self.Y2_max)
                
                yfp_final = inv_minmax(x_batch[:, -1, 0], self.X_min, self.X_max)
                
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
                    
                    x_train = X_all[train_idx].to(device)
                    y_train = Y_all[train_idx].to(device)
                    x_test  = X_all[test_idx].to(device)
                    y_test  = Y_all[test_idx].to(device)

                    test_pred = self.module(x_test)
                    test_loss = self.loss_fn(test_pred, y_test)
                    
                    train_data_epoch = sum_data / n_batches
                    train_phys_epoch = sum_phys / n_batches
                    
                    self.train_loss_lst.append(train_data_epoch)
                    self.test_loss_lst.append(test_loss.item())
                    self.train_phys_loss_lst.append(train_phys_epoch)
                    self.epochs_lst.append(self.epoch)

                print(f"{self.epoch:04d} | train {train_data_epoch:.4f} | test {test_loss.item():.4f} "
                      f"| phys {train_phys_epoch:.4f} ")
                
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
                
                plot_predictions(x_train_plot[:, -1, :].cpu().numpy(),   # [N, F] not [N]
                                 y_train_plot.cpu().numpy(),             # [N, 2]
                                 x_test_plot[:, -1, :].cpu().numpy(),    # [N, F]
                                 y_test_plot.cpu().numpy(),              # [N, 2]
                                 test_pred_plot.cpu().numpy(),           # [N, 2]
                                 title=f"PILSTM Epoch {self.epoch}")

    def plot_loss(self):
        
        plt.figure()
        
        plt.plot(self.epochs_lst, self.train_loss_lst, label='train loss')
        plt.plot(self.epochs_lst, self.test_loss_lst, label='test loss')
        plt.plot(self.epochs_lst, self.train_phys_loss_lst, label='physics loss')
        
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(f"plots/loss_{self.epoch}.pdf")
        plt.close()
    
    def predict(self):
        
        self.module.eval()
        with torch.inference_mode():
            X_all, Y_all = self.dataset.tensors
            train_idx = self.train_set.indices
            test_idx = self.test_set.indices
            
            x_train = X_all[train_idx].to(device)
            y_train = Y_all[train_idx].to(device)
            x_test  = X_all[test_idx].to(device)
            y_test  = Y_all[test_idx].to(device)

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
        
    X_lst = np.load('sim_TU_data/yfp.npy')
    Y_lst = np.load('sim_TU_data/param_labels.npy')
    
    lambda_phys_lst = [0.001]#[0, 0.001]
    accuracy_lst = []
    
    for i in range(len(lambda_phys_lst)):
        
        torch.manual_seed(308380)
    
        model = LSTM(n_epochs=8001, p_epoch=1000, lr=1e-3, weight_decay=0, 
                     lambda_phys=lambda_phys_lst[i])
        model.fit(X_lst, Y_lst)
        model.plot_loss()
        acc = model.predict()
        
        accuracy_lst.append(acc)
        
    print('lambda physics: ', lambda_phys_lst)
    print('Accuracy: ', accuracy_lst)
    
if __name__ == "__main__":
    main()
