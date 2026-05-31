from ray import tune
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split

def inv_minmax(x, X_min, X_max):
    return x * (X_max - X_min) + X_min

def get_accuracy(y_true, y_pred):
    err = y_pred - y_true

    # accuracy within 5%
    acc_within_5 = ((err.abs() / (y_true.abs())) <= 0.05).float().mean().item() * 100.0
    
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
    def __init__(self, n_epochs=2001, lr=1e-3, weight_decay=0, lambda_phys=0.02, hidden_dim=64,
                 phys_start_epoch=100):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.hidden_dim = hidden_dim
        self.phys_start_epoch = phys_start_epoch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)

        # Model
        self.module = PINN_module(input_dim=X.shape[-1], hidden_dim=self.hidden_dim, output_dim=Y.shape[-1]).to(self.device)
        self.loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lambda_phys = self.lambda_phys
        
        self.train_loss_lst, self.test_loss_lst, self.epochs_lst = [], [], []
        self.train_phys_loss_lst = []

        for self.epoch in range(self.n_epochs):
            self.module.train()
            
            sum_data = sum_phys = sum_total = sum_accuracy = 0.0
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
                dt = time_lst[-2] - time_lst[-1]

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

                ktl_true  = inv_minmax(y_batch[:, 0], self.Y0_min, self.Y0_max)
                kdil_true = inv_minmax(y_batch[:, 1], self.Y1_min, self.Y1_max)
                mRNA_true = inv_minmax(y_batch[:, 2], self.Y2_min, self.Y2_max)
                
                ktl_pred  = inv_minmax(y_pred[:, 0], self.Y0_min, self.Y0_max)
                kdil_pred = inv_minmax(y_pred[:, 1], self.Y1_min, self.Y1_max)
                mRNA_pred = inv_minmax(y_pred[:, 2], self.Y2_min, self.Y2_max)
                
                ktl_5 = get_accuracy(ktl_true,  ktl_pred)
                kdil_5 = get_accuracy(kdil_true, kdil_pred)
                mRNA_5 = get_accuracy(mRNA_true,  mRNA_pred)
        
                batch_accuracy = (ktl_5 + kdil_5 + mRNA_5) / 3
                sum_accuracy += batch_accuracy

            tune.report({"overall_accuracy": sum_accuracy / n_batches})
    
def main():

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.choice([0, 1e-5, 1e-3, 1e-2]),
        "lambda_phys": tune.uniform(0, 0.02),
        "hidden_dim": tune.choice([32, 64, 128]),
        "batch_size": tune.choice([32, 64]),
        "phys_start_epoch": tune.uniform(0, 2000)
    }

    X_lst = np.load("Simulations/sim_TU_data/yfp_culture.npy")
    Y_lst = np.load("Simulations/sim_TU_data/param_labels_culture.npy")

    def train_pinn(config):
        pinn = PINN(n_epochs=2001, lr=config["lr"], weight_decay=config["weight_decay"], 
                    lambda_phys=config["lambda_phys"], hidden_dim=config["hidden_dim"],
                    phys_start_epoch=config["phys_start_epoch"])
        pinn.fit(X_lst, Y_lst, batch_size=config["batch_size"])

    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch
    import os
    import ray

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    ray.init(runtime_env={"working_dir": "."})

    scheduler = ASHAScheduler(metric="overall_accuracy", mode="max", 
                              max_t=2001, grace_period=100, reduction_factor=4)
    optuna_search = OptunaSearch(metric="overall_accuracy", mode="max")

    tuner = tune.Tuner(train_pinn, param_space=config, 
                       tune_config=tune.TuneConfig(num_samples=50, 
                                                   scheduler=scheduler, 
                                                   search_alg=optuna_search))
    
    results = tuner.fit()
    print("Best config:", results.get_best_result(metric="overall_accuracy", mode="max").config)

    
if __name__ == "__main__":
    main()