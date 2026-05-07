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
    def __init__(self, n_epochs=2001, lr=1e-3, weight_decay=0, lambda_phys=0.02, activation=nn.ReLU):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_phys = lambda_phys
        self.activation = activation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, Y, batch_size=32):
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        
        self.X_min, self.X_max = X.min(), X.max()
        X = (X - self.X_min) / (self.X_max - self.X_min)
        
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
        self.train_phys_loss_lst = []

        for self.epoch in range(self.n_epochs):
            self.module.train()
            
            sum_data = sum_phys = sum_total = sum_accuracy = 0
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

                ktl_true  = y_batch[:, 0]
                kdil_true = y_batch[:, 1]
                mRNA_true = y_batch[:, 2]
                
                ktl_5 = get_accuracy(ktl_true, ktl)
                kdil_5 = get_accuracy(kdil_true, kdil)
                mRNA_5 = get_accuracy(mRNA_true, mrna)
        
                batch_accuracy = (ktl_5 + kdil_5 + mRNA_5) / 3
                sum_accuracy += batch_accuracy

            tune.report({"overall_accuracy": sum_accuracy / n_batches})
    
def main():

    config = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.choice([0, 1e-5]),
        "lambda_phys": tune.uniform(0.0001, 0.02),
        "activation": tune.choice([nn.ReLU, nn.Softplus]),
        "batch_size": tune.choice([32, 64])
    }

    X_lst = np.load('sim_TU_data/yfp.npy')
    Y_lst = np.load('sim_TU_data/param_labels.npy')

    def train_piae(config):
        piae = PIAE(n_epochs=2001, lr=config["lr"], weight_decay=config["weight_decay"], 
                    lambda_phys=config["lambda_phys"], activation=config["activation"])
        piae.fit(X_lst, Y_lst, batch_size=config["batch_size"])

    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    scheduler = ASHAScheduler(metric="overall_accuracy", mode="max", 
                              max_t=2001, grace_period=100, reduction_factor=4)
    optuna_search = OptunaSearch(metric="overall_accuracy", mode="max")

    tuner = tune.Tuner(train_piae, param_space=config, 
                       tune_config=tune.TuneConfig(num_samples=50, 
                                                   scheduler=scheduler, 
                                                   search_alg=optuna_search,
                                                   max_concurrent_trials=2))
    
    results = tuner.fit()
    print("Best config:", results.get_best_result(metric="overall_accuracy", mode="max").config)

    
if __name__ == "__main__":
    main()