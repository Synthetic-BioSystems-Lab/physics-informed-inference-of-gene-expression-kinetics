import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math

class PIE_module(nn.Module):

    def __init__(self, T, hidden=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(T, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
            nn.Softplus()
        )

    def forward(self, cell_curve):

        params = self.encoder(cell_curve)

        ktx = params[:, 0:1]
        kdeg_m = params[:, 1:2]
        ktl = params[:, 2:3]
        kdeg_p = params[:, 3:4]

        return ktx, kdeg_m, ktl, kdeg_p
    
def simulate(
    DNA,
    ktx,
    kdeg_m,
    ktl,
    kdeg_p,
    dt,
):

    batch, T = DNA.shape

    mRNA = torch.zeros_like(DNA)
    protein = torch.zeros_like(DNA)

    for t in range(T - 1):

        dm = (
            ktx * DNA[:, t:t+1]
            - kdeg_m * mRNA[:, t:t+1]
        )

        dp = (
            ktl * mRNA[:, t:t+1]
            - kdeg_p * protein[:, t:t+1]
        )

        mRNA[:, t + 1:t + 2] = (
            mRNA[:, t:t+1]
            + dt * dm
        )

        protein[:, t + 1:t + 2] = (
            protein[:, t:t+1]
            + dt * dp
        )

    return mRNA, protein