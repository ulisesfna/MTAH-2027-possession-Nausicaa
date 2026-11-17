"""TCN no causal de la v7 (C2).

Ve de la TCN del PR #9 (starter/train_tcn_v5.py, branca marc): projecció d'entrada 1x1, blocs
residuals de dues convolucions dilatades de nucli 3 i projecció de sortida 1x1. Canvis: padding
simètric (cada frame veu passat i futur), 8 blocs amb dilatacions 1..128, 64 canals i dropout de
0,1 dins de cada bloc. Camp receptiu: 1 + 2*2*(1+2+...+128) = 1021 frames (±510, uns ±20 s).

Entrada (B, 32, T) ja estandarditzada (35 a la v7b i la v7c); sortida (B, 3, T) amb logits de H, A i D.
Vegeu docs/tcn.md.
"""
import torch
import torch.nn as nn

CLASSES = ("H", "A", "D")
DILATACIONS = (1, 2, 4, 8, 16, 32, 64, 128)
CANALS = 64
DROPOUT = 0.1


class Bloc(nn.Module):
    def __init__(self, c, dil, p):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, 3, dilation=dil, padding=dil)
        self.conv2 = nn.Conv1d(c, c, 3, dilation=dil, padding=dil)
        self.drop = nn.Dropout(p)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.drop(self.relu(self.conv1(x)))
        h = self.drop(self.conv2(h))
        return self.relu(h + x)


class TCN(nn.Module):
    def __init__(self, n_in=32, canals=CANALS, dilatacions=DILATACIONS, dropout=DROPOUT, n_out=3):
        super().__init__()
        self.entrada = nn.Conv1d(n_in, canals, 1)
        self.blocs = nn.Sequential(*[Bloc(canals, d, dropout) for d in dilatacions])
        self.sortida = nn.Conv1d(canals, n_out, 1)

    def forward(self, x):
        return self.sortida(self.blocs(self.entrada(x)))


def camp_receptiu(dilatacions=DILATACIONS):
    return 1 + 4 * sum(dilatacions)


def carrega(path):
    """Model i estandardització d'un checkpoint desat per bench/train_tcn.py."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = TCN(n_in=len(ck["mitjana"]))
    m.load_state_dict(ck["estat"])
    m.eval()
    return m, ck


@torch.no_grad()
def probabilitats(model, ck, X):
    """X (n, k) sense estandarditzar -> (n, 3) float32, en CPU i sencer d'una passada. Si X té més
    columnes de les que el model coneix, es fan servir les primeres (les noves sempre van al final)."""
    import numpy as np
    X = X[:, :len(ck["mitjana"])]
    Z = (X.astype(np.float32) - ck["mitjana"]) / ck["desviacio"]
    x = torch.from_numpy(np.ascontiguousarray(Z.T))[None]
    p = torch.softmax(model(x)[0], dim=0).numpy().T
    return np.ascontiguousarray(p, dtype=np.float32)
