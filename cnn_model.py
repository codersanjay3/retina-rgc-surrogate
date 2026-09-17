"""Spatiotemporal CNN predicting spike counts for all 27 RGCs jointly from
10x10 stimulus history, in the spirit of Baccus/Ganguli-style deep retina
models. Trained with a Poisson loss (matches the spiking statistics used in
the LNP baseline) and validated against the same held-out PSTH."""

import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

from retina_lib import load_training_data, bin_spike_times, NLAGS, NX, NY

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)


def build_windows(stim, nlags):
    """stim: (T, 100) -> (T-nlags+1, nlags, 10, 10) float32, no copy until cast."""
    grid = stim.reshape(-1, NX, NY)
    windows = sliding_window_view(grid, nlags, axis=0)  # (T-nlags+1, 10, 10, nlags)
    windows = np.moveaxis(windows, -1, 1)  # (T-nlags+1, nlags, 10, 10)
    return windows.astype(np.float32)


class RetinaCNN(nn.Module):
    def __init__(self, nlags, ncells):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nlags, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * NX * NY, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, ncells),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    Stim, SpTimes = load_training_data()
    ncells = SpTimes.shape[0]
    slen = Stim.shape[0]

    print("Building spike-count target matrix...")
    Y = np.zeros((slen, ncells), dtype=np.float32)
    for c in range(ncells):
        Y[:, c] = bin_spike_times(SpTimes[c].flatten(), slen)

    print("Building sliding-window stimulus tensor...")
    X = build_windows(Stim, NLAGS)  # (slen-nlags+1, nlags, 10, 10)
    Y_aligned = Y[NLAGS - 1 :]  # align target to window's "current" frame
    assert X.shape[0] == Y_aligned.shape[0]
    print("X shape:", X.shape, "Y shape:", Y_aligned.shape)

    # time-contiguous split: last 15% of time as validation (no shuffling across the boundary)
    n = X.shape[0]
    n_val = int(n * 0.15)
    n_train = n - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    Y_train, Y_val = Y_aligned[:n_train], Y_aligned[n_train:]
    print(f"train: {n_train} samples, val: {n_val} samples")

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    Y_val_t = torch.from_numpy(Y_val).to(DEVICE)

    model = RetinaCNN(NLAGS, ncells).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.PoissonNLLLoss(log_input=False, full=False, eps=1e-6)

    batch_size = 256
    n_epochs = 20
    n_batches = n_train // batch_size

    train_losses, val_losses = [], []
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_train)  # shuffle order of windows, not their content, within train split only
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            xb = X_train_t[idx].to(DEVICE)
            yb = Y_train_t[idx].to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

    torch.save(model.state_dict(), "retina_cnn.pt")
    np.save("cnn_train_losses.npy", np.array(train_losses))
    np.save("cnn_val_losses.npy", np.array(val_losses))
    print("Saved retina_cnn.pt")
