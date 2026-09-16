"""Compute the STA for all 27 cells and plot the peak-lag spatial RF for each,
as a sanity check across the population before building any predictive model."""

import numpy as np
import matplotlib.pyplot as plt

from retina_lib import load_training_data, compute_sta, NX, NY, NLAGS

Stim, SpTimes = load_training_data()
ncells = SpTimes.shape[0]

fig, axes = plt.subplots(4, 7, figsize=(16, 10))
axes = axes.flatten()

stas = []
for c in range(ncells):
    spike_times = SpTimes[c].flatten()
    sta, sps, nspikes = compute_sta(Stim, spike_times)
    stas.append(sta)

    # peak lag = the row with largest |value| anywhere in space
    peak_lag = np.argmax(np.abs(sta).max(axis=1))
    rf = sta[peak_lag, :].reshape(NX, NY)
    vmax = np.abs(rf).max()

    ax = axes[c]
    ax.imshow(rf, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(f"cell {c+1} (n={int(nspikes)})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

for ax in axes[ncells:]:
    ax.axis("off")

fig.suptitle("Spatial RF at each cell's peak lag (27 cells)")
fig.tight_layout()
fig.savefig("sta_all_cells.png", dpi=150)
print("Saved sta_all_cells.png")

np.save("stas_all_cells.npy", np.array(stas))
print(f"Saved stas_all_cells.npy, shape {np.array(stas).shape}")
