"""Evaluate the trained CNN against the real held-out PSTH, using the exact
same protocol as lnp_all_cells.py, so the comparison is apples-to-apples."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from retina_lib import load_repeat_data, bin_spike_times, NLAGS
from cnn_model import RetinaCNN, build_windows, DEVICE

StimRpt, MtspRpt = load_repeat_data()
ncells = MtspRpt.shape[0]

model = RetinaCNN(NLAGS, ncells).to(DEVICE)
model.load_state_dict(torch.load("retina_cnn.pt", map_location=DEVICE))
model.eval()


def empirical_psth(mtsp_cell, nbins):
    ntrials = mtsp_cell.shape[1]
    total = np.zeros(nbins)
    for trial in range(ntrials):
        times = mtsp_cell[:, trial]
        times = times[times > 0]
        total += bin_spike_times(times, nbins)
    return total / ntrials


X_rpt = build_windows(StimRpt, NLAGS)  # (1181, 20, 10, 10)
with torch.no_grad():
    pred = model(torch.from_numpy(X_rpt).to(DEVICE)).cpu().numpy()  # (1181, 27)

cnn_correlations = []
for c in range(ncells):
    psth = empirical_psth(MtspRpt[c], StimRpt.shape[0])
    psth_aligned = psth[NLAGS - 1 :][: pred.shape[0]]
    r = np.corrcoef(pred[:, c], psth_aligned)[0, 1]
    cnn_correlations.append(r)

cnn_correlations = np.array(cnn_correlations)
lnp_correlations = np.load("lnp_correlations.npy")

print("cell   LNP r   CNN r   delta")
for c in range(ncells):
    print(f"{c+1:3d}   {lnp_correlations[c]:.3f}   {cnn_correlations[c]:.3f}   {cnn_correlations[c]-lnp_correlations[c]:+.3f}")

print(f"\nLNP: mean={lnp_correlations.mean():.3f} median={np.median(lnp_correlations):.3f}")
print(f"CNN: mean={cnn_correlations.mean():.3f} median={np.median(cnn_correlations):.3f}")
print(f"cells where CNN beat LNP: {(cnn_correlations > lnp_correlations).sum()} / {ncells}")

np.save("cnn_correlations.npy", cnn_correlations)

fig, ax = plt.subplots(figsize=(9, 4))
xw = np.arange(1, ncells + 1)
width = 0.35
ax.bar(xw - width / 2, lnp_correlations, width, label=f"LNP (mean={lnp_correlations.mean():.3f})")
ax.bar(xw + width / 2, cnn_correlations, width, label=f"CNN (mean={cnn_correlations.mean():.3f})")
ax.set_xlabel("cell")
ax.set_ylabel("predicted vs real PSTH correlation (r)")
ax.set_title("LNP baseline vs CNN: PSTH prediction quality")
ax.legend()
fig.tight_layout()
fig.savefig("lnp_vs_cnn_correlations.png", dpi=150)
print("Saved lnp_vs_cnn_correlations.png")
