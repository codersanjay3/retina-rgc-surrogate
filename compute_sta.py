"""Spike-triggered average (STA) for one RGC, ported directly from the
reference loadDataAndComputeSTA.m script that ships with the dataset."""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

DATA_DIR = "data/rgcData_Nature08"

stim_data = sio.loadmat(f"{DATA_DIR}/Stim_reduced.mat")
sp_data = sio.loadmat(f"{DATA_DIR}/SpTimesRGC.mat")

Stim = stim_data["Stim"]  # (144000, 100): [time bins, spatial pixels]
SpTimes = sp_data["SpTimes"][0]  # (27,) object array, one entry per cell

cellnum = 0  # cell 1 in MATLAB's 1-indexing
nlags = 20  # number of time bins of lags to consider
nx, ny = 10, 10
npx = nx * ny

slen = Stim.shape[0]  # number of time bins

# bin spike times into frames, matching MATLAB's hist(SpTimes{1}, 0.5:1:slen)
spike_times = SpTimes[cellnum].flatten()
bin_edges = np.arange(0, slen + 1) + 0.5  # bin centers at 1..slen -> edges at 0.5..slen+0.5
sps, _ = np.histogram(spike_times, bins=np.concatenate([[0.5], bin_edges]))
sps = sps[: slen + 1][1:]  # drop the stray first bin from the concat, keep slen bins
assert sps.shape[0] == slen, (sps.shape, slen)

print(f"Cell {cellnum + 1}: {int(sps.sum())} spikes binned "
      f"(raw spike-time array had {len(spike_times)} entries)")

# compute STA, direct port of the MATLAB loop
sta = np.zeros((nlags, npx))
for jj in range(nlags):
    row = nlags - jj - 1
    sta[row, :] = sps[jj:] @ Stim[: slen - jj, :]
sta = sta / sps.sum()

fig, axes = plt.subplots(2, 2, figsize=(9, 8))

ax = axes[0, 0]
im = ax.imshow(sta, aspect="auto", cmap="RdBu_r")
ax.set_xlabel("space (pixel index)")
ax.set_ylabel("time (lag bin, bottom = closest to spike)")
ax.set_title("full STA")
fig.colorbar(im, ax=ax)

ax = axes[1, 0]
ax.plot(sta[:, 35])
ax.set_xlabel("time (lag bin)")
ax.set_title("time slice (pixel 36)")

ax = axes[1, 1]
im2 = ax.imshow(sta[16, :].reshape(nx, ny), cmap="RdBu_r")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("space slice (lag row 17)")
fig.colorbar(im2, ax=ax)

axes[0, 1].axis("off")

fig.suptitle(f"Spike-triggered average, cell {cellnum + 1}")
fig.tight_layout()
fig.savefig("sta_cell1.png", dpi=150)
print("Saved sta_cell1.png")
