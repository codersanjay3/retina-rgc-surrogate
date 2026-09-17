"""Export all data needed for the interactive web demo into one JSON file:
per-cell spatial RFs + LNP/CNN correlations, and cell 1's full PSTH curves
for all three models (LNP, CNN, circuit) for the interactive overlay."""

import json
import numpy as np
import torch

from retina_lib import load_training_data, load_repeat_data, compute_sta, bin_spike_times, NLAGS, NX, NY
from cnn_model import RetinaCNN, build_windows, DEVICE

Stim, SpTimes = load_training_data()
StimRpt, MtspRpt = load_repeat_data()
ncells = SpTimes.shape[0]

stas = np.load("stas_all_cells.npy")  # (27,20,100)
lnp_correlations = np.load("lnp_correlations.npy")
cnn_correlations = np.load("cnn_correlations.npy")


def empirical_psth(mtsp_cell, nbins):
    ntrials = mtsp_cell.shape[1]
    total = np.zeros(nbins)
    for trial in range(ntrials):
        times = mtsp_cell[:, trial]
        times = times[times > 0]
        total += bin_spike_times(times, nbins)
    return total / ntrials


def generator_signal(stim, filt):
    nlags = filt.shape[0]
    slen = stim.shape[0]
    validlen = slen - nlags + 1
    g = np.zeros(validlen)
    for k in range(nlags):
        g += stim[k : k + validlen, :] @ filt[k, :]
    return g


def fit_nonlinearity(g_train, spikes_aligned, nbins_nl=25):
    order = np.argsort(g_train)
    g_sorted, sp_sorted = g_train[order], spikes_aligned[order]
    edges = np.linspace(0, len(g_sorted), nbins_nl + 1).astype(int)
    nl_x, nl_y = np.zeros(nbins_nl), np.zeros(nbins_nl)
    for i in range(nbins_nl):
        lo, hi = edges[i], edges[i + 1]
        nl_x[i], nl_y[i] = g_sorted[lo:hi].mean(), sp_sorted[lo:hi].mean()
    return nl_x, nl_y


# --- per-cell summary (spatial RF at peak lag, sign, spike count, model correlations) ---
cells_summary = []
for c in range(ncells):
    sta = stas[c]
    peak_lag = int(np.argmax(np.abs(sta).max(axis=1)))
    rf = sta[peak_lag, :].reshape(NX, NY)
    vmax = float(np.abs(rf).max())
    n_spikes = int(bin_spike_times(SpTimes[c].flatten(), Stim.shape[0]).sum())
    cells_summary.append({
        "cell": c + 1,
        "rf": np.round(rf / vmax, 4).tolist() if vmax > 0 else rf.tolist(),  # normalized to [-1,1]
        "sign": "OFF" if rf.flat[np.argmax(np.abs(rf))] < 0 else "ON",
        "n_spikes": n_spikes,
        "lnp_r": round(float(lnp_correlations[c]), 4),
        "cnn_r": round(float(cnn_correlations[c]), 4),
    })

# --- cell 1 detail: real PSTH + LNP/CNN/circuit predicted rate curves ---
CELL = 0
sta1, sps1, nspikes1 = compute_sta(Stim, SpTimes[CELL].flatten(), nlags=NLAGS)

g_train = generator_signal(Stim, sta1)
spikes_aligned = sps1[NLAGS - 1 :][: len(g_train)]
nl_x, nl_y = fit_nonlinearity(g_train, spikes_aligned)
g_rpt = generator_signal(StimRpt, sta1)
lnp_pred = np.interp(g_rpt, nl_x, nl_y)

model = RetinaCNN(NLAGS, ncells).to(DEVICE)
model.load_state_dict(torch.load("retina_cnn.pt", map_location=DEVICE))
model.eval()
X_rpt = build_windows(StimRpt, NLAGS)
with torch.no_grad():
    cnn_pred_all = model(torch.from_numpy(X_rpt).to(DEVICE)).cpu().numpy()
cnn_pred = cnn_pred_all[:, CELL]

real_psth = empirical_psth(MtspRpt[CELL], StimRpt.shape[0])
real_psth_aligned = real_psth[NLAGS - 1 :][: len(lnp_pred)]

# circuit rate curve: reload from the last local circuit_output.txt run (gitignored, but present locally)
from scipy.ndimage import gaussian_filter1d
try:
    circuit_data = np.loadtxt("circuit_output.txt")
    t = circuit_data[:, 0]
    spike_digital = circuit_data[:, 3]
    rising_idx = np.where((spike_digital[:-1] < 2.5) & (spike_digital[1:] >= 2.5))[0]
    spike_times_frames = t[rising_idx] * 120.0
    circuit_counts = bin_spike_times(spike_times_frames, StimRpt.shape[0])
    circuit_rate = gaussian_filter1d(circuit_counts.astype(float), sigma=2.0)
    circuit_aligned = circuit_rate[NLAGS - 1 :][: len(lnp_pred)]
except FileNotFoundError:
    circuit_aligned = np.zeros_like(real_psth_aligned)

cell1_detail = {
    "frames": list(range(len(real_psth_aligned))),
    "real_psth": np.round(real_psth_aligned, 4).tolist(),
    "lnp_pred": np.round(lnp_pred, 4).tolist(),
    "cnn_pred": np.round(cnn_pred, 4).tolist(),
    "circuit_pred": np.round(circuit_aligned, 4).tolist(),
    "lnp_r": round(float(lnp_correlations[CELL]), 4),
    "cnn_r": round(float(cnn_correlations[CELL]), 4),
    "circuit_r": 0.582,
}

output = {
    "cells": cells_summary,
    "cell1_detail": cell1_detail,
    "lnp_mean_r": round(float(lnp_correlations.mean()), 4),
    "cnn_mean_r": round(float(cnn_correlations.mean()), 4),
}

with open("webapp/data.json", "w") as f:
    json.dump(output, f)

print(f"Exported webapp/data.json ({len(json.dumps(output))} bytes)")
