"""Run the LNP baseline (STA filter + fitted static nonlinearity) for all 27
cells and report the predicted-vs-real PSTH correlation for each, as the
population-level Phase 1 baseline to beat with a CNN later."""

import numpy as np
import matplotlib.pyplot as plt

from retina_lib import (
    load_training_data,
    load_repeat_data,
    compute_sta,
    bin_spike_times,
    NLAGS,
)


def generator_signal(stim, filt):
    nlags = filt.shape[0]
    slen = stim.shape[0]
    validlen = slen - nlags + 1
    g = np.zeros(validlen)
    for k in range(nlags):
        g += stim[k : k + validlen, :] @ filt[k, :]
    return g


def empirical_psth(mtsp_cell, nbins):
    ntrials = mtsp_cell.shape[1]
    total = np.zeros(nbins)
    for trial in range(ntrials):
        times = mtsp_cell[:, trial]
        times = times[times > 0]
        total += bin_spike_times(times, nbins)
    return total / ntrials


def fit_nonlinearity(g_train, spikes_aligned, nbins_nl=25):
    order = np.argsort(g_train)
    g_sorted = g_train[order]
    spikes_sorted = spikes_aligned[order]
    edges = np.linspace(0, len(g_sorted), nbins_nl + 1).astype(int)
    nl_x = np.zeros(nbins_nl)
    nl_y = np.zeros(nbins_nl)
    for i in range(nbins_nl):
        lo, hi = edges[i], edges[i + 1]
        nl_x[i] = g_sorted[lo:hi].mean()
        nl_y[i] = spikes_sorted[lo:hi].mean()
    return nl_x, nl_y


Stim, SpTimes = load_training_data()
StimRpt, MtspRpt = load_repeat_data()
ncells = SpTimes.shape[0]

correlations = []
for c in range(ncells):
    spike_times = SpTimes[c].flatten()
    sta, sps, nspikes = compute_sta(Stim, spike_times, nlags=NLAGS)

    g_train = generator_signal(Stim, sta)
    spikes_aligned = sps[NLAGS - 1 :][: len(g_train)]
    nl_x, nl_y = fit_nonlinearity(g_train, spikes_aligned)

    g_rpt = generator_signal(StimRpt, sta)
    pred_rate = np.interp(g_rpt, nl_x, nl_y)

    psth = empirical_psth(MtspRpt[c], StimRpt.shape[0])
    psth_aligned = psth[NLAGS - 1 :][: len(pred_rate)]

    r = np.corrcoef(pred_rate, psth_aligned)[0, 1]
    correlations.append(r)
    print(f"cell {c+1:2d}: n_spikes={int(nspikes):6d}  r={r:.3f}")

correlations = np.array(correlations)
print(f"\nmean r = {correlations.mean():.3f}, median r = {np.median(correlations):.3f}, "
      f"min = {correlations.min():.3f}, max = {correlations.max():.3f}")

np.save("lnp_correlations.npy", correlations)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(np.arange(1, ncells + 1), correlations)
ax.axhline(correlations.mean(), color="red", linestyle="--", label=f"mean = {correlations.mean():.3f}")
ax.set_xlabel("cell")
ax.set_ylabel("predicted vs real PSTH correlation (r)")
ax.set_title("LNP baseline: PSTH prediction quality across all 27 cells")
ax.legend()
fig.tight_layout()
fig.savefig("lnp_correlations_all_cells.png", dpi=150)
print("Saved lnp_correlations_all_cells.png")
