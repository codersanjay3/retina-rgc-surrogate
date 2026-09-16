"""LNP (linear-nonlinear-Poisson) baseline: use each cell's STA as a linear
filter, fit a static nonlinearity from training data, then validate the
resulting rate prediction against the real held-out PSTH from the 600-trial
repeat data. This is the classical baseline referenced in Phase 1/3."""

import numpy as np
import matplotlib.pyplot as plt

from retina_lib import (
    load_training_data,
    load_repeat_data,
    compute_sta,
    bin_spike_times,
    NLAGS,
)

Stim, SpTimes = load_training_data()
StimRpt, MtspRpt = load_repeat_data()

CELL = 0  # cell 1


def generator_signal(stim, filt):
    """g[s] = sum_k stim[s+k, :] . filt[k, :] for s = 0..len(stim)-nlags."""
    nlags = filt.shape[0]
    slen = stim.shape[0]
    validlen = slen - nlags + 1
    g = np.zeros(validlen)
    for k in range(nlags):
        g += stim[k : k + validlen, :] @ filt[k, :]
    return g  # g[s] predicts spiking at frame t = s + nlags - 1


def empirical_psth(mtsp_cell, nbins):
    """mtsp_cell: (nmax, ntrials) zero-padded per-trial spike times (frame units).
    Returns mean spike count per frame, averaged across trials."""
    ntrials = mtsp_cell.shape[1]
    total = np.zeros(nbins)
    for trial in range(ntrials):
        times = mtsp_cell[:, trial]
        times = times[times > 0]
        total += bin_spike_times(times, nbins)
    return total / ntrials


# --- fit linear filter (STA) and nonlinearity on training data ---
spike_times = SpTimes[CELL].flatten()
sta, sps, nspikes = compute_sta(Stim, spike_times, nlags=NLAGS)

g_train = generator_signal(Stim, sta)
# g_train[s] predicts frame t = s + NLAGS - 1 -> align spike counts accordingly
spikes_aligned = sps[NLAGS - 1 :][: len(g_train)]

# fit static nonlinearity via quantile-binned histogram method
nbins_nl = 25
order = np.argsort(g_train)
g_sorted = g_train[order]
spikes_sorted = spikes_aligned[order]
bin_edges_idx = np.linspace(0, len(g_sorted), nbins_nl + 1).astype(int)
nl_x = np.zeros(nbins_nl)
nl_y = np.zeros(nbins_nl)
for i in range(nbins_nl):
    lo, hi = bin_edges_idx[i], bin_edges_idx[i + 1]
    nl_x[i] = g_sorted[lo:hi].mean()
    nl_y[i] = spikes_sorted[lo:hi].mean()

# --- apply filter + nonlinearity to repeat stimulus ---
g_rpt = generator_signal(StimRpt, sta)  # length 1200 - NLAGS + 1
pred_rate = np.interp(g_rpt, nl_x, nl_y)  # spikes/frame predicted

# --- empirical PSTH from real repeat spikes ---
mtsp_cell = MtspRpt[CELL]
nbins_rpt = StimRpt.shape[0]
psth = empirical_psth(mtsp_cell, nbins_rpt)
psth_aligned = psth[NLAGS - 1 :][: len(pred_rate)]

r = np.corrcoef(pred_rate, psth_aligned)[0, 1]
print(f"Cell {CELL+1}: correlation between predicted rate and real PSTH = {r:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(psth_aligned, label="real PSTH (600-trial avg)", color="black", lw=1)
axes[0].set_ylabel("spikes/frame")
axes[0].legend()
axes[0].set_title(f"Cell {CELL+1}: real PSTH vs LNP prediction (r = {r:.3f})")

axes[1].plot(pred_rate, label="LNP predicted rate", color="tab:orange", lw=1)
axes[1].set_ylabel("spikes/frame")
axes[1].legend()

axes[2].plot(psth_aligned / psth_aligned.max(), label="real PSTH (norm.)", color="black", lw=1)
axes[2].plot(pred_rate / pred_rate.max(), label="LNP prediction (norm.)", color="tab:orange", lw=1, alpha=0.7)
axes[2].set_ylabel("normalized")
axes[2].set_xlabel("frame (repeat stimulus)")
axes[2].legend()

fig.tight_layout()
fig.savefig("lnp_psth_validation.png", dpi=150)
print("Saved lnp_psth_validation.png")

fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(nl_x, nl_y, "o-")
ax2.set_xlabel("generator signal g(t)")
ax2.set_ylabel("mean spike count")
ax2.set_title(f"Fitted static nonlinearity, cell {CELL+1}")
fig2.tight_layout()
fig2.savefig("lnp_nonlinearity.png", dpi=150)
print("Saved lnp_nonlinearity.png")
