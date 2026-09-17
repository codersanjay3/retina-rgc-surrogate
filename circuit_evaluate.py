"""Finalize the tuned circuit, run it, extract spike times, and evaluate
against the real held-out PSTH using the same protocol as the LNP and CNN
evaluations, for a genuine three-way comparison."""

import numpy as np
import matplotlib.pyplot as plt

from tune_circuit import run_circuit
from retina_lib import load_repeat_data, bin_spike_times, NLAGS

FINAL_GAIN = 150.0
VTH, VHYST = 1.0, 0.2

print("Running final tuned circuit...")
n_spikes, vint_max, rc = run_circuit(FINAL_GAIN, VTH, VHYST, timeout=280)
print(f"final circuit: spikes={n_spikes}  vint_max={vint_max:.3f}  returncode={rc}")

data = np.loadtxt("circuit_output.txt")
t = data[:, 0]  # seconds
spike_digital = data[:, 3]

# spike times at rising edges of the digital marker
rising_idx = np.where((spike_digital[:-1] < 2.5) & (spike_digital[1:] >= 2.5))[0]
spike_times_s = t[rising_idx]
print(f"detected {len(spike_times_s)} spike times, first 10 (s): {spike_times_s[:10]}")

# convert to frame units (same units as MtspRGCrpt / SpTimes: frame index, 120 Hz)
FRAME_HZ = 120.0
spike_times_frames = spike_times_s * FRAME_HZ

StimRpt, MtspRpt = load_repeat_data()
CELL = 0
nframes = StimRpt.shape[0]

circuit_counts = bin_spike_times(spike_times_frames, nframes)  # binary/low-count per-frame spikes


def empirical_psth(mtsp_cell, nbins):
    ntrials = mtsp_cell.shape[1]
    total = np.zeros(nbins)
    for trial in range(ntrials):
        times = mtsp_cell[:, trial]
        times = times[times > 0]
        total += bin_spike_times(times, nbins)
    return total / ntrials


real_psth = empirical_psth(MtspRpt[CELL], nframes)

# the circuit produces a single deterministic spike train (no trial noise),
# so smooth it with a small causal-ish kernel before comparing to the
# 600-trial averaged PSTH -- an honest apples-to-apples adjustment, not a fit
from scipy.ndimage import gaussian_filter1d
smooth_sigma_frames = 2.0  # ~16.7 ms, comparable to the frame-scale PSTH structure
circuit_rate_smooth = gaussian_filter1d(circuit_counts.astype(float), sigma=smooth_sigma_frames)

align = slice(NLAGS - 1, None)
r_raw = np.corrcoef(circuit_counts[align], real_psth[align])[0, 1]
r_smooth = np.corrcoef(circuit_rate_smooth[align], real_psth[align])[0, 1]
print(f"circuit vs real PSTH: r (raw binary) = {r_raw:.3f}, r (smoothed) = {r_smooth:.3f}")

lnp_correlations = np.load("lnp_correlations.npy")
cnn_correlations = np.load("cnn_correlations.npy")
print(f"\nCell 1 comparison:")
print(f"  LNP:     r = {lnp_correlations[CELL]:.3f}")
print(f"  CNN:     r = {cnn_correlations[CELL]:.3f}")
print(f"  Circuit: r = {r_smooth:.3f} (smoothed), {r_raw:.3f} (raw spike train)")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(real_psth[align], color="black", lw=1, label="real PSTH (600-trial avg)")
axes[0].plot(circuit_rate_smooth[align], color="tab:green", lw=1, alpha=0.8,
             label=f"circuit (smoothed, r={r_smooth:.3f})")
axes[0].set_ylabel("spikes/frame")
axes[0].legend()
axes[0].set_title("Cell 1: real PSTH vs equivalent-circuit prediction")

axes[1].bar(["LNP", "CNN", "Circuit"],
            [lnp_correlations[CELL], cnn_correlations[CELL], r_smooth],
            color=["tab:blue", "tab:orange", "tab:green"])
axes[1].set_ylabel("correlation with real PSTH (r)")
axes[1].set_title("Cell 1: three-model comparison")

fig.tight_layout()
fig.savefig("three_model_comparison_cell1.png", dpi=150)
print("Saved three_model_comparison_cell1.png")
