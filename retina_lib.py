"""Shared data loading and STA utilities for the retina RGC baseline project."""

import numpy as np
import scipy.io as sio

DATA_DIR = "data/rgcData_Nature08"
NX, NY = 10, 10
NPX = NX * NY
NLAGS = 20


def load_training_data():
    Stim = sio.loadmat(f"{DATA_DIR}/Stim_reduced.mat")["Stim"]  # (144000, 100)
    SpTimes = sio.loadmat(f"{DATA_DIR}/SpTimesRGC.mat")["SpTimes"][0]  # (27,) object array
    return Stim, SpTimes


def load_repeat_data():
    StimRpt = sio.loadmat(f"{DATA_DIR}/Stim_reducedRpt.mat")["Stim"]  # (1200, 100)
    MtspRpt = sio.loadmat(f"{DATA_DIR}/MtspRGCrpt.mat")["MtspRGCrpt"][0]  # (27,) object array, each (nmax, 600)
    return StimRpt, MtspRpt


def load_stim_coords():
    return sio.loadmat(f"{DATA_DIR}/StimCoords.mat")["StimCoords"]  # (27, 4)


def bin_spike_times(spike_times, nbins):
    """Bin frame-index spike times into per-frame counts, matching
    MATLAB's hist(SpTimes{1}, 0.5:1:slen) (bin k catches times in [k-0.5, k+0.5))."""
    edges = np.arange(0.5, nbins + 1.5, 1.0)  # nbins+1 edges -> nbins bins, centers at 1..nbins
    counts, _ = np.histogram(spike_times, bins=edges)
    return counts


def compute_sta(Stim, spike_times, nlags=NLAGS):
    """Direct port of loadDataAndComputeSTA.m for one cell."""
    slen = Stim.shape[0]
    sps = bin_spike_times(spike_times, slen)
    sta = np.zeros((nlags, Stim.shape[1]))
    for jj in range(nlags):
        row = nlags - jj - 1
        sta[row, :] = sps[jj:] @ Stim[: slen - jj, :]
    nspikes = sps.sum()
    if nspikes > 0:
        sta = sta / nspikes
    return sta, sps, nspikes
