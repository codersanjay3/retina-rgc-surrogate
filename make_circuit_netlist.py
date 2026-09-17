"""Generate an ngspice netlist implementing an analog "equivalent circuit"
model of RGC cell 1, per EJ's Path A brief: a spatial summing network
(2D resistor-grid equivalent, realized as a weighted summer sized from the
cell's own measured STA), two RC cascades forming the biphasic temporal
filter (also fit to the measured STA), and a leaky integrate-and-fire spike
generator built from a real RC integrator + a self-resetting switch."""

import numpy as np
import scipy.io as sio

DATA_DIR = "data/rgcData_Nature08"
NX, NY = 10, 10

spatial_w = np.load("cell1_spatial_weights.npy")  # (100,) STA weights, peak lag
a, tau1_ms, b, tau2_ms = np.load("temporal_fit_params.npy")  # fitted temporal kernel

# component values
C_VAL = 100e-9  # 100 nF, shared by all RC stages
R_FAST = (tau1_ms * 1e-3) / C_VAL
R_SLOW = (tau2_ms * 1e-3) / C_VAL

# tunable overall gains (empirically tuned below in tune_circuit.py)
SPATIAL_GAIN = 8.0
RECTIFY_GAIN = 1.0
R_INT = 200e3
C_INT = 100e-9
VTH = 1.0
VHYST = 0.3

stim = sio.loadmat(f"{DATA_DIR}/Stim_reducedRpt.mat")["Stim"]  # (1200, 100)
nframes, npx = stim.shape
FRAME_MS = 1000.0 / 120.0  # 8.3333 ms/frame
EPS_MS = FRAME_MS / 1000.0  # tiny offset for zero-order hold


def pwl_points(values):
    pts = []
    for i, v in enumerate(values):
        t0 = i * FRAME_MS
        t1 = (i + 1) * FRAME_MS - EPS_MS
        pts.append((t0, v))
        pts.append((t1, v))
    return pts


def build_netlist(spatial_gain=SPATIAL_GAIN, rectify_gain=RECTIFY_GAIN,
                   r_int=R_INT, c_int=C_INT, vth=VTH, vhyst=VHYST):
    lines = []
    lines.append("* Equivalent circuit model of RGC cell 1 (Path A)")
    lines.append(".subckt OPAMP inp inn out")
    lines.append("Eo out 0 inp inn 1meg")
    lines.append(".ends OPAMP")
    lines.append("")
    lines.append(".subckt LPSTAGE in out")
    lines.append("Rin in mid RVAL")
    lines.append("Cmid mid 0 CVAL")
    lines.append("Xbuf mid out out OPAMP")
    lines.append(".ends LPSTAGE")
    lines.append("")

    # 100 PWL pixel voltage sources
    for i in range(npx):
        pts = pwl_points(stim[:, i])
        pwl_str = " ".join(f"{t:.5f}m {v:.4f}" for t, v in pts)
        lines.append(f"Vpx{i+1} px{i+1} 0 PWL({pwl_str})")

    lines.append("")
    # spatial summing network: E-source realizing a weighted-sum amplifier
    ctrl = " ".join(f"(px{i+1},0)" for i in range(npx))
    coefs = " ".join(f"{spatial_gain * w:.6f}" for w in spatial_w)
    lines.append(f"Espatial spatial_sum 0 POLY({npx}) {ctrl} 0 {coefs}")
    lines.append("")

    # fast branch (2-stage RC cascade, tau1)
    lines.append("Xfast1 spatial_sum fast_mid LPSTAGE")
    lines.append("Xfast2 fast_mid fast_out LPSTAGE")
    lines.append("")
    # slow branch (2-stage RC cascade, tau2)
    lines.append("Xslow1 spatial_sum slow_mid LPSTAGE2")
    lines.append("Xslow2 slow_mid slow_out LPSTAGE2")
    lines.append("")
    lines.append(".subckt LPSTAGE2 in out")
    lines.append("Rin in mid RVAL2")
    lines.append("Cmid mid 0 CVAL")
    lines.append("Xbuf mid out out OPAMP")
    lines.append(".ends LPSTAGE2")
    lines.append("")

    # temporal difference: b*slow - a*fast
    # The real 2-stage RC cascade's impulse response is (t/tau^2)e^(-t/tau), which
    # carries an implicit dt/(e*tau) scale relative to the peak-normalized alpha()
    # kernel used for fitting (peak=1 regardless of tau). Since tau1 != tau2, this
    # scale differs per branch and must be corrected, or the fitted a,b balance
    # (derived against the normalized kernel) gets distorted in the real circuit.
    dt_s = FRAME_MS / 1000.0
    scale_fast = np.e * (tau1_ms * 1e-3) / dt_s
    scale_slow = np.e * (tau2_ms * 1e-3) / dt_s
    b_circuit = b / scale_slow
    a_circuit = a / scale_fast
    lines.append(f"Etemporal temporal_out 0 POLY(2) (slow_out,0) (fast_out,0) 0 {b_circuit:.6e} {-a_circuit:.6e}")
    lines.append("")

    # rectify to positive-only drive
    lines.append(f"Bdrive drive_pos 0 V = {rectify_gain}*max(V(temporal_out),0)")
    lines.append("")

    # leaky integrator
    lines.append(f"Rint drive_pos vint {r_int:.1f}")
    lines.append(f"Cint vint 0 {c_int:.3e}")
    lines.append("")

    # self-resetting switch (relaxation-oscillator integrate-and-fire)
    lines.append(f".model SWMOD SW(Ron=50 Roff=1Meg Vt={vth} Vh={vhyst})")
    lines.append("Sreset vint 0 vint 0 SWMOD")
    lines.append("")

    # instrumentation: spike marker node (does not affect circuit dynamics)
    lines.append(f"Bspike spike_out 0 V = (V(vint) > {vth - vhyst/2}) ? 5 : 0")
    lines.append("")

    duration_ms = nframes * FRAME_MS
    lines.append(f".tran {FRAME_MS/10:.5f}m {duration_ms:.3f}m")
    lines.append(".control")
    lines.append("run")
    lines.append("wrdata circuit_output.txt vint spike_out temporal_out spatial_sum")
    lines.append(".endc")
    lines.append(".end")

    netlist = "\n".join(lines)
    # substitute RVAL/CVAL placeholders used inside subckt definitions
    netlist = netlist.replace("RVAL2", f"{R_SLOW:.1f}")
    netlist = netlist.replace("RVAL", f"{R_FAST:.1f}")
    netlist = netlist.replace("CVAL", f"{C_VAL:.3e}")
    return netlist


if __name__ == "__main__":
    netlist = build_netlist()
    with open("cell1_circuit.cir", "w") as f:
        f.write(netlist)
    print(f"Wrote cell1_circuit.cir  (R_FAST={R_FAST:.0f} ohm, R_SLOW={R_SLOW:.0f} ohm, C={C_VAL*1e9:.0f} nF)")
