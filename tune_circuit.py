"""Empirically tune the overall gain (and threshold) of the equivalent
circuit so it produces a spike count in the right ballpark for cell 1 on
the repeat stimulus (target: ~310 spikes over the 10s window, matching the
real cell's trial-averaged count)."""

import subprocess
import numpy as np

from make_circuit_netlist import build_netlist

TARGET_SPIKES = 310


def run_circuit(spatial_gain, vth, vhyst, r_int=200e3, c_int=100e-9, timeout=280):
    netlist = build_netlist(spatial_gain=spatial_gain, rectify_gain=1.0,
                             r_int=r_int, c_int=c_int, vth=vth, vhyst=vhyst)
    with open("cell1_circuit.cir", "w") as f:
        f.write(netlist)
    result = subprocess.run(["ngspice", "-b", "cell1_circuit.cir"],
                             capture_output=True, text=True, timeout=timeout)
    data = np.loadtxt("circuit_output.txt")
    spike = data[:, 3]
    rising = np.sum((spike[:-1] < 2.5) & (spike[1:] >= 2.5))
    vint = data[:, 1]
    return rising, vint.max(), result.returncode


if __name__ == "__main__":
    print(f"target spike count: {TARGET_SPIKES}")
    vth, vhyst = 1.0, 0.2
    for gain in [15, 20, 30, 40, 55, 75, 100, 140]:
        n_spikes, vint_max, rc = run_circuit(gain, vth, vhyst)
        print(f"spatial_gain={gain:6.1f}  vth={vth}  -> spikes={n_spikes:4d}  vint_max={vint_max:.3f}  rc={rc}")
