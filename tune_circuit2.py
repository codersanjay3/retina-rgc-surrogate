from tune_circuit import run_circuit, TARGET_SPIKES

for gain in [82, 88, 95]:
    n_spikes, vint_max, rc = run_circuit(gain, vth=1.0, vhyst=0.2)
    print(f"spatial_gain={gain:6.1f} -> spikes={n_spikes:4d} vint_max={vint_max:.3f} rc={rc}  "
          f"(target={TARGET_SPIKES})")
