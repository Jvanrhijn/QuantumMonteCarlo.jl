import sys
from pydmc import *
from util import *
import os
import pandas as pd


force_hf, force_hf_warp, \
        force_pulay_exact, force_pulay_exact_warp, \
        force_pulay_vd, force_pulay_vd_warp, \
        force_pulay_exact_nocutoff, force_pulay_exact_warp_nocutoff, \
        weights \
    = DMCForcesInput().compute_forces(sys.argv[1])

steps_per_block = 1

fhf = np.average(force_hf, weights=weights)
fhf_err = block_error(force_hf, steps_per_block, weights=weights)

fhf_warp = np.average(force_hf_warp, weights=weights)
fhf_err_warp = block_error(force_hf_warp, steps_per_block, weights=weights)

fpulay_exact = np.average(force_pulay_exact, weights=weights)
fpulay_exact_err = block_error(force_pulay_exact, steps_per_block, weights=weights)

fpulay_vd  = np.average(force_pulay_vd, weights=weights)
fpulay_vd_err = block_error(force_pulay_vd, steps_per_block, weights=weights)

fpulay_exact_warp = np.average(force_pulay_exact_warp, weights=weights)
fpulay_exact_err_warp = block_error(force_pulay_exact_warp, steps_per_block, weights=weights)

fpulay_exact_nocutoff = np.average(force_pulay_exact_nocutoff, weights=weights)
fpulay_exact_nocutoff_err = block_error(force_pulay_exact_nocutoff, steps_per_block, weights=weights)

fpulay_exact_nocutoff_warp = np.average(force_pulay_exact_warp_nocutoff, weights=weights)
fpulay_exact_nocutoff_err_warp = block_error(force_pulay_exact_warp_nocutoff, steps_per_block, weights=weights)

fpulay_vd_warp = np.average(force_pulay_vd_warp, weights=weights)
fpulay_vd_err_warp = block_error(force_pulay_vd_warp, steps_per_block, weights=weights)

ftot_exact = np.average(force_hf + force_pulay_exact, weights=weights)
ftot_exact_err = block_error(force_hf + force_pulay_exact, steps_per_block, weights=weights)

ftot_vd = np.average(force_hf + force_pulay_vd, weights=weights)
ftot_vd_err = block_error(force_hf + force_pulay_vd, steps_per_block, weights=weights)

ftot_exact_warp = np.average(force_hf_warp + force_pulay_exact_warp, weights=weights)
ftot_exact_err_warp = block_error(force_hf_warp + force_pulay_exact_warp, steps_per_block, weights=weights)

ftot_vd_warp = np.average(force_hf_warp + force_pulay_vd_warp, weights=weights)
ftot_vd_err_warp = block_error(force_hf_warp + force_pulay_vd_warp, steps_per_block, weights=weights)

ftot_mm = np.average(force_hf_warp + force_pulay_exact, weights=weights)
ftot_mm_err = block_error(force_hf_warp + force_pulay_exact, steps_per_block, weights=weights)

ftot_exact_nocutoff = np.average(force_hf + force_pulay_exact_nocutoff, weights=weights)
ftot_exact_nocutoff_err = block_error(force_hf + force_pulay_exact_nocutoff, steps_per_block, weights=weights)

ftot_exact_nocutoff_warp = np.average(force_hf_warp + force_pulay_exact_warp_nocutoff, weights=weights)
ftot_exact_nocutoff_err_warp = block_error(force_hf_warp + force_pulay_exact_warp_nocutoff, steps_per_block, weights=weights)

print(f"HF force:                    {fhf:.5f} +/- {fhf_err:.5f}")
print(f"HF force (warp):             {fhf_warp:.5f} +/- {fhf_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (exact):         {fpulay_exact:.5f} +/- {fpulay_exact_err:.5f}")
print(f"Pulay force (exact, warp):   {fpulay_exact_warp:.5f} +/- {fpulay_exact_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (exact, nc):       {fpulay_exact_nocutoff:.5f} +/- {fpulay_exact_nocutoff_err:.5f}")
print(f"Pulay force (exact, warp, nc): {fpulay_exact_nocutoff_warp:.5f} +/- {fpulay_exact_nocutoff_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (vd):            {fpulay_vd:.5f} +/- {fpulay_vd_err:.5f}")
print(f"Pulay force (vd, warp):      {fpulay_vd_warp:.5f} +/- {fpulay_vd_err_warp:.5f}")
print(f"\n")
print(f"Total force (exact):         {ftot_exact:.5f} +/- {ftot_exact_err:.5f}")
print(f"Total force (exact, warp):   {ftot_exact_warp:.5f} +/- {ftot_exact_err_warp:.5f}")
print(f"Total force (exact, nc):         {ftot_exact_nocutoff:.5f} +/- {ftot_exact_nocutoff_err:.5f}")
print(f"Total force (exact, warp, nc):   {ftot_exact_nocutoff_warp:.5f} +/- {ftot_exact_nocutoff_err_warp:.5f}")
print(f"Total force (vd):            {ftot_vd:.5f} +/- {ftot_vd_err:.5f}")
print(f"Total force (vd, warp):      {ftot_vd_warp:.5f} +/- {ftot_vd_err_warp:.5f}")
print(f"Total force (mix-match)      {ftot_mm:.5f} +/- {ftot_mm_err:.5f}")

npoints = 20

fig, _ = plot_force_data_trace(force_hf, force_pulay_exact, force_hf_warp, force_pulay_exact_warp, bin_size=steps_per_block)
fig.suptitle("Exact force")
fig, _ = plot_error_over_time(force_hf, force_pulay_exact, force_hf_warp, force_pulay_exact_warp, npoints, steps_per_block)
fig.suptitle("Exact force")

fig, _ = plot_force_data_trace(force_hf, force_pulay_vd, force_hf_warp, force_pulay_vd_warp, bin_size=steps_per_block)
fig.suptitle("VD force")
fig, _ = plot_error_over_time(force_hf, force_pulay_vd, force_hf_warp, force_pulay_vd_warp, npoints, steps_per_block)
fig.suptitle("VD force")

fig, _ = plot_force_data_trace(force_hf, force_pulay_exact_nocutoff, force_hf_warp, force_pulay_exact_warp_nocutoff, bin_size=steps_per_block)
fig.suptitle("Exact force, no cutoff")
fig, _ = plot_error_over_time(force_hf, force_pulay_exact_nocutoff, force_hf_warp, force_pulay_exact_warp_nocutoff, npoints, steps_per_block)
fig.suptitle("Exact force, no cutoff")

plt.show()
