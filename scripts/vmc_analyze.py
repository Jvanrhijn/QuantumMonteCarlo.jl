import sys
import os
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py


def plot_force_data_trace(flhf, flpulay, flhf_warp, flpulay_warp):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    axes[0].plot(flhf, label="No warp")
    axes[0].plot(flhf_warp, label="warp")

    axes[1].plot(flpulay, label="No warp")
    axes[1].plot(flpulay_warp, label="warp")

    axes[2].plot(flhf + flpulay, label="No warp")
    axes[2].plot(flhf_warp + flpulay_warp, label="warp")
    titles = ["Hellmann-Feynman Force", "Pulay Force", "Total Force"]
    for title, ax in zip(titles, axes):
        ax.legend(); ax.grid()
        ax.set_title(title)
    return fig, axes


def error(block_means, weights=None):
    if weights is None:
        weights = np.ones(block_means.shape)
    mean = np.average(block_means, weights=weights)
    meansq = np.average(block_means**2, weights=weights)
    return math.sqrt(abs(meansq - mean**2) / len(block_means))


# function to compute error bar given a slice of local force
def error_over_time(data, num_points, weights=None):
    if weights is None:
        weights = np.ones(data.shape)
    partition_size = len(data) // num_points
    errs = np.array([error(data[:(i+1)*partition_size], weights=weights[:(i+1)*partition_size]) for i in range(num_points)])
    means = np.array([np.average(data[:(i+1)*partition_size], weights=weights[:(i+1)*partition_size]) for i in range(num_points)])
    return means, errs


def plot_error_over_time(flhf, flpulay, flhf_warp, flpulay_warp, npoints, weights=None):
    if weights is None:
        weights = np.ones(flhf.shape)
    ns = np.linspace(1, len(flhf), npoints)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), sharey=False)
    for i, (f, fwarp) in enumerate([(flhf, flhf_warp), (flpulay, flpulay_warp), (flhf + flpulay, flhf_warp + flpulay_warp)]):
        means, errs = error_over_time(f, npoints, weights=weights)
        means_warp, errs_warp = error_over_time(fwarp, npoints, weights=weights)
        axes[0, i].errorbar(ns, means, yerr=errs, marker='o', label="Not warped")
        axes[1, i].plot(ns, errs, marker='o', label="Not warped")
        axes[0, i].errorbar(ns, means_warp, yerr=errs_warp, marker='o', label="Warped")
        axes[1, i].plot(ns, errs_warp, marker='o', label="Warped")
        axes[0, i].legend(); axes[0, i].grid()
        axes[1, i].legend(); axes[1, i].grid()
    return fig, axes


def compute_forces(fpath):
    data = h5py.File(fpath, "r")

    weights = data["Weight"][()][1:]
    energy = np.average(data["Local energy"][()][1:], weights=weights)

    gderiv_sum_pq = data["grad g (p/q)"][()][1:]        
    gderiv_sum_warp_pq = data["grad g (warp, p/q)"][()][1:]

    gderiv_sum = data["grad g"][()][1:]
    gderiv_sum_warp = data["grad g (warp)"][()][1:]        

    # Get j (sum) derivative
    jderiv_sum = data["sum grad log j"][()][1:]
    jac_logderiv = data["grad log j"][()][1:]
    jderiv_sum_approx = data["sum grad log j approx"][()][1:]
    jac_logderiv_approx = data["grad log j approx"][()][1:]

    el_times_gderiv_sum_pq = data["Local energy * grad g (p/q)"][()][1:]        
    el_times_gderiv_sum_warp_pq = data["Local energy * grad g (warp, p/q)"][()][1:]        

    el_times_gderiv_sum = data["Local energy * grad g"][()][1:]        
    el_times_gderiv_sum_warp = data["Local energy * grad g (warp)"][()][1:]        

    # Get product of Local energy and j (sum) derivative
    el_times_jderiv_sum = data["Local energy * sum grad log j"][()][1:]
    el_times_jac_logderiv = data["Local energy * grad log j"][()][1:]

    el_times_jderiv_sum_approx = data["Local energy * sum grad log j approx"][()][1:]
    el_times_jac_logderiv_approx = data["Local energy * grad log j approx"][()][1:]

    # Get local e derivative
    local_e_deriv = data["grad el"][()][1:]
    local_e_deriv_warp = data["grad el (warp)"][()][1:]

    # Get psi derivative
    psilogderiv = data["grad log psi"][()][1:]
    psilogderiv_warp = data["grad log psi (warp)"][()][1:]

    el_times_psilogderiv = data["Local energy * grad log psi"][()][1:]
    el_times_psilogderiv_warp = data["Local energy * grad log psi (warp)"][()][1:]

    # Hellmann-Feynman force
    force_hf = -local_e_deriv
    force_hf_warp = -local_e_deriv_warp

    # Pulay force
    force_pulay_exact = -(
                el_times_gderiv_sum - energy*gderiv_sum \
            )

    force_pulay_exact_warp = -(
                (el_times_gderiv_sum_warp - energy*gderiv_sum_warp) \
            +   (el_times_jderiv_sum - energy*jderiv_sum) \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )


    force_pulay_exact_pq = -(
                el_times_gderiv_sum_pq - energy*gderiv_sum_pq \
            )

    force_pulay_exact_warp_pq = -(
                el_times_gderiv_sum_warp_pq - energy*gderiv_sum_warp_pq \
            +   el_times_jderiv_sum - energy*jderiv_sum \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )

    force_pulay_exact_warp_pq_approx = -(
                el_times_gderiv_sum_warp_pq - energy*gderiv_sum_warp_pq \
            +   el_times_jderiv_sum_approx - energy*jderiv_sum_approx \
            +   (el_times_jac_logderiv_approx - energy*jac_logderiv_approx) \
            )

    force_pulay_ = -(
            2 * (el_times_psilogderiv - energy*psilogderiv) \
            )

    force_pulay__warp = -(
            2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )

    data.close()

    return force_hf.flatten(), \
           force_hf_warp.flatten(), \
           force_pulay_exact.flatten(), \
           force_pulay_exact_warp.flatten(), \
           force_pulay_.flatten(), \
           force_pulay__warp.flatten(), \
           force_pulay_exact_pq.flatten(), \
           force_pulay_exact_warp_pq.flatten(), \
           force_pulay_exact_warp_pq_approx.flatten(), \
           weights.flatten()


force_hf, force_hf_warp, \
        force_pulay_exact, force_pulay_exact_warp, \
        force_pulay_, force_pulay__warp, \
        force_pulay_exact_pq, force_pulay_exact_warp_pq, force_pulay_exact_warp_pq_approx, \
        weights \
    = compute_forces(sys.argv[1])

fhf = np.average(force_hf, weights=weights)
fhf_err = error(force_hf, weights=weights)

fhf_warp = np.average(force_hf_warp, weights=weights)
fhf_err_warp = error(force_hf_warp, weights=weights)

fpulay_exact = np.average(force_pulay_exact, weights=weights)
fpulay_exact_err = error(force_pulay_exact, weights=weights)

fpulay_  = np.average(force_pulay_, weights=weights)
fpulay__err = error(force_pulay_, weights=weights)

fpulay_exact_warp = np.average(force_pulay_exact_warp, weights=weights)
fpulay_exact_err_warp = error(force_pulay_exact_warp, weights=weights)

fpulay_exact_pq = np.average(force_pulay_exact_pq, weights=weights)
fpulay_exact_pq_err = error(force_pulay_exact_pq, weights=weights)

fpulay_exact_pq_warp = np.average(force_pulay_exact_warp_pq, weights=weights)
fpulay_exact_pq_err_warp = error(force_pulay_exact_warp_pq, weights=weights)

fpulay__warp = np.average(force_pulay__warp, weights=weights)
fpulay__err_warp = error(force_pulay__warp, weights=weights)

ftot_exact = np.average(force_hf + force_pulay_exact, weights=weights)
ftot_exact_err = error(force_hf + force_pulay_exact, weights=weights)

ftot_ = np.average(force_hf + force_pulay_, weights=weights)
ftot__err = error(force_hf + force_pulay_, weights=weights)

ftot_exact_warp = np.average(force_hf_warp + force_pulay_exact_warp, weights=weights)
ftot_exact_err_warp = error(force_hf_warp + force_pulay_exact_warp, weights=weights)

ftot__warp = np.average(force_hf_warp + force_pulay__warp, weights=weights)
ftot__err_warp = error(force_hf_warp + force_pulay__warp, weights=weights)

ftot_exact_pq = np.average(force_hf + force_pulay_exact_pq, weights=weights)
ftot_exact_pq_err = error(force_hf + force_pulay_exact_pq, weights=weights)

ftot_exact_pq_warp = np.average(force_hf_warp + force_pulay_exact_warp_pq, weights=weights)
ftot_exact_pq_err_warp = error(force_hf_warp + force_pulay_exact_warp_pq, weights=weights)

ftot_exact_pq_warp_approx = np.average(force_hf_warp + force_pulay_exact_warp_pq_approx, weights=weights)
ftot_exact_pq_err_warp_approx = error(force_hf_warp + force_pulay_exact_warp_pq_approx, weights=weights)

print(f"HF force:                                     {fhf:.5f} +/- {fhf_err:.5f}")
print(f"HF force (warp):                              {fhf_warp:.5f} +/- {fhf_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (Green's):                        {fpulay_exact:.5f} +/- {fpulay_exact_err:.5f}")
print(f"Pulay force (Green's, warp):                  {fpulay_exact_warp:.5f} +/- {fpulay_exact_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (Green's, p/q):                   {fpulay_exact_pq:.5f} +/- {fpulay_exact_pq_err:.5f}")
print(f"Pulay force (Green's, warp, p/q):             {fpulay_exact_pq_warp:.5f} +/- {fpulay_exact_pq_err_warp:.5f}")
print(f"\n")
print(f"Pulay force:                                  {fpulay_:.5f} +/- {fpulay__err:.5f}")
print(f"Pulay force (warp):                           {fpulay__warp:.5f} +/- {fpulay__err_warp:.5f}")
print(f"\n")
print(f"Total force (Green's):                        {ftot_exact:.5f} +/- {ftot_exact_err:.5f}")
print(f"Total force (Green's, warp):                  {ftot_exact_warp:.5f} +/- {ftot_exact_err_warp:.5f}")
print(f"\n")
print(f"Total force (Green's, p/q):                   {ftot_exact_pq:.5f} +/- {ftot_exact_pq_err:.5f}")
print(f"Total force (Green's, warp, p/q):             {ftot_exact_pq_warp:.5f} +/- {ftot_exact_pq_err_warp:.5f}")
print(f"Total force (Green's, warp, p/q, approx J):   {ftot_exact_pq_warp_approx:.5f} +/- {ftot_exact_pq_err_warp_approx:.5f}")
print(f"Total force:                                  {ftot_:.5f} +/- {ftot__err:.5f}")
print(f"Total force (warp):                           {ftot__warp:.5f} +/- {ftot__err_warp:.5f}")

npoints = 20

fig, _ = plot_force_data_trace(force_hf, force_pulay_exact, force_hf_warp, force_pulay_exact_warp)
fig.suptitle("Green's function force")
fig, _ = plot_error_over_time(force_hf, force_pulay_exact, force_hf_warp, force_pulay_exact_warp, npoints, weights=weights)
fig.suptitle("Green's function force")

fig, _ = plot_force_data_trace(force_hf, force_pulay_, force_hf_warp, force_pulay__warp)
fig.suptitle("Force")
fig, _ = plot_error_over_time(force_hf, force_pulay_, force_hf_warp, force_pulay__warp, npoints, weights=weights)
fig.suptitle("Force")

fig, _ = plot_force_data_trace(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq)
fig.suptitle("Green's function force, p/q")
fig, _ = plot_error_over_time(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq, npoints, weights=weights)
fig.suptitle("Green's function force, p/q")

fig, _ = plot_force_data_trace(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq_approx)
fig.suptitle("Green's function force, p/q, approx J")
fig, _ = plot_error_over_time(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq_approx, npoints, weights=weights)
fig.suptitle("Green's function force, p/q, approx J")

plt.show()
