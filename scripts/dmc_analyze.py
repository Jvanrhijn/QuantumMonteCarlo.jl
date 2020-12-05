import sys
import os
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import matplotlib

font = {'family' : 'normal',
        'size'   : 16}

matplotlib.rc('font', **font)


def plot_force_data_trace(flhf, flpulay, flhf_warp, flpulay_warp):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)
    axes[0].plot(flhf, label="No warp")
    axes[0].plot(flhf_warp, label="warp")

    axes[1].plot(flpulay, label="No warp")
    axes[1].plot(flpulay_warp, label="warp")

    axes[2].plot(flhf + flpulay, label="No warp")
    axes[2].plot(flhf_warp + flpulay_warp, label="warp")
    titles = ["Hellmann-Feynman Force", "Pulay Force", "Total Force"]

    axes[0].set_ylabel("Block average force")

    for title, ax in zip(titles, axes):
        ax.legend(); ax.grid()
        ax.set_title(title)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_xlabel("Monte Carlo block")

    axes[2].legend(fancybox=True, shadow=True)

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
    chunks = np.array_split(data, num_points)
    weights_chunks = np.array_split(weights, num_points)
    errs = np.array([error(np.concatenate(chunks[:i+1]), weights=np.concatenate(weights_chunks[:i+1])) for i in range(num_points)])
    means = np.array([np.average(np.concatenate(chunks[:i+1]), weights=np.concatenate(weights_chunks[:i+1])) for i in range(num_points)])
    return means, errs


def plot_errors_over_time(*forces, labels=[], weights=None, npoints=20):
    # forces should be tuples of HF and Pulay forces

    if weights is None:
        weights = np.ones(forces[0][0].shape)

    ns = np.linspace(1, len(forces[0][0]), npoints)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False)
    axes[0].set_ylabel("Error in force")

    for i, (fhf, fp) in enumerate(forces):

        _, hf_errs = error_over_time(fhf, npoints, weights=weights)
        _, pulay_errs = error_over_time(fp, npoints, weights=weights)
        _, total_errs = error_over_time(fhf + fp, npoints, weights=weights)

        axes[0].plot(ns, hf_errs, marker='o', label=labels[i])
        axes[0].set_title("Hellmann-Feynman term")
        axes[1].plot(ns, pulay_errs, marker='o', label=labels[i])
        axes[1].set_title("Pulay term")
        axes[2].plot(ns, total_errs, marker='o', label=labels[i])
        axes[2].set_title("Total force")

    for ax in axes:
        ax.grid()
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_xlabel("Monte Carlo block")
    axes[2].legend(fancybox=True, shadow=True)

    return fig, axes


def plot_forces_over_time(*forces, labels=[], weights=None, npoints=50):
    # forces should be tuples of HF and Pulay forces

    if weights is None:
        weights = np.ones(forces[0][0].shape)

    ns = np.linspace(1, len(forces[0][0]), npoints)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False)
    #axes[2].plot(ns, [np.pi**2/4]*len(ns), label="Exact", color="black", linestyle="--")
    axes[2].plot(ns, [3.3014]*len(ns), label="Exact", color="black", linestyle="--")
    axes[0].set_ylabel("Force")

    for i, (fhf, fp) in enumerate(forces):

        hf_means, hf_errs = error_over_time(fhf, npoints, weights=weights)
        pulay_means, pulay_errs = error_over_time(fp, npoints, weights=weights)
        total_means, total_errs = error_over_time(fhf + fp, npoints, weights=weights)


        axes[0].fill_between(ns, hf_means - hf_errs, hf_means + hf_errs, alpha=0.2)
        axes[0].plot(ns, hf_means, label=labels[i])
        axes[0].set_title("Hellmann-Feynman term")

        axes[1].fill_between(ns, pulay_means - pulay_errs, pulay_means + pulay_errs, alpha=0.2)
        axes[1].plot(ns, pulay_means, label=labels[i])
        axes[1].set_title("Pulay term")

        axes[2].fill_between(ns, total_means - total_errs, total_means + total_errs, alpha=0.2)
        axes[2].plot(ns, total_means, label=labels[i])
        axes[2].set_title("Total force")


    for ax in axes:
        ax.grid()
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax.set_xlabel("Monte Carlo block")
    axes[2].legend(fancybox=True, shadow=True)

    return fig, axes


def compute_forces(fpath):
    data = h5py.File(fpath, "r")

    weights = data["Weight"][()][1:]
    energy = np.average(data["Local energy"][()][1:], weights=weights)

    # Get Green's function derivatives
    sderiv_sum = data["grad s"][()][1:]        
    sderiv_sum_warp = data["grad s (warp)"][()][1:]        

    gderiv_sum_pq = data["grad g (p/q)"][()][1:]        
    gderiv_sum_warp_pq = data["grad g (warp, p/q)"][()][1:]

    gderiv_sum = data["grad g"][()][1:]
    gderiv_sum_warp = data["grad g (warp)"][()][1:]        

    # Get j (sum) derivative
    jderiv_sum = data["sum grad log j"][()][1:]
    jac_logderiv = data["grad log j"][()][1:]
    jderiv_sum_approx = data["sum grad log j approx"][()][1:]
    jac_logderiv_approx = data["grad log j approx"][()][1:]

    # Get products of Local energy and Green's function derivatives
    el_times_sderiv_sum = data["Local energy * grad s"][()][1:] 
    el_times_sderiv_sum_warp = data["Local energy * grad s (warp)"][()][1:]       

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
            +   0*(el_times_jderiv_sum - energy*jderiv_sum) \
            +   0*(el_times_jac_logderiv - energy*jac_logderiv) \
            )


    force_pulay_exact_pq = -(
                el_times_gderiv_sum_pq - energy*gderiv_sum_pq \
            )

    force_pulay_exact_warp_pq = -(
                el_times_gderiv_sum_warp_pq - energy*gderiv_sum_warp_pq \
            +   0*(el_times_jderiv_sum - energy*jderiv_sum) \
            +   0*(el_times_jac_logderiv - energy*jac_logderiv) \
            )

    force_pulay_exact_warp_pq_approx = -(
                el_times_gderiv_sum_warp_pq - energy*gderiv_sum_warp_pq \
            +   0*(el_times_jderiv_sum_approx - energy*jderiv_sum_approx) \
            +   0*(el_times_jac_logderiv_approx - energy*jac_logderiv_approx) \
            )

    force_pulay_vd = -(
            2 * (el_times_psilogderiv - energy*psilogderiv) \
            +   (el_times_sderiv_sum - energy*sderiv_sum)
            )

    force_pulay_vd_warp = -(
            2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
            +   (el_times_sderiv_sum_warp - energy*sderiv_sum_warp) \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )

    data.close()

    return force_hf.flatten(), \
           force_hf_warp.flatten(), \
           force_pulay_exact.flatten(), \
           force_pulay_exact_warp.flatten(), \
           force_pulay_vd.flatten(), \
           force_pulay_vd_warp.flatten(), \
           force_pulay_exact_pq.flatten(), \
           force_pulay_exact_warp_pq.flatten(), \
           force_pulay_exact_warp_pq_approx.flatten(), \
           weights.flatten()


force_hf, force_hf_warp, \
        force_pulay_exact, force_pulay_exact_warp, \
        force_pulay_vd, force_pulay_vd_warp, \
        force_pulay_exact_pq, force_pulay_exact_warp_pq, force_pulay_exact_warp_pq_approx, \
        weights \
    = compute_forces(sys.argv[1])

fhf = np.average(force_hf, weights=weights)
fhf_err = error(force_hf, weights=weights)

fhf_warp = np.average(force_hf_warp, weights=weights)
fhf_err_warp = error(force_hf_warp, weights=weights)

fpulay_exact = np.average(force_pulay_exact, weights=weights)
fpulay_exact_err = error(force_pulay_exact, weights=weights)

fpulay_vd  = np.average(force_pulay_vd, weights=weights)
fpulay_vd_err = error(force_pulay_vd, weights=weights)

fpulay_exact_warp = np.average(force_pulay_exact_warp, weights=weights)
fpulay_exact_err_warp = error(force_pulay_exact_warp, weights=weights)

fpulay_exact_pq = np.average(force_pulay_exact_pq, weights=weights)
fpulay_exact_pq_err = error(force_pulay_exact_pq, weights=weights)

fpulay_exact_pq_warp = np.average(force_pulay_exact_warp_pq, weights=weights)
fpulay_exact_pq_err_warp = error(force_pulay_exact_warp_pq, weights=weights)

fpulay_vd_warp = np.average(force_pulay_vd_warp, weights=weights)
fpulay_vd_err_warp = error(force_pulay_vd_warp, weights=weights)

ftot_exact = np.average(force_hf + force_pulay_exact, weights=weights)
ftot_exact_err = error(force_hf + force_pulay_exact, weights=weights)

ftot_vd = np.average(force_hf + force_pulay_vd, weights=weights)
ftot_vd_err = error(force_hf + force_pulay_vd, weights=weights)

ftot_exact_warp = np.average(force_hf_warp + force_pulay_exact_warp, weights=weights)
ftot_exact_err_warp = error(force_hf_warp + force_pulay_exact_warp, weights=weights)

ftot_vd_warp = np.average(force_hf_warp + force_pulay_vd_warp, weights=weights)
ftot_vd_err_warp = error(force_hf_warp + force_pulay_vd_warp, weights=weights)

ftot_exact_pq = np.average(force_hf + force_pulay_exact_pq, weights=weights)
ftot_exact_pq_err = error(force_hf + force_pulay_exact_pq, weights=weights)

ftot_exact_pq_warp = np.average(force_hf_warp + force_pulay_exact_warp_pq, weights=weights)
ftot_exact_pq_err_warp = error(force_hf_warp + force_pulay_exact_warp_pq, weights=weights)

ftot_exact_pq_warp_approx = np.average(force_hf_warp + force_pulay_exact_warp_pq_approx, weights=weights)
ftot_exact_pq_err_warp_approx = error(force_hf_warp + force_pulay_exact_warp_pq_approx, weights=weights)

print(f"HF force:                                  {fhf:.5f} +/- {fhf_err:.5f}")
print(f"HF force (warp):                           {fhf_warp:.5f} +/- {fhf_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (exact):                       {fpulay_exact:.5f} +/- {fpulay_exact_err:.5f}")
print(f"Pulay force (exact, warp):                 {fpulay_exact_warp:.5f} +/- {fpulay_exact_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (exact, p/q):                   {fpulay_exact_pq:.5f} +/- {fpulay_exact_pq_err:.5f}")
print(f"Pulay force (exact, warp, p/q):             {fpulay_exact_pq_warp:.5f} +/- {fpulay_exact_pq_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (vd):                          {fpulay_vd:.5f} +/- {fpulay_vd_err:.5f}")
print(f"Pulay force (vd, warp):                    {fpulay_vd_warp:.5f} +/- {fpulay_vd_err_warp:.5f}")
print(f"\n")
print(f"Total force (exact):                       {ftot_exact:.5f} +/- {ftot_exact_err:.5f}")
print(f"Total force (exact, warp):                 {ftot_exact_warp:.5f} +/- {ftot_exact_err_warp:.5f}")
print(f"Total force (exact, p/q):                  {ftot_exact_pq:.5f} +/- {ftot_exact_pq_err:.5f}")
print(f"Total force (exact, warp, p/q):            {ftot_exact_pq_warp:.5f} +/- {ftot_exact_pq_err_warp:.5f}")
print(f"Total force (exact, warp, p/q approx J):   {ftot_exact_pq_warp_approx:.5f} +/- {ftot_exact_pq_err_warp_approx:.5f}")
print(f"Total force (vd):                          {ftot_vd:.5f} +/- {ftot_vd_err:.5f}")
print(f"Total force (vd, warp):                    {ftot_vd_warp:.5f} +/- {ftot_vd_err_warp:.5f}")

npoints = 20

plot_forces_over_time(
    (force_hf, force_pulay_exact_pq), 
    #(force_hf_warp, force_pulay_exact_warp), 
    #(force_hf, force_pulay_vd), 
    #(force_hf_warp, force_pulay_vd_warp), 
    (force_hf_warp, force_pulay_exact_warp_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=[
        "Not warped", 
        #"Warped, naive"
        #"VD", 
        #"VD, warp", 
        "Warped", 
        #"Warped, approx. J"
    ], 
    weights=weights
)
plt.tight_layout()

plot_errors_over_time(
    (force_hf, force_pulay_exact_pq), 
    #(force_hf_warp, force_pulay_exact_warp), 
    #(force_hf, force_pulay_vd), 
    #(force_hf_warp, force_pulay_vd_warp), 
    (force_hf_warp, force_pulay_exact_warp_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=[
        "Not warped", 
        #"Warped, naive"
        #"VD", 
        #"VD, warp", 
        "Warped", 
        #"Warped, approx. J"
    ], 
    weights=weights
)
plt.tight_layout()

plot_force_data_trace(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq_approx)
plt.tight_layout()

plt.show()
