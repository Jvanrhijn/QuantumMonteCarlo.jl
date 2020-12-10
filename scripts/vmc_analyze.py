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
        ax.grid()
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

    #axes[2].plot(ns, [3.43196]*len(ns), label="Exact", color="black", linestyle='--')
    axes[2].plot(ns, [2.5]*len(ns), label="Exact", color="black", linestyle='--')
    axes[0].set_ylabel("Force")

    for i, (fhf, fp) in enumerate(forces):

        hf_means, hf_errs = error_over_time(fhf, npoints, weights=weights)
        pulay_means, pulay_errs = error_over_time(fp, npoints, weights=weights)
        total_means, total_errs = error_over_time(fhf + fp, npoints, weights=weights)

        #axes[0].errorbar(ns, hf_means, yerr=hf_errs, marker='o', label=labels[i])
        axes[0].fill_between(ns, hf_means - hf_errs, hf_means + hf_errs, alpha=0.2)#, yerr=hf_errs, marker='o', label=labels[i])
        axes[0].plot(ns, hf_means, label=labels[i])#, yerr=hf_errs, marker='o', label=labels[i])
        axes[0].set_title("Hellmann-Feynman term")

        #axes[1].errorbar(ns, pulay_means, yerr=pulay_errs, marker='o', label=labels[i])
        axes[1].fill_between(ns, pulay_means - pulay_errs, pulay_means + pulay_errs, alpha=0.2)#, yerr=hf_errs, marker='o', label=labels[i])
        axes[1].plot(ns, pulay_means, label=labels[i])#, yerr=hf_errs, marker='o', label=labels[i])
        axes[1].set_title("Pulay term")

        #axes[2].errorbar(ns, total_means, yerr=total_errs, marker='o', label=labels[i])
        axes[2].fill_between(ns, total_means - total_errs, total_means + total_errs, alpha=0.2)#, yerr=hf_errs, marker='o', label=labels[i])
        axes[2].plot(ns, total_means, label=labels[i])#, yerr=hf_errs, marker='o', label=labels[i])
        axes[2].set_title("Total force")

    #axes[2].plot(ns, [3.43196]*len(ns), label="PES", color="black")

    for ax in axes:
        ax.set_xlabel("Monte Carlo block")
        ax.grid()
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    axes[2].legend(fancybox=True, shadow=True)

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

    warpfacs = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0][4:]
    force_hf_pathaks = []
    force_pulay_pathaks = []
    force_hf_warp_approxs = []
    force_pulay_warp_approxs = []

    for wf in warpfacs:
        force_hf_pathak = -data[f"grad el pathak ({wf})"][()][1:]

        psilogderiv_pathak = data[f"grad log psi pathak ({wf})"][()][1:]
        el_times_psilogderiv_pathak = data[f"Local energy * grad log psi pathak ({wf})"][()][1:]

        psilogderiv_warp_approx = data[f"grad log psi (warp) ({wf})"][()][1:]
        el_times_psilogderiv_warp_approx = data[f"Local energy * grad log psi (warp) ({wf})"][()][1:]

        jac_logderiv_approx_wf = data[f"grad log j approx ({wf})"][()][1:]
        el_times_jac_logderiv_approx_wf = data[f"Local energy * grad log j approx ({wf})"][()][1:]

        force_hf_warp_approx = -data[f"grad el (warp) ({wf})"][()][1:]

        force_hf_pathaks.append(force_hf_pathak.flatten())
        force_pulay_pathaks.append(
            -(2*(el_times_psilogderiv_pathak - energy*psilogderiv_pathak)).flatten()
            )
        force_hf_warp_approxs.append(force_hf_warp_approx)
        force_pulay_warp_approxs.append(
            -(2*(el_times_psilogderiv_warp_approx - energy*psilogderiv_warp_approx)
            +   el_times_jac_logderiv_approx_wf - energy*jac_logderiv_approx_wf).flatten()
        )

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

    force_pulay_ = -(
            2 * (el_times_psilogderiv - energy*psilogderiv) \
            )

    force_pulay__warp = -(
            2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )

    force_pulay__warp_approx = -(
            2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
            +   (el_times_jac_logderiv_approx - energy*jac_logderiv_approx) \
            )

    data.close()

    return force_hf.flatten(), \
           force_hf_warp.flatten(), \
           force_hf_warp_approxs, \
           force_hf_pathaks, \
           force_pulay_exact.flatten(), \
           force_pulay_exact_warp.flatten(), \
           force_pulay_.flatten(), \
           force_pulay_pathaks, \
           force_pulay__warp.flatten(), \
           force_pulay__warp_approx.flatten(), \
           force_pulay_warp_approxs, \
           force_pulay_exact_pq.flatten(), \
           force_pulay_exact_warp_pq.flatten(), \
           force_pulay_exact_warp_pq_approx.flatten(), \
           weights.flatten(), warpfacs


force_hf, force_hf_warp, force_hf_warp_approxs, force_hf_pathaks, \
        force_pulay_exact, force_pulay_exact_warp, \
        force_pulay_, \
        force_pulay_pathaks, \
        force_pulay__warp, \
        force_pulay__warp_approx, \
        force_pulay__warp_approxs, \
        force_pulay_exact_pq, force_pulay_exact_warp_pq, force_pulay_exact_warp_pq_approx, \
        weights, warpfacs \
    = compute_forces(sys.argv[1])

fhf = np.average(force_hf, weights=weights)
fhf_err = error(force_hf, weights=weights)

fhf_warp = np.average(force_hf_warp, weights=weights)
fhf_err_warp = error(force_hf_warp, weights=weights)

fhf_pathak = np.average(force_hf_pathaks[0], weights=weights)
fhf_err_pathaks = error(force_hf_pathaks[0], weights=weights)

fpulay_exact = np.average(force_pulay_exact, weights=weights)
fpulay_exact_err = error(force_pulay_exact, weights=weights)

fpulay_  = np.average(force_pulay_, weights=weights)
fpulay__err = error(force_pulay_, weights=weights)

fpulay_pathak  = np.average(force_pulay_pathaks[0], weights=weights)
fpulay_pathak_err  = error(force_pulay_pathaks[0], weights=weights)

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

ftot_pathak = np.average(force_hf_pathaks[0] + force_pulay_pathaks[0], weights=weights)
ftot_pathak_err = error(force_hf_pathaks[0] + force_pulay_pathaks[0], weights=weights)

ftot_exact_warp = np.average(force_hf_warp + force_pulay_exact_warp, weights=weights)
ftot_exact_err_warp = error(force_hf_warp + force_pulay_exact_warp, weights=weights)

ftot__warp = np.average(force_hf_warp + force_pulay__warp, weights=weights)
ftot__err_warp = error(force_hf_warp + force_pulay__warp, weights=weights)

ftot__warp_approx = np.average(force_hf_warp + force_pulay__warp_approx, weights=weights)
ftot__err_warp_approx = error(force_hf_warp + force_pulay__warp_approx, weights=weights)

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
print(f"Pulay force pathak:                           {fpulay_pathak:.5f} +/- {fpulay_pathak_err:.5f}")
print(f"Pulay force (warp):                           {fpulay__warp:.5f} +/- {fpulay__err_warp:.5f}")
print(f"\n")
print(f"Total force (Green's):                        {ftot_exact:.5f} +/- {ftot_exact_err:.5f}")
print(f"Total force (Green's, warp):                  {ftot_exact_warp:.5f} +/- {ftot_exact_err_warp:.5f}")
print(f"Total force (Green's, p/q):                   {ftot_exact_pq:.5f} +/- {ftot_exact_pq_err:.5f}")
print(f"Total force (Green's, warp, p/q):             {ftot_exact_pq_warp:.5f} +/- {ftot_exact_pq_err_warp:.5f}")
print(f"Total force (Green's, warp, p/q, approx J):   {ftot_exact_pq_warp_approx:.5f} +/- {ftot_exact_pq_err_warp_approx:.5f}")
print(f"Total force:                                  {ftot_:.5f} +/- {ftot__err:.5f}")
print(f"Total force pathak:                           {ftot_pathak:.5f} +/- {ftot_pathak_err:.5f}")
print(f"Total force (warp):                           {ftot__warp:.5f} +/- {ftot__err_warp:.5f}")
print(f"Total force (warp, approx.):                  {ftot__warp_approx:.5f} +/- {ftot__err_warp_approx:.5f}")
print(f"Exact force:                                  {3.43196:.5f}")


plot_forces_over_time(
    (force_hf, force_pulay_), 
    #(force_hf_pathak, force_pulay_pathak), 
    (force_hf_warp, force_pulay__warp), 
    #(force_hf_warp, force_pulay__warp_approx), 
    #(force_hf, force_pulay_exact), 
    #(force_hf_warp, force_pulay_exact_warp), 
    #(force_hf, force_pulay_exact_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=[
        "Not warped",  
        #"Pathak", 
        "Warped", 
        #"Warped, approx. J", 
        #"Projector, naive",
        #"Projector, naive, warp",
        #"Projector", 
        #"Projector, warped", 
        #"Projector, approx. J"
    ], 
    weights=weights
)
plt.tight_layout()

plot_errors_over_time(
    (force_hf, force_pulay_), 
    #(force_hf_pathak, force_pulay_pathak), 
    (force_hf_warp, force_pulay__warp), 
    #(force_hf_warp, force_pulay__warp_approx), 
    #(force_hf, force_pulay_exact), 
    #(force_hf_warp, force_pulay_exact_warp), 
    #(force_hf, force_pulay_exact_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq), 
    #(force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=[
        "Not warped", 
        #"Pathak", 
        "Warped", 
        #"Warped, approx. J", 
        #"Projector, naive",
        #"Projector, naive, warp",
        #"Projector", 
        #"Projector, warped", 
        #"Projector, approx. J"
    ], 
    weights=weights
)
plt.tight_layout()


plot_force_data_trace(force_hf, force_pulay_, force_hf_warp, force_pulay__warp)
plt.tight_layout()

fs_pathak = np.array([np.average(fhf + fp, weights=weights) for (fhf, fp) in zip(force_hf_pathaks, force_pulay_pathaks)])
errs_pathak = np.array([error(fhf + fp, weights=weights) for (fhf, fp) in zip(force_hf_pathaks, force_pulay_pathaks)])

fs_approx = np.array([np.average(fhf + fp, weights=weights) for (fhf, fp) in zip(force_hf_warp_approxs, force_pulay__warp_approxs)])
errs_approx = np.array([error(fhf + fp, weights=weights) for (fhf, fp) in zip(force_hf_warp_approxs, force_pulay__warp_approxs)])


plt.figure()
plt.fill_between(warpfacs, fs_pathak - errs_pathak, fs_pathak + errs_pathak, alpha=0.2)
plt.plot(warpfacs, fs_pathak, marker="o", label="Pathak")

plt.fill_between(warpfacs, fs_approx - errs_approx, fs_approx + errs_approx, alpha=0.2)
plt.plot(warpfacs, fs_approx, marker="o", label="Warped force, approx. J")

from scipy.optimize import curve_fit
lin = lambda x, a, b: a*x + b
plt.plot([0, max(warpfacs)], [3.43196]*2, "black", linestyle="--")
#plt.plot([0, max(warpfacs)], [2.5]*2, "black")
plt.xlim(0)
plt.xlabel("Cutoff scale")
plt.ylabel("Force estimate")
plt.grid()
plt.legend()

#plt.gcf().subplots_adjust(right=0.15)
plt.tight_layout()

plt.show()
