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


def plot_errors_over_time(*forces, labels=[], weights=None, npoints=20):
    # forces should be tuples of HF and Pulay forces

    if weights is None:
        weights = np.ones(forces[0][0].shape)

    ns = np.linspace(1, len(forces[0][0]), npoints)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False)

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

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig, axes


def plot_forces_over_time(*forces, labels=[], weights=None, npoints=20):
    # forces should be tuples of HF and Pulay forces

    if weights is None:
        weights = np.ones(forces[0][0].shape)

    ns = np.linspace(1, len(forces[0][0]), npoints)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=False)

    for i, (fhf, fp) in enumerate(forces):

        hf_means, hf_errs = error_over_time(fhf, npoints, weights=weights)
        pulay_means, pulay_errs = error_over_time(fp, npoints, weights=weights)
        total_means, total_errs = error_over_time(fhf + fp, npoints, weights=weights)

        axes[0].errorbar(ns, hf_means, yerr=hf_errs, marker='o', label=labels[i])
        axes[0].set_title("Hellmann-Feynman term")
        axes[1].errorbar(ns, pulay_means, yerr=pulay_errs, marker='o', label=labels[i])
        axes[1].set_title("Pulay term")
        axes[2].errorbar(ns, total_means, yerr=total_errs, marker='o', label=labels[i])
        axes[2].set_title("Total force")

    axes[2].plot(ns, [3.45]*len(ns), label="PES", color="black")

    axes[0].grid()
    axes[1].grid()
    axes[2].grid()

    axes[2].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

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
    local_e_deriv_pathak0 = data["grad el pathak (1e-2)"][()][1:]
    local_e_deriv_pathak1 = data["grad el pathak (0.5e-1)"][()][1:]
    local_e_deriv_pathak2 = data["grad el pathak (1e-1)"][()][1:]
    local_e_deriv_pathak3 = data["grad el pathak (1.5e-1)"][()][1:]

    # Get psi derivative
    psilogderiv = data["grad log psi"][()][1:]
    psilogderiv_pathak0 = data["grad log psi pathak (1e-2)"][()][1:]
    psilogderiv_pathak1 = data["grad log psi pathak (0.5e-1)"][()][1:]
    psilogderiv_pathak2 = data["grad log psi pathak (1e-1)"][()][1:]
    psilogderiv_pathak3 = data["grad log psi pathak (1.5e-1)"][()][1:]
    psilogderiv_warp = data["grad log psi (warp)"][()][1:]

    el_times_psilogderiv = data["Local energy * grad log psi"][()][1:]
    el_times_psilogderiv_pathak0 = data["Local energy * grad log psi pathak (1e-2)"][()][1:]
    el_times_psilogderiv_pathak1 = data["Local energy * grad log psi pathak (0.5e-1)"][()][1:]
    el_times_psilogderiv_pathak2 = data["Local energy * grad log psi pathak (1e-1)"][()][1:]
    el_times_psilogderiv_pathak3 = data["Local energy * grad log psi pathak (1.5e-1)"][()][1:]
    el_times_psilogderiv_warp = data["Local energy * grad log psi (warp)"][()][1:]

    # Hellmann-Feynman force
    force_hf = -local_e_deriv
    force_hf_pathak0 = -local_e_deriv_pathak0
    force_hf_pathak1 = -local_e_deriv_pathak1
    force_hf_pathak2 = -local_e_deriv_pathak2
    force_hf_pathak3 = -local_e_deriv_pathak3
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

    force_pulay_pathak0 = -(
            2 * (el_times_psilogderiv_pathak0 - energy*psilogderiv_pathak0) \
            )
    force_pulay_pathak1 = -(
            2 * (el_times_psilogderiv_pathak1 - energy*psilogderiv_pathak1) \
            )
    force_pulay_pathak2 = -(
            2 * (el_times_psilogderiv_pathak2 - energy*psilogderiv_pathak2) \
            )
    force_pulay_pathak3 = -(
            2 * (el_times_psilogderiv_pathak3 - energy*psilogderiv_pathak3) \
            )

    force_pulay__warp = -(
            2 * (el_times_psilogderiv_warp - energy*psilogderiv_warp) \
            +   (el_times_jac_logderiv - energy*jac_logderiv) \
            )

    data.close()

    return force_hf.flatten(), \
           force_hf_warp.flatten(), \
           force_hf_pathak0.flatten(), \
           force_hf_pathak1.flatten(), \
           force_hf_pathak2.flatten(), \
           force_hf_pathak3.flatten(), \
           force_pulay_exact.flatten(), \
           force_pulay_exact_warp.flatten(), \
           force_pulay_.flatten(), \
           force_pulay_pathak0.flatten(), \
           force_pulay_pathak1.flatten(), \
           force_pulay_pathak2.flatten(), \
           force_pulay_pathak3.flatten(), \
           force_pulay__warp.flatten(), \
           force_pulay_exact_pq.flatten(), \
           force_pulay_exact_warp_pq.flatten(), \
           force_pulay_exact_warp_pq_approx.flatten(), \
           weights.flatten()


force_hf, force_hf_warp, \
    force_hf_pathak0, \
    force_hf_pathak1, \
    force_hf_pathak2, \
    force_hf_pathak3, \
        force_pulay_exact, force_pulay_exact_warp, \
        force_pulay_, \
        force_pulay_pathak0, \
        force_pulay_pathak1, \
        force_pulay_pathak2, \
        force_pulay_pathak3, \
        force_pulay__warp, \
        force_pulay_exact_pq, force_pulay_exact_warp_pq, force_pulay_exact_warp_pq_approx, \
        weights \
    = compute_forces(sys.argv[1])

fhf = np.average(force_hf, weights=weights)
fhf_err = error(force_hf, weights=weights)

fhf_pathak0 = np.average(force_hf_pathak0, weights=weights)
fhf_pathak0_err = error(force_hf_pathak0, weights=weights)
fhf_pathak1 = np.average(force_hf_pathak1, weights=weights)
fhf_pathak1_err = error(force_hf_pathak1, weights=weights)
fhf_pathak2 = np.average(force_hf_pathak2, weights=weights)
fhf_pathak2_err = error(force_hf_pathak2, weights=weights)
fhf_pathak3 = np.average(force_hf_pathak3, weights=weights)
fhf_pathak3_err = error(force_hf_pathak3, weights=weights)

fhf_warp = np.average(force_hf_warp, weights=weights)
fhf_err_warp = error(force_hf_warp, weights=weights)

fpulay_exact = np.average(force_pulay_exact, weights=weights)
fpulay_exact_err = error(force_pulay_exact, weights=weights)

fpulay_  = np.average(force_pulay_, weights=weights)
fpulay__err = error(force_pulay_, weights=weights)

fpulay_pathak0  = np.average(force_pulay_pathak0, weights=weights)
fpulay_pathak0_err = error(force_pulay_pathak0, weights=weights)
fpulay_pathak1  = np.average(force_pulay_pathak1, weights=weights)
fpulay_pathak1_err = error(force_pulay_pathak1, weights=weights)
fpulay_pathak2  = np.average(force_pulay_pathak2, weights=weights)
fpulay_pathak2_err = error(force_pulay_pathak2, weights=weights)
fpulay_pathak3  = np.average(force_pulay_pathak3, weights=weights)
fpulay_pathak3_err = error(force_pulay_pathak3, weights=weights)

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

ftot_pathak0 = np.average(force_hf_pathak0 + force_pulay_pathak0, weights=weights)
ftot_pathak_err0 = error(force_hf_pathak0 + force_pulay_pathak0, weights=weights)
ftot_pathak1 = np.average(force_hf_pathak1 + force_pulay_pathak1, weights=weights)
ftot_pathak_err1 = error(force_hf_pathak1 + force_pulay_pathak1, weights=weights)
ftot_pathak2 = np.average(force_hf_pathak2 + force_pulay_pathak2, weights=weights)
ftot_pathak_err2 = error(force_hf_pathak2 + force_pulay_pathak2, weights=weights)
ftot_pathak3 = np.average(force_hf_pathak3 + force_pulay_pathak3, weights=weights)
ftot_pathak_err3 = error(force_hf_pathak3 + force_pulay_pathak3, weights=weights)

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
print(f"HF force pathak:                              {fhf_pathak0:.5f} +/- {fhf_pathak0_err:.5f}")
print(f"\n")
print(f"Pulay force (Green's):                        {fpulay_exact:.5f} +/- {fpulay_exact_err:.5f}")
print(f"Pulay force (Green's, warp):                  {fpulay_exact_warp:.5f} +/- {fpulay_exact_err_warp:.5f}")
print(f"\n")
print(f"Pulay force (Green's, p/q):                   {fpulay_exact_pq:.5f} +/- {fpulay_exact_pq_err:.5f}")
print(f"Pulay force (Green's, warp, p/q):             {fpulay_exact_pq_warp:.5f} +/- {fpulay_exact_pq_err_warp:.5f}")
print(f"\n")
print(f"Pulay force:                                  {fpulay_:.5f} +/- {fpulay__err:.5f}")
print(f"Pulay force pathak:                           {fpulay_pathak0:.5f} +/- {fpulay_pathak0_err:.5f}")
print(f"Pulay force (warp):                           {fpulay__warp:.5f} +/- {fpulay__err_warp:.5f}")
print(f"\n")
print(f"Total force (Green's):                        {ftot_exact:.5f} +/- {ftot_exact_err:.5f}")
print(f"Total force (Green's, warp):                  {ftot_exact_warp:.5f} +/- {ftot_exact_err_warp:.5f}")
print(f"Total force (Green's, p/q):                   {ftot_exact_pq:.5f} +/- {ftot_exact_pq_err:.5f}")
print(f"Total force (Green's, warp, p/q):             {ftot_exact_pq_warp:.5f} +/- {ftot_exact_pq_err_warp:.5f}")
print(f"Total force (Green's, warp, p/q, approx J):   {ftot_exact_pq_warp_approx:.5f} +/- {ftot_exact_pq_err_warp_approx:.5f}")
print(f"Total force:                                  {ftot_:.5f} +/- {ftot__err:.5f}")
print(f"Total force pathak:                           {ftot_pathak0:.5f} +/- {ftot_pathak_err0:.5f}")
print(f"Total force (warp):                           {ftot__warp:.5f} +/- {ftot__err_warp:.5f}")


plot_forces_over_time(
    (force_hf, force_pulay_), 
    (force_hf_pathak0, force_pulay_pathak0), 
    (force_hf_warp, force_pulay__warp), 
    (force_hf, force_pulay_exact_pq), 
    (force_hf_warp, force_pulay_exact_warp_pq), 
    (force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=["Not warped",  "Pathak", "Warped", "Projector", "Projector, warped", "Projector, approx. J"], 
    weights=weights
)

plot_errors_over_time(
    (force_hf, force_pulay_), 
    (force_hf_pathak0, force_pulay_pathak0), 
    (force_hf_warp, force_pulay__warp), 
    (force_hf, force_pulay_exact_pq), 
    (force_hf_warp, force_pulay_exact_warp_pq), 
    (force_hf_warp, force_pulay_exact_warp_pq_approx), 
    labels=["Not warped", "Pathak", "Warped", "Projector", "Projector, warped", "Projector, approx. J"], 
    weights=weights
)

epss = [1e-2, 0.5e-1, 1e-1, 1.5e-1]

plot_force_data_trace(force_hf, force_pulay_exact_pq, force_hf_warp, force_pulay_exact_warp_pq)

fs_pathak = [ftot_pathak0, ftot_pathak1, ftot_pathak2, ftot_pathak3]
errs_pathak = [ftot_pathak_err0, ftot_pathak_err1, ftot_pathak_err2, ftot_pathak_err3]
plt.figure()
plt.errorbar(epss, fs_pathak, yerr=errs_pathak, marker="o")
plt.plot([0, epss[-1]], [3.45]*2, "black")
plt.xlim(0)
plt.grid()

plt.show()