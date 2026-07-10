"""Driven-star validation experiment.

This script tests the fully time-dependent anisotropic star implementation.
It generates smooth drives, computes an exact sparse/Krylov oracle for a small
star, and compares branch-continuous log-amplitude errors for L0 and L0+G1.
It also computes a small final-time scaling scan.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_qdp import (  # noqa: E402
    make_driven_star_instance,
    exact_sparse_driven,
    L0G1_driven,
    log_unwrapped,
    fit_slope,
)

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def driven_time_trace():
    T = 2.0
    nt = 81
    d = 6
    seed = 3
    ts, B0t, bct, Kct, hub_ang, leaf_angles = make_driven_star_instance(
        d=d, nt=nt, T=T, seed=seed, coupling_scale=1.0, anisotropy=0.2
    )
    exact = exact_sparse_driven(Kct, B0t, bct, ts, hub_ang, leaf_angles)

    l0_amp = []
    g1_amp = []
    for m in range(2, nt + 1):
        logA, dG = L0G1_driven(
            Kct[:, :m], B0t[:m], bct[:, :m], ts[m - 1], hub_ang, leaf_angles, nt=m
        )
        l0_amp.append(np.exp(logA))
        g1_amp.append(np.exp(logA + dG))

    log_exact = log_unwrapped(exact[1:])
    log_l0 = log_unwrapped(np.asarray(l0_amp))
    log_g1 = log_unwrapped(np.asarray(g1_amp))
    err_l0 = np.abs(log_exact - log_l0)
    err_g1 = np.abs(log_exact - log_g1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(ts[1:], err_l0, label="L0", linestyle="-")
    ax.semilogy(ts[1:], err_g1, label="L0+G1", linestyle="--")
    ax.set_xlabel(r"time $T$")
    ax.set_ylabel(r"branch-continuous log-error")
    ax.set_title(r"Driven anisotropic star, $d=6$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_driven_validation_error.png", dpi=300)
    fig.savefig(OUT / "fig_driven_validation_error.pdf")
    plt.close(fig)

    np.savez(
        OUT / "fig_driven_validation_error_data.npz",
        ts=ts,
        exact=exact,
        err_l0=err_l0,
        err_g1=err_g1,
        B0t=B0t,
        bct=bct,
        Kct=Kct,
    )
    return ts, err_l0, err_g1


def driven_final_scaling():
    T = 2.0
    nt = 81
    ds = np.array([4, 6, 8, 10])
    err_l0 = []
    err_g1 = []
    for d in ds:
        ts, B0t, bct, Kct, hub_ang, leaf_angles = make_driven_star_instance(
            d=d, nt=nt, T=T, seed=4, coupling_scale=1.0, anisotropy=0.2
        )
        exact = exact_sparse_driven(Kct, B0t, bct, ts, hub_ang, leaf_angles)[-1]
        logA, dG = L0G1_driven(Kct, B0t, bct, T, hub_ang, leaf_angles, nt=nt)
        log_exact = np.log(np.abs(exact)) + 1j * np.angle(exact)
        err_l0.append(abs(log_exact - logA))
        err_g1.append(abs(log_exact - (logA + dG)))
    err_l0 = np.asarray(err_l0)
    err_g1 = np.asarray(err_g1)
    slope_l0 = fit_slope(ds, err_l0)
    slope_g1 = fit_slope(ds, err_g1)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(ds, err_l0, marker="o", label=f"L0, slope {slope_l0:.2f}", linestyle="-")
    ax.loglog(ds, err_g1, marker="s", label=f"L0+G1, slope {slope_g1:.2f}", linestyle="--")
    ax.set_xlabel(r"degree $d$")
    ax.set_ylabel(r"final-time log-error")
    ax.set_title(r"Driven-star final-time scaling, $T=2$")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_driven_validation_scaling.png", dpi=300)
    fig.savefig(OUT / "fig_driven_validation_scaling.pdf")
    plt.close(fig)

    np.savez(
        OUT / "fig_driven_validation_scaling_data.npz",
        ds=ds,
        err_l0=err_l0,
        err_g1=err_g1,
        slope_l0=slope_l0,
        slope_g1=slope_g1,
    )
    return ds, err_l0, err_g1, slope_l0, slope_g1


if __name__ == "__main__":
    ts, err_l0, err_g1 = driven_time_trace()
    ds, e0, e1, s0, s1 = driven_final_scaling()
    print("time-trace final errors:", err_l0[-1], err_g1[-1])
    print("scaling d:", ds)
    print("scaling L0:", e0, "slope", s0)
    print("scaling L0+G1:", e1, "slope", s1)
