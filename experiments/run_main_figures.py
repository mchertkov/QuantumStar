"""Generate main figures for the quantum-star paper rewrite.

The script creates the five figures that were previously placeholders:
1. star primitive / tree-message schematic;
2. L0 + G1 hierarchy schematic;
3. aligned homogeneous benchmark and revival;
4. headline 1/d scaling;
5. static inhomogeneous validation.

Figure 6 is generated separately by run_driven_star_validation.py.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Arc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_qdp import (  # noqa: E402
    aligned_two_sector,
    A_L0,
    dlog_G1_series,
    dlog_G1,
    unit,
    exact_sparse_driven,
    L0G1_inhom,
    log_unwrapped,
    fit_slope,
)

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def fig_star_primitive():
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.set_aspect("equal")
    ax.axis("off")

    hub = np.array([-0.25, 0.0])
    ax.add_patch(Circle(hub, 0.17, fill=False, linewidth=2.0))
    ax.text(hub[0], hub[1], "$S_0$", ha="center", va="center", fontsize=15)
    ax.text(hub[0] + 0.55, 0.82, r"hub drive $\mathbf{B}_0(t)$", ha="center", fontsize=11)

    angles = np.linspace(0.70 * np.pi, 1.30 * np.pi, 6)
    R = 1.35
    for k, th in enumerate(angles, start=1):
        p = hub + np.array([R * np.cos(th), R * np.sin(th)])
        ax.add_patch(Circle(p, 0.12, fill=False, linewidth=1.6))
        ax.text(p[0], p[1], f"$c_{k}$", ha="center", va="center", fontsize=10)
        direction = (p - hub) / np.linalg.norm(p - hub)
        ax.add_patch(FancyArrowPatch(posA=tuple(p - 0.14 * direction), posB=tuple(hub + 0.20 * direction),
                                     arrowstyle="-", linewidth=1.2))
        midp = 0.56 * p + 0.44 * hub
        if k in (2, 4, 6):
            ax.text(midp[0], midp[1], "$K_c(t)$", fontsize=8, rotation=np.degrees(th) - 180,
                    ha="center", va="center")
    ax.text(-2.30, -0.05, r"leaf drives $\mathbf{b}_c(t)$", rotation=90, ha="center", va="center", fontsize=11)

    box_x, box_y, box_w, box_h = 0.75, -0.55, 1.95, 1.10
    box = FancyBboxPatch((box_x, box_y), box_w, box_h, boxstyle="round,pad=0.05", fill=False, linewidth=1.5)
    ax.add_patch(box)
    ax.text(box_x + box_w/2, 0.31, "leaf / subtree", ha="center", fontsize=11)
    ax.text(box_x + box_w/2, 0.05, "influence message", ha="center", fontsize=11)
    ax.text(box_x + box_w/2, -0.24, r"$(\ell,\,\mu(t),\,\kappa(t,t'))$", ha="center", fontsize=11)
    ax.add_patch(FancyArrowPatch(posA=(hub[0] + 0.23, 0.0), posB=(box_x - 0.05, 0.0),
                                 arrowstyle="->", mutation_scale=13, linewidth=1.4))

    ax.add_patch(FancyArrowPatch(posA=(box_x + box_w + 0.05, 0.0), posB=(3.15, 0.0),
                                 arrowstyle="->", mutation_scale=13, linewidth=1.4))
    ax.text(3.45, 0.38, "tree update", ha="center", fontsize=11)
    ax.add_patch(Circle((3.25, 0.0), 0.13, fill=False, linewidth=1.4))
    for yy in [-0.38, 0.0, 0.38]:
        ax.add_patch(Circle((3.95, yy), 0.10, fill=False, linewidth=1.2))
        ax.add_patch(FancyArrowPatch(posA=(3.85, yy), posB=(3.37, 0.02 * yy), arrowstyle="-", linewidth=1.0))
    ax.set_xlim(-2.45, 4.25)
    ax.set_ylim(-1.28, 1.28)
    ax.set_title("Driven star as an influence primitive", fontsize=13)
    _save(fig, "fig_star_primitive")

def fig_theory_schematic():
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.axis("off")

    left = FancyBboxPatch((0.05, 0.18), 0.39, 0.64, boxstyle="round,pad=0.04", fill=False, linewidth=1.6,
                          transform=ax.transAxes)
    right = FancyBboxPatch((0.56, 0.18), 0.39, 0.64, boxstyle="round,pad=0.04", fill=False, linewidth=1.6,
                           transform=ax.transAxes)
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.245, 0.72, "L0: weak mean field", ha="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.245, 0.58, r"$\mathbf{B}_{\mathrm{eff}}(t)=\mathbf{B}_0(t)$", ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.245, 0.51, r"$+\sum_c K_c(t)\,\mu_c(t)$", ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.245, 0.39, "one driven hub spin", ha="center", fontsize=12, transform=ax.transAxes)
    ax.add_patch(Arc((0.245, 0.31), 0.11, 0.11, theta1=30, theta2=330, linewidth=1.2, transform=ax.transAxes))
    ax.add_patch(Circle((0.245, 0.31), 0.025, fill=False, transform=ax.transAxes))

    ax.text(0.755, 0.72, "G1: Gaussian influence", ha="center", fontsize=13, transform=ax.transAxes)
    ax.text(0.755, 0.61, "sum over leaves", ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.755, 0.54, "double-time kernel", ha="center", fontsize=11, transform=ax.transAxes)
    ax.text(0.755, 0.47, r"$\kappa_c(t,t')$ paired with $G_0(t,t')$", ha="center", fontsize=10.5, transform=ax.transAxes)
    ax.text(0.755, 0.39, "nonlocal in time", ha="center", fontsize=12, transform=ax.transAxes)
    ax.plot([0.68, 0.89], [0.24, 0.24], transform=ax.transAxes, linewidth=1.1)
    ax.plot([0.68, 0.68], [0.24, 0.34], transform=ax.transAxes, linewidth=1.1)
    ax.text(0.89, 0.21, "t", transform=ax.transAxes, fontsize=11)
    ax.text(0.66, 0.35, "t'", transform=ax.transAxes, fontsize=11)
    x = np.linspace(0.69, 0.88, 80)
    y = 0.25 + 0.075 * np.exp(-((x - 0.78) / 0.045) ** 2)
    ax.plot(x, y, transform=ax.transAxes, linewidth=1.3)
    ax.add_patch(FancyArrowPatch(posA=(0.45, 0.50), posB=(0.55, 0.50), arrowstyle="->", mutation_scale=14,
                                 linewidth=1.4, transform=ax.transAxes))
    ax.text(0.50, 0.43, "O(1/d) correction", ha="center", fontsize=11, transform=ax.transAxes)
    ax.set_title("1/d hierarchy for the driven star", fontsize=14)
    _save(fig, "fig_theory_schematic")

def fig_aligned_benchmark():
    J0 = 1.0
    d = 24
    hub_ang = (1.0, 0.30)
    leaf_ang = (0.0, 0.0)
    Trev = 4 * np.pi * d / (J0 * (d + 1))
    ts = np.linspace(0.0, 1.25 * Trev, 105)
    nus = np.tile(unit(*leaf_ang), (d, 1))
    A_exact = aligned_two_sector(d, J0, ts, hub_ang, leaf_ang)
    A_l0 = A_L0(J0, d, ts, hub_ang, nus)
    dG = dlog_G1_series(J0, d, ts, hub_ang, nus, nt=101)
    A_g1 = A_l0 * np.exp(dG)

    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    ax.plot(ts, np.abs(A_exact), label="exact", linewidth=2.0)
    ax.plot(ts, np.abs(A_l0), label="L0", linestyle="--")
    ax.plot(ts, np.abs(A_g1), label="L0+G1", linestyle="-.")
    ax.axvline(Trev, linestyle=":", linewidth=1.2)
    ax.text(Trev * 1.01, 0.18, "$T_{rev}$", rotation=90, va="bottom", fontsize=10)
    ax.set_xlabel("time T")
    ax.set_ylabel("return magnitude |A(T)|")
    ax.set_title(f"Aligned homogeneous star, d={d}")
    ax.set_ylim(0.0, 1.04)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    _save(fig, "fig_aligned_benchmark")
    np.savez(OUT / "fig_aligned_benchmark_data.npz", ts=ts, exact=A_exact, A_l0=A_l0, A_g1=A_g1, Trev=Trev, d=d)


def fig_one_over_d_scaling():
    J0 = 1.0
    T = 3.0
    hub_ang = (1.0, 0.30)
    leaf_ang = (0.0, 0.0)
    ds = np.array([8, 12, 16, 24, 32, 48, 64, 96])
    e0, e1 = [], []
    for d in ds:
        nus = np.tile(unit(*leaf_ang), (d, 1))
        A_exact = aligned_two_sector(d, J0, np.array([T]), hub_ang, leaf_ang)[0]
        A_l0 = A_L0(J0, d, np.array([T]), hub_ang, nus)[0]
        dG = dlog_G1(J0, d, T, hub_ang, nus, nt=121)
        log_exact = np.log(A_exact)
        log_l0 = np.log(A_l0)
        e0.append(abs(log_exact - log_l0))
        e1.append(abs(log_exact - log_l0 - dG))
    e0 = np.asarray(e0)
    e1 = np.asarray(e1)
    s0 = fit_slope(ds, e0)
    s1 = fit_slope(ds, e1)

    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    ax.loglog(ds, e0, marker="o", label=f"L0, slope {s0:.2f}", linestyle="-")
    ax.loglog(ds, e1, marker="s", label=f"L0+G1, slope {s1:.2f}", linestyle="--")
    xref = np.array([ds[0], ds[-1]])
    ax.loglog(xref, e0[-1] * (xref / ds[-1]) ** (-1), linestyle=":", linewidth=1.2, label=r"$d^{-1}$")
    ax.loglog(xref, e1[-1] * (xref / ds[-1]) ** (-2), linestyle=":", linewidth=1.2, label=r"$d^{-2}$")
    ax.set_xlabel("degree d")
    ax.set_ylabel("final-time log-error")
    ax.set_title(f"Headline scaling at T={T}")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, "fig_one_over_d_scaling")
    np.savez(OUT / "fig_one_over_d_scaling_data.npz", ds=ds, err_l0=e0, err_g1=e1, slope_l0=s0, slope_g1=s1)


def fig_static_inhomogeneous_validation():
    rng = np.random.default_rng(12)
    d = 7
    T = 3.0
    nt = 61
    ts = np.linspace(0, T, nt)
    J0 = 1.0
    Jc = (J0 / d) * rng.normal(1.0, 0.25, size=d)
    B0 = np.array([0.19, -0.11, 0.27])
    bcs = rng.normal(0, 0.18, size=(d, 3)) + np.array([0.04, -0.02, 0.01])
    hub_ang = (0.87, 0.23)
    leaf_angles = []
    for _ in range(d):
        theta = np.arccos(rng.uniform(-0.8, 0.8))
        phi = rng.uniform(0, 2 * np.pi)
        leaf_angles.append((theta, phi))
    K = np.array([Jc[c] * np.eye(3) for c in range(d)])
    exact = exact_sparse_driven(K, B0, bcs, ts, hub_ang, leaf_angles)

    l0_amp = []
    g1_amp = []
    for Tm in ts[1:]:
        logA, dG = L0G1_inhom(Jc, B0, bcs, Tm, hub_ang, leaf_angles, nt=61)
        l0_amp.append(np.exp(logA))
        g1_amp.append(np.exp(logA + dG))
    log_exact = log_unwrapped(exact[1:])
    log_l0 = log_unwrapped(np.asarray(l0_amp))
    log_g1 = log_unwrapped(np.asarray(g1_amp))
    err_l0 = np.abs(log_exact - log_l0)
    err_g1 = np.abs(log_exact - log_g1)

    fig, ax = plt.subplots(figsize=(6.5, 4.25))
    ax.semilogy(ts[1:], err_l0, label="L0", linestyle="-")
    ax.semilogy(ts[1:], err_g1, label="L0+G1", linestyle="--")
    ax.set_xlabel("time T")
    ax.set_ylabel("branch-continuous log-error")
    ax.set_title(f"Static inhomogeneous star, d={d}")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    _save(fig, "fig_static_inhom_validation")
    np.savez(OUT / "fig_static_inhom_validation_data.npz", ts=ts, exact=exact, err_l0=err_l0, err_g1=err_g1,
             Jc=Jc, B0=B0, bcs=bcs)


if __name__ == "__main__":
    fig_star_primitive()
    fig_theory_schematic()
    fig_aligned_benchmark()
    fig_one_over_d_scaling()
    fig_static_inhomogeneous_validation()
    print("Generated figures 1--5 in", OUT)
