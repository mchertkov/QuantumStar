"""Uniform-star LL-to-quantum transition experiment.

This script constructs a homogeneous star with uniform couplings

    H = (J0/d) S0 . sum_c Sc,

but with a nontrivial product state over leaves (two coherent-state populations),
so that many total-leaf-spin Schur sectors are populated.  The goals are:

1. show a representative time trace where the L0 surrogate works only briefly,
   L0+G1 extends the quantitative window substantially, and both eventually fail;
2. quantify the validity horizon T_eps(d) defined by the first time the branch-
   continuous log-error exceeds a fixed threshold epsilon;
3. provide a clean figure for brainstorming the transition from an LL-substitutable
   regime to a genuinely quantum / discrete-spectrum regime.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.star_qdp import (  # noqa: E402
    schur_blocks,
    spinL_ops,
    unit,
    A_L0,
    dlog_G1,
    log_unwrapped,
)

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def exact_homog_series(d: int, J0: float, ts, hub_ang, leaf_angles):
    """Exact homogeneous-star amplitude on a whole time grid.

    Compared with repeated calls to ``exact_homog``, this routine precomputes the
    Schur blocks once and then evaluates the exact amplitude vectorized in time.
    """
    lam = J0 / d
    n0v = unit(*hub_ang)
    blocks = schur_blocks(leaf_angles)
    ts = np.asarray(ts, dtype=float)
    A = np.zeros_like(ts, dtype=complex)
    for L, ML in blocks.items():
        xp, xm = L / 2, -(L + 1) / 2
        ep = np.exp(-1j * lam * ts * xp)
        em = np.exp(-1j * lam * ts * xm)
        g = (ep - em) / (xp - xm)
        f = ep - g * xp
        leaf_moment = np.array([np.trace(ML @ op) for op in spinL_ops(L)])
        A += f * np.trace(ML) + 0.5 * g * np.dot(n0v, leaf_moment)
    return A


def build_two_population_leaves(d: int, frac_a: float = 0.70,
                                ang_a=(0.22 * np.pi, 0.10 * np.pi),
                                ang_b=(0.40 * np.pi, 0.80 * np.pi)):
    """Deterministic two-population leaf state for the homogeneous star."""
    n_a = int(round(frac_a * d))
    n_b = d - n_a
    leaf_angles = [ang_a] * n_a + [ang_b] * n_b
    nus = np.array([unit(*ang) for ang in leaf_angles])
    return leaf_angles, nus


def first_crossing(ts, err, eps):
    idx = np.where(err > eps)[0]
    return ts[idx[0]] if len(idx) else ts[-1]


def generate_uniform_star_transition_data(
    J0: float = 1.0,
    hub_ang=(0.92, 0.35),
    rep_d: int = 32,
    ds=(12, 16, 24, 32, 48, 64),
    eps: float = 0.10,
    horizon_factor: float = 2.0,
    rep_nt: int = 81,
    scan_nt: int = 41,
    g1_nt: int = 61,
):
    """Generate representative traces and horizon-scan data."""
    rep_leaf_angles, rep_nus = build_two_population_leaves(rep_d)
    rep_ts = np.linspace(0.0, horizon_factor * rep_d, rep_nt)
    rep_exact = exact_homog_series(rep_d, J0, rep_ts, hub_ang, rep_leaf_angles)
    rep_l0 = A_L0(J0, rep_d, rep_ts, hub_ang, rep_nus)
    rep_dg1 = np.array([
        0.0j if T == 0 else dlog_G1(J0, rep_d, T, hub_ang, rep_nus, nt=g1_nt)
        for T in rep_ts
    ])
    rep_g1 = rep_l0 * np.exp(rep_dg1)

    rep_log_exact = log_unwrapped(rep_exact)
    rep_log_l0 = log_unwrapped(rep_l0)
    rep_log_g1 = log_unwrapped(rep_g1)
    rep_err_l0 = np.abs(rep_log_exact - rep_log_l0)
    rep_err_g1 = np.abs(rep_log_exact - rep_log_g1)
    rep_h_l0 = first_crossing(rep_ts, rep_err_l0, eps)
    rep_h_g1 = first_crossing(rep_ts, rep_err_g1, eps)

    horizon_l0 = []
    horizon_g1 = []
    for d in ds:
        leaf_angles, nus = build_two_population_leaves(d)
        ts = np.linspace(0.0, horizon_factor * d, scan_nt)
        exact = exact_homog_series(d, J0, ts, hub_ang, leaf_angles)
        l0 = A_L0(J0, d, ts, hub_ang, nus)
        dg1 = np.array([
            0.0j if T == 0 else dlog_G1(J0, d, T, hub_ang, nus, nt=g1_nt)
            for T in ts
        ])
        g1 = l0 * np.exp(dg1)
        log_exact = log_unwrapped(exact)
        err_l0 = np.abs(log_exact - log_unwrapped(l0))
        err_g1 = np.abs(log_exact - log_unwrapped(g1))
        horizon_l0.append(first_crossing(ts, err_l0, eps))
        horizon_g1.append(first_crossing(ts, err_g1, eps))

    data = {
        'J0': J0,
        'hub_ang': np.asarray(hub_ang),
        'rep_d': rep_d,
        'rep_ts': rep_ts,
        'rep_exact': rep_exact,
        'rep_l0': rep_l0,
        'rep_g1': rep_g1,
        'rep_err_l0': rep_err_l0,
        'rep_err_g1': rep_err_g1,
        'rep_h_l0': rep_h_l0,
        'rep_h_g1': rep_h_g1,
        'eps': eps,
        'ds': np.asarray(ds),
        'horizon_l0': np.asarray(horizon_l0),
        'horizon_g1': np.asarray(horizon_g1),
        'horizon_factor': horizon_factor,
        'scan_nt': scan_nt,
        'rep_leaf_angles': np.asarray(rep_leaf_angles),
    }
    return data


def plot_uniform_star_transition(data, save=True, basename="fig_uniform_star_LL_to_quantum"):
    """Create the brainstorming figure for the LL-to-quantum transition."""
    rep_ts = data['rep_ts']
    rep_exact = data['rep_exact']
    rep_l0 = data['rep_l0']
    rep_g1 = data['rep_g1']
    rep_err_l0 = data['rep_err_l0']
    rep_err_g1 = data['rep_err_g1']
    rep_h_l0 = data['rep_h_l0']
    rep_h_g1 = data['rep_h_g1']
    eps = float(data['eps'])
    ds = data['ds']
    horizon_l0 = data['horizon_l0']
    horizon_g1 = data['horizon_g1']
    rep_d = int(data['rep_d'])

    fig = plt.figure(figsize=(8.2, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # (a) Representative amplitudes.
    ax0.plot(rep_ts, np.abs(rep_exact), label="exact", linewidth=2.0)
    ax0.plot(rep_ts, np.abs(rep_l0), label="L0", linestyle="--")
    ax0.plot(rep_ts, np.abs(rep_g1), label="L0+G1", linestyle="-.")
    ax0.axvline(rep_h_g1, linestyle=":", linewidth=1.2)
    ax0.text(rep_h_g1 * 1.01, 0.12, r"$T_{\epsilon}^{\mathrm{G1}}$", rotation=90,
             va="bottom", fontsize=10)
    ax0.set_ylabel(r"$|\mathcal{A}(T)|$")
    ax0.set_title(
        f"Uniform star with two leaf populations: representative transition at d={rep_d}"
    )
    ax0.set_ylim(0.0, 1.04)
    ax0.grid(True, alpha=0.25)
    ax0.legend(frameon=False, ncol=3)

    # (b) Representative errors.
    ax1.semilogy(rep_ts, rep_err_l0, label="L0")
    ax1.semilogy(rep_ts, rep_err_g1, label="L0+G1")
    ax1.axhline(eps, linestyle=":", linewidth=1.1)
    ax1.axvline(rep_h_l0, linestyle=":", linewidth=1.0)
    ax1.axvline(rep_h_g1, linestyle=":", linewidth=1.0)
    ax1.set_xlabel("time T")
    ax1.set_ylabel("branch log-error")
    ax1.set_title(rf"Error growth; threshold $\epsilon={eps:.2f}$")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=False)

    # (c) Validity horizons.
    ax2.plot(ds, horizon_l0, marker="o", label="L0")
    ax2.plot(ds, horizon_g1, marker="s", label="L0+G1")
    ax2.plot(ds, ds, linestyle=":", linewidth=1.2, label=r"$T\sim d/J$")
    ax2.set_xlabel("degree d")
    ax2.set_ylabel(r"$T_{\epsilon}(d)$")
    ax2.set_title("Validity horizon versus degree")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False)

    fig.tight_layout()
    if save:
        fig.savefig(OUT / f"{basename}.png", dpi=300)
        fig.savefig(OUT / f"{basename}.pdf")
        np.savez(OUT / f"{basename}_data.npz", **data)
    return fig


if __name__ == "__main__":
    data = generate_uniform_star_transition_data()
    fig = plot_uniform_star_transition(data)
    plt.close(fig)
