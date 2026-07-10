r"""star_qdp.py — Return amplitudes of quantum Heisenberg stars: exact oracles and
the L0 (on-shell / mean-field) + G1 (Gaussian, nonlocal-in-time) hierarchy in 1/d.

Object
------
    A_d(T) = <Psi_0| exp(-i H T) |Psi_0>,
    H      = (J0/d) * sum_{c=1..d} S_0 . S_c            (star: hub 0, leaves c),
    |Psi_0> = |n_0> (x) prod_c |nu_c>                    (spin-1/2 coherent states).

Approximations (all continuous in time; time grids below are numerical quadrature
only, never part of the theory):

    L0:  A_L0 = <n_0| exp(-i T Bbar.S) |n_0>,  Bbar = (J0/2d) sum_c nu_c.
    G1:  dlogA = -(1/2) (J0/d)^2 sum_c  \iint_0^T dt dt'
                     kappa_c^{ab}(t,t') G_0^{ab}(t,t'),
         kappa_c^{ab}(t,t') = (1/4)(delta_ab - nu^a nu^b)
                              + (i/4) eps^{abg} nu^g sgn(t-t'),
         G_0^{ab}(t,t') = weak, T-ordered, full two-point function of the hub spin
                          under the L0 drive, with |n_0> at both ends.

Claim (numerically validated):
    |log A - log A_L0|            = O(1/d),
    |log A - log A_L0 - dlogA_G1| = O(1/d^2),
uniformly on time windows short of the aligned-star revival T_rev ~ 4 pi d/(J0 (d+1)).
"""
import numpy as np
from scipy.linalg import expm

# ----------------------------------------------------------------------------- basics
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
S = [sx / 2, sy / 2, sz / 2]

eps = np.zeros((3, 3, 3))
for _a, _b, _c in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    eps[_a, _b, _c] = 1.0
    eps[_a, _c, _b] = -1.0


def coh(theta, phi):
    """Spin-1/2 coherent state |n(theta,phi)> = cos(t/2)|up> + e^{i phi} sin(t/2)|dn>."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)],
                    dtype=complex)


def unit(theta, phi):
    """Unit vector n(theta,phi)."""
    return np.array([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)])


def ang_of(nu):
    """(theta, phi) of a unit vector."""
    return np.arccos(np.clip(nu[2], -1, 1)), np.arctan2(nu[1], nu[0])


# --------------------------------------------------------- collective (aligned) oracle
def spinL_ops(L):
    """Spin-L operators (Sx,Sy,Sz), basis m = L, L-1, ..., -L."""
    dim = int(round(2 * L)) + 1
    m = np.arange(L, -L - 1, -1)
    Sz = np.diag(m).astype(complex)
    sp = np.zeros((dim, dim), dtype=complex)
    for i in range(1, dim):
        mm = m[i]
        sp[i - 1, i] = np.sqrt(L * (L + 1) - mm * (mm + 1))
    Sx = (sp + sp.conj().T) / 2
    Sy = (sp - sp.conj().T) / 2j
    return [Sx, Sy, Sz]


def coherent_L(L, theta, phi):
    """Spin-L coherent state along n(theta,phi): rotation of the highest weight."""
    ops = spinL_ops(L)
    v = np.zeros(ops[0].shape[0], dtype=complex)
    v[0] = 1.0                                   # |L, m=+L>
    return expm(-1j * phi * ops[2]) @ (expm(-1j * theta * ops[1]) @ v)


def exact_aligned(d, J0, ts, hub_ang, leaf_ang):
    """Exact A(t) on grid ts for uniform couplings and ALL leaves aligned along
    leaf_ang.  Uses the collective-spin reduction: leaves form |L=d/2, coherent>,
    H = (J0/d) S_0.L acts on a 2(d+1)-dimensional space -> exact for any d."""
    L = d / 2
    SL = spinL_ops(L)
    H = (J0 / d) * sum(np.kron(S[a], SL[a]) for a in range(3))
    psi = np.kron(coh(*hub_ang), coherent_L(L, *leaf_ang))
    w, V = np.linalg.eigh(H)
    c = V.conj().T @ psi
    ph = np.exp(-1j * np.outer(np.atleast_1d(ts), w))      # (nt, dim)
    return np.einsum('k,tk->t', (psi.conj() @ V), ph * c[None, :])


def aligned_two_sector(d, J0, ts, hub_ang, leaf_ang):
    """Closed-form aligned-star amplitude via the two Clebsch-Gordan sectors
    j = L +- 1/2 of 1/2 (x) L, L = d/2.  Energies: E_+ = (J0/d) L/2,
    E_- = -(J0/d)(L+1)/2.  Returns A(t) on ts."""
    L = d / 2
    lam = J0 / d
    Ep, Em = lam * L / 2, -lam * (L + 1) / 2
    # rotate to frame where leaves point along +z; hub angle -> relative angle
    n0, nl = unit(*hub_ang), unit(*leaf_ang)
    cth = float(np.clip(n0 @ nl, -1, 1))
    # amplitude depends only on the relative angle theta between hub and leaves:
    # |Psi> = cos(t/2)|up>|L,L> + e^{i phi} sin(t/2)|dn>|L,L>  in the leaf frame.
    c2, s2 = (1 + cth) / 2, (1 - cth) / 2      # cos^2(t/2), sin^2(t/2)
    # CG: |up,L> = |j=L+1/2, m=L+1/2>;
    #     |dn,L> = a |L+1/2, L-1/2> + b |L-1/2, L-1/2>, a^2 = 1/(2L+1), b^2 = 2L/(2L+1)
    a2 = 1 / (2 * L + 1)
    b2 = 2 * L / (2 * L + 1)
    ts = np.atleast_1d(ts)
    return (c2 * np.exp(-1j * Ep * ts)
            + s2 * (a2 * np.exp(-1j * Ep * ts) + b2 * np.exp(-1j * Em * ts)))


def exact_ED(d, J0, ts, hub_ang, leaf_angles):
    """Brute-force ED oracle, arbitrary leaf orientations (and easily generalized
    couplings).  Cost 2^(d+1); use for d <= ~12."""
    def op_at(site, a, N):
        mats = [np.eye(2, dtype=complex)] * N
        mats[site] = S[a]
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out
    N = d + 1
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for c in range(1, N):
        for a in range(3):
            H += (J0 / d) * op_at(0, a, N) @ op_at(c, a, N)
    psi = coh(*hub_ang)
    for ang in leaf_angles:
        psi = np.kron(psi, coh(*ang))
    w, V = np.linalg.eigh(H)
    cvec = V.conj().T @ psi
    ph = np.exp(-1j * np.outer(np.atleast_1d(ts), w))
    return np.einsum('k,tk->t', (psi.conj() @ V), ph * cvec[None, :])


# ------------------------------------------------------------------------------ L0
def Bbar_of(J0, d, nus):
    return (J0 / (2 * d)) * np.sum(nus, axis=0)


def A_L0(J0, d, ts, hub_ang, nus):
    """L0 amplitude on grid ts (closed form: hub precessing in the static mean
    leaf field Bbar)."""
    Bbar = Bbar_of(J0, d, nus)
    b = np.linalg.norm(Bbar)
    n0v = unit(*hub_ang)
    ts = np.atleast_1d(ts)
    if b < 1e-15:
        return np.ones_like(ts, dtype=complex)
    ndotb = n0v @ (Bbar / b)
    return np.cos(b * ts / 2) - 1j * ndotb * np.sin(b * ts / 2)


# ------------------------------------------------------------------------------ G1
def dlog_G1(J0, d, T, hub_ang, nus, nt=161):
    """Gaussian nonlocal-in-time correction to log A at time T.

    Continuous-time double integral evaluated by ordered-triangle trapezoid
    (both integrand branches are smooth on the closed triangle t >= t', which
    restores second-order quadrature accuracy across the T-ordering kink)."""
    ts = np.linspace(0, T, nt)
    h = ts[1] - ts[0]
    Bbar = Bbar_of(J0, d, nus)
    HB = sum(Bbar[a] * S[a] for a in range(3))
    n0 = coh(*hub_ang)
    U1 = expm(-1j * h * HB)
    P = np.empty((nt, 2, 2), dtype=complex)
    P[0] = np.eye(2)
    for i in range(1, nt):
        P[i] = U1 @ P[i - 1]
    PT = P[-1]
    A0 = n0.conj() @ PT @ n0
    Pinv = np.linalg.inv(P)
    St = np.einsum('ikl,alm,imn->aikn', Pinv, np.array(S), P)   # (3,nt,2,2)
    w = n0.conj() @ PT
    Lrow = np.einsum('k,aikl->ail', w, St)     # <n0| P(T) S~a(ti)      (3,nt,2)
    Rcol = np.einsum('aikl,l->aik', St, n0)    # S~b(tj) |n0>           (3,nt,2)
    Ggt = np.einsum('ail,bjl->abij', Lrow, Rcol) / A0   # G_>[a,b,i,j] for ti >= tj
    tri = np.tril(np.ones((nt, nt)))
    Wt = np.full(nt, h); Wt[0] = Wt[-1] = h / 2
    lam2 = (J0 / d) ** 2
    total = 0.0 + 0.0j
    for nu in nus:
        ks = 0.25 * (np.eye(3) - np.outer(nu, nu))          # noise (symmetric)
        ka = 0.25j * np.einsum('abg,g->ab', eps, nu)        # response (antisym)
        f1 = np.einsum('ab,abij->ij', ks + ka, Ggt)
        f2 = np.einsum('ab,abij->ij', ks - ka, np.transpose(Ggt, (1, 0, 2, 3)))
        g = (f1 + f2) * tri
        inner = np.zeros(nt, dtype=complex)
        for i in range(1, nt):
            wj = np.full(i + 1, h); wj[0] = wj[-1] = h / 2
            inner[i] = wj @ g[i, :i + 1]
        total += Wt @ inner
    return -0.5 * lam2 * total


def dlog_G1_series(J0, d, Tlist, hub_ang, nus, nt=161):
    """dlog_G1 evaluated at each T in Tlist (each is an independent boundary-value
    computation: the weak two-point depends on the endpoint T)."""
    return np.array([dlog_G1(J0, d, T, hub_ang, nus, nt=nt) for T in Tlist])


# ----------------------------------------------------- self-consistent L0 (optional)
def _propagate(Bt, ts):
    """2x2 propagators of i dU/dt = (B(t).S) U for complex B(t); midpoint rule."""
    nt = len(ts)
    h = ts[1] - ts[0]
    P = np.empty((nt, 2, 2), dtype=complex)
    P[0] = np.eye(2)
    for i in range(1, nt):
        Bm = 0.5 * (Bt[i - 1] + Bt[i])
        P[i] = expm(-1j * h * sum(Bm[a] * S[a] for a in range(3))) @ P[i - 1]
    return P


def _weak_traj(P, psi):
    """Weak-value trajectory <psi|U(T,t) S U(t,0)|psi> / <psi|U(T)|psi>."""
    nt = P.shape[0]
    PT = P[-1]
    Pinv = np.linalg.inv(P)
    A = psi.conj() @ PT @ psi
    w = psi.conj() @ PT
    out = np.empty((nt, 3), dtype=complex)
    for i in range(nt):
        for a in range(3):
            out[i, a] = (w @ Pinv[i] @ S[a] @ P[i] @ psi) / A
    return out, A


def A_L0sc(J0, d, T, hub_ang, nus, nt=161, iters=80, damp=0.5, tol=1e-10):
    """Self-consistent on-shell approximation: hub and leaf weak-value trajectories
    determined jointly (complexified Landau-Lifshitz boundary-value equations),
    amplitude assembled with tangent bookkeeping.  Same O(1/d) order as strict L0,
    improved constants; embryo of the tree recursion."""
    ts = np.linspace(0, T, nt)
    n0 = coh(*hub_ang)
    Bw = np.tile(Bbar_of(J0, d, nus), (nt, 1)).astype(complex)
    for _ in range(iters):
        m0, _ = _weak_traj(_propagate(Bw, ts), n0)
        mu = np.zeros((nt, 3), dtype=complex)
        Pleaf = _propagate((J0 / d) * m0, ts)   # same field for every leaf
        for nu in nus:
            mc, _ = _weak_traj(Pleaf, coh(*ang_of(nu)))
            mu += mc
        Bnew = (J0 / d) * mu
        if np.max(np.abs(Bnew - Bw)) < tol:
            Bw = Bnew
            break
        Bw = damp * Bnew + (1 - damp) * Bw
    Phub = _propagate(Bw, ts)
    m0, Ahub = _weak_traj(Phub, n0)
    h = ts[1] - ts[0]
    wts = np.full(nt, h); wts[0] = wts[-1] = h / 2
    Pleaf = _propagate((J0 / d) * m0, ts)
    Phi = 0.0 + 0j
    for nu in nus:
        psi_c = coh(*ang_of(nu))
        Phi += np.log(psi_c.conj() @ Pleaf[-1] @ psi_c)
    corr = 1j * np.einsum('i,ia,ia->', wts, Bw, m0)
    return np.exp(Phi + corr) * Ahub


# --------------------------------------------------------------------------- helpers
def log_unwrapped(A):
    """Branch-continuous log of an amplitude sampled on a (fine) time grid starting
    at A(0)=1: log|A| + i * unwrap(arg A)."""
    A = np.asarray(A)
    return np.log(np.abs(A)) + 1j * np.unwrap(np.angle(A))


def fit_slope(x, y):
    """Least-squares slope of log y vs log x."""
    return np.polyfit(np.log(np.asarray(x, float)),
                      np.log(np.asarray(y, float)), 1)[0]


# =============================================================================
# Homogeneous star with ARBITRARY orientations: Schur-block (M_L) exact oracle
# =============================================================================
def cg_up_down(L):
    """Clebsch-Gordan isometries C_{L'} : V_L (x) C^2 -> V_{L'}, L' = L +- 1/2.
    Basis: |L,m> with m = L..-L (as spinL_ops); qubit ordered (up, dn);
    kron index = (m-index)*2 + s."""
    dimL = int(round(2 * L)) + 1
    mL = np.arange(L, -L - 1, -1)
    out = {}
    for sgn in (+1, -1):
        Lp = L + sgn * 0.5
        if Lp < -1e-9:
            continue
        dimP = int(round(2 * Lp)) + 1
        mP = np.arange(Lp, -Lp - 1, -1)
        C = np.zeros((dimP, dimL * 2))
        for ip, m in enumerate(mP):
            if sgn == +1:
                a = np.sqrt((L + m + 0.5) / (2 * L + 1))
                b = np.sqrt((L - m + 0.5) / (2 * L + 1))
            else:
                a = -np.sqrt((L - m + 0.5) / (2 * L + 1))
                b = np.sqrt((L + m + 0.5) / (2 * L + 1))
            for (mm, s, coef) in [(m - 0.5, 0, a), (m + 0.5, 1, b)]:
                idx = np.where(np.abs(mL - mm) < 1e-9)[0]
                if len(idx):
                    C[ip, idx[0] * 2 + s] = coef
        out[sgn] = C
    return out


def schur_blocks(leaf_angles):
    """Multiplicity-traced Schur blocks M_L = sum_copies V^+ |Phi><Phi| V of the
    leaf product coherent state; dict L -> (2L+1)x(2L+1) matrix.  Polynomial
    recursion (one CG contraction per added leaf).  sum_L Tr M_L = 1."""
    v = coh(*leaf_angles[0])
    M = {0.5: np.outer(v, v.conj())}
    for ang in leaf_angles[1:]:
        rho = np.outer(coh(*ang), coh(*ang).conj())
        Mnew = {}
        for L, ML in M.items():
            big = np.kron(ML, rho)
            for sgn, C in cg_up_down(L).items():
                Lp = L + sgn * 0.5
                blk = C @ big @ C.conj().T
                Mnew[Lp] = Mnew.get(Lp, 0) + blk
        M = Mnew
    return M


def exact_homog(d, J0, T, hub_ang, leaf_angles):
    """Exact homogeneous-star return amplitude for ARBITRARY leaf orientations,
    polynomial in d.  Uses the sector-wise operator identity
    exp(-i tau S0.L) = f(L^2) + g(L^2) S0.L  and the Schur blocks."""
    lam = J0 / d
    n0v = unit(*hub_ang)
    A = 0.0 + 0j
    for L, ML in schur_blocks(leaf_angles).items():
        xp, xm = L / 2, -(L + 1) / 2
        ep, em = np.exp(-1j * lam * T * xp), np.exp(-1j * lam * T * xm)
        g = (ep - em) / (xp - xm)
        f = ep - g * xp
        ops = spinL_ops(L)
        A += f * np.trace(ML)
        A += g * 0.5 * sum(n0v[a] * np.trace(ML @ ops[a]) for a in range(3))
    return A


# =============================================================================
# Fully inhomogeneous star: H = B0.S0 + sum_c [ b_c.S_c + J_c S0.S_c ]
# =============================================================================
def weak_objects(Bt, ts, psi):
    """For a (possibly complex) drive B(t) on grid ts and boundary state psi at
    both ends: returns (tau, mu, Ggt) = (return amplitude, weak mean trajectory
    (nt,3), weak T-ordered FULL two-point Ggt[a,b,i,j] valid for t_i >= t_j)."""
    nt = len(ts)
    h = ts[1] - ts[0]
    P = np.empty((nt, 2, 2), dtype=complex)
    P[0] = np.eye(2)
    for i in range(1, nt):
        Bm = 0.5 * (Bt[i - 1] + Bt[i])
        P[i] = expm(-1j * h * sum(Bm[a] * S[a] for a in range(3))) @ P[i - 1]
    PT = P[-1]
    tau = psi.conj() @ PT @ psi
    Pinv = np.linalg.inv(P)
    St = np.einsum('ikl,alm,imn->aikn', Pinv, np.array(S), P)
    w = psi.conj() @ PT
    Lrow = np.einsum('k,aikl->ail', w, St)
    Rcol = np.einsum('aikl,l->aik', St, psi)
    mu = np.einsum('ail,l->ia', Lrow, psi) / tau
    Ggt = np.einsum('ail,bjl->abij', Lrow, Rcol) / tau
    return tau, mu, Ggt


def tri_weights(nt, h):
    """Trapezoid weights on the closed ordered triangle i >= j."""
    W = np.zeros((nt, nt))
    for i in range(1, nt):
        wj = np.full(i + 1, h); wj[0] = wj[-1] = h / 2
        W[i, :i + 1] = wj
    Wt = np.full(nt, h); Wt[0] = Wt[-1] = h / 2
    return W * Wt[:, None]


def L0G1_inhom(Jc, B0, bcs, T, hub_ang, leaf_angles, nt=161):
    """Fully inhomogeneous L0 and G1: returns (log A_L0, dlog_G1).

    L0: each leaf c evolves under its own field b_c -> weak (complex) mean
    trajectory mu_c(t) and free factor tau_c; the hub is driven by
    Bbar(t) = B0 + sum_c J_c mu_c(t).  log A_L0 = sum_c log tau_c + log tau_hub.

    G1: dlog A = - sum_c J_c^2 * INT_{t>=t'} kappa_c^{ab} G0^{ab}  (the full
    square equals twice the ordered triangle), with kappa_c the weak CONNECTED
    leaf two-point and G0 the weak FULL hub two-point, both driven."""
    ts = np.linspace(0, T, nt)
    d = len(Jc)
    taus, mus, kappas = [], [], []
    for c in range(d):
        Bt = np.tile(np.asarray(bcs[c], dtype=complex), (nt, 1))
        tau, mu, Ggt = weak_objects(Bt, ts, coh(*leaf_angles[c]))
        kappas.append(Ggt - np.einsum('ia,jb->abij', mu, mu))
        taus.append(tau); mus.append(mu)
    Bbar = np.tile(np.asarray(B0, dtype=complex), (nt, 1)) \
        + sum(Jc[c] * mus[c] for c in range(d))
    tau0, mu0, G0 = weak_objects(Bbar, ts, coh(*hub_ang))
    logA_L0 = np.sum(np.log(np.asarray(taus))) + np.log(tau0)
    W = tri_weights(nt, ts[1] - ts[0])
    dG1 = 0.0 + 0j
    for c in range(d):
        f = np.einsum('abij,abij->ij', kappas[c], G0)
        dG1 += -Jc[c] ** 2 * np.sum(W * f)
    return logA_L0, dG1


def exact_sparse_inhom(Jc, B0, bcs, T, hub_ang, leaf_angles):
    """Sparse-Krylov ED oracle for the fully inhomogeneous star (d <= ~14)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    d = len(Jc)
    N = d + 1

    def op_at(site, a):
        mats = [sp.identity(2, format='csr', dtype=complex)] * N
        mats[site] = sp.csr_matrix(S[a])
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format='csr')
        return out
    H = sum(B0[a] * op_at(0, a) for a in range(3))
    for c in range(d):
        for a in range(3):
            H = H + bcs[c][a] * op_at(c + 1, a) \
                  + Jc[c] * (op_at(0, a) @ op_at(c + 1, a))
    psi = coh(*hub_ang)
    for ang in leaf_angles:
        psi = np.kron(psi, coh(*ang))
    return psi.conj() @ expm_multiply(-1j * T * H.tocsc(), psi)

# =============================================================================
# Fully driven star: H(t) = B0(t).S0 + sum_c [ b_c(t).Sc + S0^T K_c(t) Sc ]
# =============================================================================
def _as_time_array(x, nt, shape_tail, name="array"):
    """Convert a constant or time-dependent array to shape (nt, *shape_tail)."""
    arr = np.asarray(x, dtype=complex)
    if arr.shape == tuple(shape_tail):
        return np.tile(arr, (nt,) + (1,) * len(shape_tail))
    if arr.shape == (nt,) + tuple(shape_tail):
        return arr
    raise ValueError(f"{name} must have shape {shape_tail} or {(nt,) + tuple(shape_tail)}, got {arr.shape}")


def L0G1_driven(Kct, B0t, bct, T, hub_ang, leaf_angles, nt=161, return_diagnostics=False):
    """L0 and G1 for the fully driven anisotropic star.

    Parameters
    ----------
    Kct : array-like, shape (d, nt, 3, 3) or (d, 3, 3)
        Hub-leaf pair tensors in H_int(t)=sum_{a,b} S0^a K_c^{ab}(t) Sc^b.
        The intended controlled regime is ||K_c(t)||=O(1/d).
    B0t : array-like, shape (nt, 3) or (3,)
        Time-dependent hub field.
    bct : array-like, shape (d, nt, 3) or (d, 3)
        Time-dependent leaf fields.
    T : float
        Final time.  The arrays are interpreted on a uniform grid [0,T].
    hub_ang, leaf_angles : coherent-state angles.
    nt : int
        Number of grid points.  If Kct/B0t/bct are time-dependent, their second
        dimension must equal nt.
    return_diagnostics : bool
        If True, also return a dictionary with leaf means/kernels and the hub field.

    Returns
    -------
    logA_L0, dlog_G1 or (logA_L0, dlog_G1, diagnostics)

    Notes
    -----
    The G1 term is evaluated on the ordered triangle t_i >= t_j:
        Delta_G1 = - sum_c int_{t>=t'} K_i K_j kappa_c^>(i,j) G0^>(i,j).
    This is equivalent to the full time-ordered square expression with the
    conventional factor 1/2, up to the usual diagonal convention represented here
    by trapezoid weights.
    """
    d = len(leaf_angles)
    ts = np.linspace(0, T, nt)
    Kct = np.asarray(Kct, dtype=complex)
    bct = np.asarray(bct, dtype=complex)
    if Kct.shape == (d, 3, 3):
        Kct = np.tile(Kct[:, None, :, :], (1, nt, 1, 1))
    if bct.shape == (d, 3):
        bct = np.tile(bct[:, None, :], (1, nt, 1))
    B0t = _as_time_array(B0t, nt, (3,), "B0t")
    if Kct.shape != (d, nt, 3, 3):
        raise ValueError(f"Kct must have shape {(d, nt, 3, 3)} or {(d, 3, 3)}, got {Kct.shape}")
    if bct.shape != (d, nt, 3):
        raise ValueError(f"bct must have shape {(d, nt, 3)} or {(d, 3)}, got {bct.shape}")

    taus, mus, kappas = [], [], []
    for c in range(d):
        tau, mu, Ggt = weak_objects(bct[c], ts, coh(*leaf_angles[c]))
        # Connected leaf two-point, with indices beta,delta,i,j on ordered triangle.
        kappas.append(Ggt - np.einsum('ib,jd->bdij', mu, mu))
        taus.append(tau)
        mus.append(mu)

    # Effective hub field B_eff^alpha(t)=B0^alpha(t)+sum_c K_c^{alpha beta}(t) mu_c^beta(t).
    Beff = B0t.copy()
    for c in range(d):
        Beff += np.einsum('iab,ib->ia', Kct[c], mus[c])

    tau0, mu0, G0 = weak_objects(Beff, ts, coh(*hub_ang))
    logA_L0 = np.sum(np.log(np.asarray(taus))) + np.log(tau0)

    W = tri_weights(nt, ts[1] - ts[0])
    dG1 = 0.0 + 0.0j
    for c in range(d):
        # f_ij = K_i^{a b} K_j^{g d} kappa^{b d}_{ij} G0^{a g}_{ij}
        f = np.einsum('iab,jgd,bdij,agij->ij', Kct[c], Kct[c], kappas[c], G0, optimize=True)
        dG1 += -np.sum(W * f)

    if return_diagnostics:
        return logA_L0, dG1, {
            'ts': ts,
            'taus': np.asarray(taus),
            'mus': np.asarray(mus),
            'kappas': np.asarray(kappas),
            'Beff': Beff,
            'tau0': tau0,
            'mu0': mu0,
            'G0': G0,
        }
    return logA_L0, dG1


def exact_sparse_driven(Kct, B0t, bct, ts, hub_ang, leaf_angles, cache_ops=None):
    """Sparse midpoint/Krylov oracle for the fully driven anisotropic star.

    Returns the exact return amplitude sampled on the time grid `ts`, using a
    piecewise-constant midpoint approximation to the time-ordered exponential.
    Intended for validation at d <= about 10-12, depending on hardware.

    Parameters are as in L0G1_driven, but arrays must be defined on len(ts) grid
    points or be constant in time.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply

    ts = np.asarray(ts, dtype=float)
    nt = len(ts)
    if nt < 2:
        raise ValueError("ts must contain at least two grid points")
    d = len(leaf_angles)
    N = d + 1
    Kct = np.asarray(Kct, dtype=complex)
    bct = np.asarray(bct, dtype=complex)
    if Kct.shape == (d, 3, 3):
        Kct = np.tile(Kct[:, None, :, :], (1, nt, 1, 1))
    if bct.shape == (d, 3):
        bct = np.tile(bct[:, None, :], (1, nt, 1))
    B0t = _as_time_array(B0t, nt, (3,), "B0t")
    if Kct.shape != (d, nt, 3, 3):
        raise ValueError(f"Kct must have shape {(d, nt, 3, 3)} or {(d, 3, 3)}, got {Kct.shape}")
    if bct.shape != (d, nt, 3):
        raise ValueError(f"bct must have shape {(d, nt, 3)} or {(d, 3)}, got {bct.shape}")

    if cache_ops is None:
        def op_at(site, a):
            mats = [sp.identity(2, format='csr', dtype=complex)] * N
            mats[site] = sp.csr_matrix(S[a])
            out = mats[0]
            for m in mats[1:]:
                out = sp.kron(out, m, format='csr')
            return out
        O = [[op_at(site, a) for a in range(3)] for site in range(N)]
        E = [[[O[0][a] @ O[c + 1][b] for b in range(3)] for a in range(3)] for c in range(d)]
    else:
        O, E = cache_ops

    psi0 = coh(*hub_ang)
    for ang in leaf_angles:
        psi0 = np.kron(psi0, coh(*ang))
    psi = psi0.copy()
    amps = np.empty(nt, dtype=complex)
    amps[0] = psi0.conj() @ psi

    for i in range(1, nt):
        dt = ts[i] - ts[i - 1]
        Bm = 0.5 * (B0t[i - 1] + B0t[i])
        bm = 0.5 * (bct[:, i - 1, :] + bct[:, i, :])
        Km = 0.5 * (Kct[:, i - 1, :, :] + Kct[:, i, :, :])
        H = sp.csr_matrix((2 ** N, 2 ** N), dtype=complex)
        for a in range(3):
            H = H + Bm[a] * O[0][a]
        for c in range(d):
            for b in range(3):
                H = H + bm[c, b] * O[c + 1][b]
            for a in range(3):
                for b in range(3):
                    if abs(Km[c, a, b]) > 0:
                        H = H + Km[c, a, b] * E[c][a][b]
        psi = expm_multiply(-1j * dt * H.tocsc(), psi)
        amps[i] = psi0.conj() @ psi
    return amps


def make_driven_star_instance(d, nt, T, seed=1, coupling_scale=1.0, anisotropy=0.25):
    """Generate a reproducible smooth driven star instance for validation.

    Couplings are O(1/d): K_c(t) = (coupling_scale/d) * [smooth isotropic part +
    smooth anisotropic symmetric/antisymmetric perturbations].  Fields are O(1)
    and smooth, with leaf-dependent phases.
    """
    rng = np.random.default_rng(seed)
    ts = np.linspace(0, T, nt)
    x = ts / max(T, 1e-15)

    B0_base = np.array([0.23, -0.17, 0.31])
    B0_drive = np.array([0.11 * np.sin(2 * np.pi * x + 0.3),
                         0.08 * np.cos(2 * np.pi * 0.7 * x - 0.2),
                         0.05 * np.sin(2 * np.pi * 1.3 * x + 0.6)]).T
    B0t = B0_base[None, :] + B0_drive

    bct = np.zeros((d, nt, 3), dtype=float)
    Kct = np.zeros((d, nt, 3, 3), dtype=float)
    leaf_angles = []
    for c in range(d):
        th = np.arccos(rng.uniform(-0.8, 0.8))
        ph = rng.uniform(0, 2 * np.pi)
        leaf_angles.append((th, ph))
        phase = rng.uniform(0, 2 * np.pi, size=3)
        freq = rng.uniform(0.5, 1.7, size=3)
        b0 = rng.normal(0.0, 0.18, size=3) + np.array([0.05, -0.03, 0.02])
        for a in range(3):
            bct[c, :, a] = b0[a] + 0.09 * np.sin(2 * np.pi * freq[a] * x + phase[a])

        # Construct a small anisotropic tensor with both symmetric and DM-like components.
        A = rng.normal(size=(3, 3))
        sym = 0.5 * (A + A.T)
        anti = 0.5 * (A - A.T)
        sym /= max(np.linalg.norm(sym), 1e-12)
        anti /= max(np.linalg.norm(anti), 1e-12)
        j0 = rng.normal(1.0, 0.15)
        amp = rng.uniform(0.08, 0.18)
        omega = rng.uniform(0.7, 1.5)
        phi = rng.uniform(0, 2 * np.pi)
        for i, xx in enumerate(x):
            scalar = j0 + amp * np.sin(2 * np.pi * omega * xx + phi)
            Kct[c, i] = (coupling_scale / d) * (scalar * np.eye(3) + anisotropy * (sym + 0.5 * anti))
    hub_ang = (0.92, 0.37)
    return ts, B0t, bct, Kct, hub_ang, leaf_angles
