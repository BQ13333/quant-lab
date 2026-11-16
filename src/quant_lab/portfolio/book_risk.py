"""Arithmetic checks for planning priors; these are not admission evidence."""
import numpy as np


def book_active_risk(weights, te, rho):
    w, t = np.asarray(weights, dtype=float), np.asarray(te, dtype=float)
    r = np.asarray(rho, dtype=float)
    if r.ndim == 0:
        r = np.full((len(w), len(w)), float(r)); np.fill_diagonal(r, 1.)
    if w.ndim != 1 or t.shape != w.shape or r.shape != (len(w),len(w)) or not len(w) or not all(np.isfinite(x).all() for x in (w,t,r)) or np.any(w < 0) or np.any(t < 0) or not np.isclose(w.sum(),1.) or not np.allclose(r,r.T) or not np.allclose(np.diag(r),1.) or np.any(np.abs(r)>1) or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('invalid risk inputs or non-PSD correlation')
    exposure = w*t
    return float(np.sqrt(max(0., exposure@r@exposure)))


def net_ir_range(g_lo, g_hi, cost, impact, te):
    if not np.isfinite([g_lo,g_hi,cost,impact,te]).all() or g_lo > g_hi or min(cost,impact)<0 or te<=0:
        raise ValueError('invalid IR inputs')
    return ((g_lo-cost-impact)/te, (g_hi-cost-impact)/te)
