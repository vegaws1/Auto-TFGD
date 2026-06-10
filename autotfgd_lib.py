"""
Auto-TFGD experimental library
===============================
Reference implementation of Tempered Fractional Gradient Descent (TFGD),
its adaptive extension Auto-TFGD (windowed + recursive), FractionalGD, and a
fair-comparison harness (per-optimizer LR tuning, multi-seed evaluation,
paired statistical tests, effect sizes, and cost profiling).

Everything here is deterministic given a seed.  No result is hard-coded.

Tempered fractional kernel convention
-------------------------------------
The fractional-memory weights w_j(alpha) are the coefficients of the
generating function (1 - x)^{-alpha}:
        w_0 = 1,   w_j = w_{j-1} * (alpha + j - 1) / j  >= 0,
so that  sum_{j>=0} w_j x^j = (1-x)^{-alpha}.
With tempering x = e^{-lambda} this gives the alignment / kernel mass
        d(alpha,lambda) = sum_{j>=0} w_j e^{-lambda j} = (1 - e^{-lambda})^{-alpha},
which is exactly the coefficient used in the manuscript.

  * Windowed TFGD/Auto-TFGD :  theta <- theta - eta * sum_{j=0}^{J} w_j(a) e^{-l j} g_{k-j}
  * Recursive surrogate     :  S <- d(a,l) g_k + e^{-l} S ;  theta <- theta - eta S
  * FractionalGD            :  windowed with lambda = 0 (no tempering), fixed alpha.
"""
import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Fractional-weight helper                                                   #
# --------------------------------------------------------------------------- #
def gl_weights(alpha, J):
    """w_j(alpha), j = 0..J  for generating function (1-x)^{-alpha} (all >= 0)."""
    w = [1.0]
    for j in range(1, J + 1):
        w.append(w[-1] * (alpha + j - 1) / j)
    return w


def kernel_mass(alpha, lam):
    """d(alpha,lambda) = (1 - e^{-lambda})^{-alpha}."""
    return (1.0 - math.exp(-lam)) ** (-alpha)


# --------------------------------------------------------------------------- #
#  Base optimizer with global gradient statistics + (optional) adaptation     #
# --------------------------------------------------------------------------- #
class _MemoryOpt(torch.optim.Optimizer):
    """
    Shared machinery for the fractional-memory optimizers.

    variant : 'frac' | 'tfgd_win' | 'tfgd_rec' | 'auto_win' | 'auto_rec'
    """

    def __init__(self, params, lr=1e-2, alpha0=0.8, lam0=0.1,
                 variant='tfgd_rec', W_cap=50,
                 alpha_min=0.3, alpha_max=0.95, lam_min=0.02, lam_max=1.0,
                 c_sigma=1.0, c_kappa=1.0, c_kappa_alpha=1.0,
                 ema_beta=0.9, adapt_alpha=True, adapt_lambda=True,
                 random_adapt=False, seed=0):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.variant = variant
        self.alpha0, self.lam0 = alpha0, lam0
        self.W_cap = W_cap
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.lam_min, self.lam_max = lam_min, lam_max
        self.c_sigma, self.c_kappa, self.c_kappa_alpha = c_sigma, c_kappa, c_kappa_alpha
        self.ema_beta = ema_beta
        self.adapt_alpha, self.adapt_lambda = adapt_alpha, adapt_lambda
        self.random_adapt = random_adapt
        self._rng = np.random.default_rng(seed)
        # running statistics
        self.m1 = None    # EMA(||g||)
        self.m2 = None    # EMA(||g||^2)
        self.t = 0
        # trajectories (for plots / sanity)
        self.alpha_hist, self.lam_hist = [], []

    # -- statistics ------------------------------------------------------- #
    def _global_stats(self):
        sq, diffsq = 0.0, 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                sq += float(torch.sum(g * g))
                st = self.state[p]
                gprev = st.get('g_prev')
                if gprev is not None:
                    d = g - gprev
                    diffsq += float(torch.sum(d * d))
                st['g_prev'] = g.clone()
        return math.sqrt(sq), math.sqrt(diffsq)

    def _adapt(self, gnorm, gdiff):
        b = self.ema_beta
        self.m1 = gnorm if self.m1 is None else b * self.m1 + (1 - b) * gnorm
        self.m2 = gnorm * gnorm if self.m2 is None else b * self.m2 + (1 - b) * gnorm * gnorm
        var = max(self.m2 - self.m1 * self.m1, 0.0)
        sig = var / (self.m1 * self.m1 + 1e-12)        # normalised relative variance
        kap = gdiff / (self.m1 + 1e-12)                # normalised curvature proxy
        if self.random_adapt:
            # control: ignore signals, jump randomly inside the boxes
            lam = float(self._rng.uniform(self.lam_min, self.lam_max))
            alp = float(self._rng.uniform(self.alpha_min, self.alpha_max))
            return alp, lam
        lam = self.lam0 * (1 + self.c_sigma * sig) / (1 + self.c_kappa * kap)
        alp = self.alpha0 * (1 + self.c_kappa_alpha * kap / (1 + sig)) ** (-1)
        if not self.adapt_lambda:
            lam = self.lam0
        if not self.adapt_alpha:
            alp = self.alpha0
        lam = min(max(lam, self.lam_min), self.lam_max)
        alp = min(max(alp, self.alpha_min), self.alpha_max)
        return alp, lam

    # -- step ------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self.t += 1

        if self.variant in ('auto_win', 'auto_rec'):
            gnorm, gdiff = self._global_stats()
            alpha, lam = self._adapt(gnorm, gdiff)
        else:
            alpha, lam = self.alpha0, self.lam0
            if self.variant in ('tfgd_win', 'tfgd_rec', 'frac'):
                # still track g_prev cheaply not needed; skip stats for speed
                pass
        self.alpha_hist.append(alpha)
        self.lam_hist.append(lam)

        recursive = self.variant in ('tfgd_rec', 'auto_rec')
        windowed = self.variant in ('tfgd_win', 'auto_win', 'frac')
        lam_eff = 0.0 if self.variant == 'frac' else lam

        if recursive:
            d = kernel_mass(alpha, lam)
            decay = math.exp(-lam)
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    S = st.get('S')
                    S = d * g.clone() if S is None else d * g + decay * S
                    st['S'] = S
                    p.add_(S, alpha=-lr)
        elif windowed:
            J = self.W_cap if self.variant == 'frac' else min(self.W_cap, int(math.ceil(5.0 / max(lam, 1e-6))))
            w = gl_weights(alpha, J)
            coeff = [w[j] * math.exp(-lam_eff * j) for j in range(J + 1)]
            for group in self.param_groups:
                lr = group['lr']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    hist = st.get('hist')
                    if hist is None:
                        hist = deque(maxlen=self.W_cap + 1)
                        st['hist'] = hist
                    hist.appendleft(g.clone())
                    direction = None
                    for j, gj in enumerate(hist):
                        if j > J:
                            break
                        term = gj * coeff[j]
                        direction = term if direction is None else direction + term
                    p.add_(direction, alpha=-lr)
        return loss


def make_optimizer(name, params, lr, seed=0, W_cap=50,
                   alpha0=0.8, lam0=0.1,
                   adapt_alpha=True, adapt_lambda=True, random_adapt=False):
    """Factory. Standard optimizers use torch.optim; the rest use _MemoryOpt."""
    plist = list(params)
    if name == 'SGD':
        return torch.optim.SGD(plist, lr=lr, momentum=0.9)
    if name == 'Adam':
        return torch.optim.Adam(plist, lr=lr)
    if name == 'AdamW':
        return torch.optim.AdamW(plist, lr=lr, weight_decay=1e-2)
    if name == 'RMSprop':
        return torch.optim.RMSprop(plist, lr=lr)
    if name == 'FractionalGD':
        return _MemoryOpt(plist, lr=lr, variant='frac', alpha0=alpha0, W_cap=W_cap, seed=seed)
    if name == 'TFGD_Truncated':
        return _MemoryOpt(plist, lr=lr, variant='tfgd_win', alpha0=alpha0, lam0=lam0, W_cap=W_cap, seed=seed)
    if name == 'TFGD_Recursive':
        return _MemoryOpt(plist, lr=lr, variant='tfgd_rec', alpha0=alpha0, lam0=lam0, seed=seed)
    if name == 'AutoTFGD_Truncated':
        return _MemoryOpt(plist, lr=lr, variant='auto_win', alpha0=alpha0, lam0=lam0, W_cap=W_cap, seed=seed,
                          adapt_alpha=adapt_alpha, adapt_lambda=adapt_lambda, random_adapt=random_adapt)
    if name == 'AutoTFGD_Recursive':
        return _MemoryOpt(plist, lr=lr, variant='auto_rec', alpha0=alpha0, lam0=lam0, seed=seed,
                          adapt_alpha=adapt_alpha, adapt_lambda=adapt_lambda, random_adapt=random_adapt)
    raise ValueError(name)


STANDARD = ['SGD', 'Adam', 'AdamW', 'RMSprop']
FRACTIONAL = ['FractionalGD', 'TFGD_Recursive', 'TFGD_Truncated']
PROPOSED = ['AutoTFGD_Recursive', 'AutoTFGD_Truncated']
ALL_OPT = STANDARD + FRACTIONAL + PROPOSED


# --------------------------------------------------------------------------- #
#  Models                                                                     #
# --------------------------------------------------------------------------- #
class LogReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


class MLPBin(nn.Module):
    def __init__(self, d, h=(64, 32)):
        super().__init__()
        layers, prev = [], d
        for hh in h:
            layers += [nn.Linear(prev, hh), nn.ReLU()]
            prev = hh
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPMulti(nn.Module):
    def __init__(self, d, k, h=(256, 128)):
        super().__init__()
        layers, prev = [], d
        for hh in h:
            layers += [nn.Linear(prev, hh), nn.ReLU()]
            prev = hh
        layers += [nn.Linear(prev, k)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
#  Training / evaluation                                                      #
# --------------------------------------------------------------------------- #
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_eval(model_fn, opt_name, data, *, lr, epochs, batch_size, multiclass,
               seed, W_cap=50, alpha0=0.8, lam0=0.1,
               adapt_alpha=True, adapt_lambda=True, random_adapt=False,
               profile=False):
    """Train one model and return metrics on the test split."""
    from sklearn.metrics import roc_auc_score, accuracy_score
    Xtr, ytr, Xte, yte = data
    set_seed(seed)
    model = model_fn()
    opt = make_optimizer(opt_name, model.parameters(), lr, seed=seed, W_cap=W_cap,
                         alpha0=alpha0, lam0=lam0, adapt_alpha=adapt_alpha,
                         adapt_lambda=adapt_lambda, random_adapt=random_adapt)
    if multiclass:
        lossf = nn.CrossEntropyLoss()
        ytr_t = torch.as_tensor(ytr, dtype=torch.long)
    else:
        lossf = nn.BCEWithLogitsLoss()
        ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    n = Xtr_t.shape[0]
    ep_times = []
    for _ in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for bidx in _batches(n, batch_size, rng):
            bi = torch.as_tensor(bidx)
            opt.zero_grad()
            out = model(Xtr_t[bi])
            loss = lossf(out, ytr_t[bi])
            loss.backward()
            opt.step()
        ep_times.append(time.perf_counter() - t0)
    # evaluation
    model.eval()
    with torch.no_grad():
        Xte_t = torch.as_tensor(Xte, dtype=torch.float32)
        out = model(Xte_t)
        if multiclass:
            prob = torch.softmax(out, dim=1).numpy()
            pred = prob.argmax(1)
            acc = accuracy_score(yte, pred)
            try:
                auc = roc_auc_score(yte, prob, multi_class='ovr', average='macro')
            except Exception:
                auc = float('nan')
        else:
            prob = torch.sigmoid(out).numpy()
            pred = (prob >= 0.5).astype(int)
            acc = accuracy_score(yte, pred)
            try:
                auc = roc_auc_score(yte, prob)
            except Exception:
                auc = float('nan')
    res = dict(acc=float(acc), auc=float(auc),
               time_per_epoch=float(np.median(ep_times)),
               n_params=sum(p.numel() for p in model.parameters()))
    if profile and hasattr(opt, 'alpha_hist'):
        res['alpha_final'] = float(np.mean(opt.alpha_hist[-50:])) if opt.alpha_hist else float('nan')
        res['lam_final'] = float(np.mean(opt.lam_hist[-50:])) if opt.lam_hist else float('nan')
    return res
