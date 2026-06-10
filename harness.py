"""
Harness: datasets, fair per-optimizer LR tuning, multi-seed evaluation,
paired statistical tests with multiple-comparison correction and effect sizes.

CLI:
    python harness.py --task lr      --seeds 20
    python harness.py --task mlp     --seeds 20
    python harness.py --task noisy   --seeds 20
    python harness.py --task mnist   --seeds 8  --epochs 30 --mnist_n 20000
    python harness.py --task ablation --seeds 20
Results are written to results/<task>.json
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import autotfgd_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(HERE, 'results')
os.makedirs(RESDIR, exist_ok=True)


# --------------------------------------------------------------------------- #
#  Datasets  ->  ((Xtr,ytr,Xte,yte), (Xtr,ytr,Xval,yval), meta)               #
# --------------------------------------------------------------------------- #
def _split_scale(X, y, seed=0, val=True):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if not val:
        return Xtr, ytr, Xte, yte
    Xtr2, Xval, ytr2, yval = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=ytr)
    return (Xtr, ytr, Xte, yte), (Xtr2, ytr2, Xval, yval)


def get_dataset(task, seed=0, mnist_n=20000, noise_std=0.25):
    if task in ('lr', 'mlp'):
        d = load_breast_cancer()
        X, y = d.data.astype('float32'), d.target.astype('int64')
        full, tune = _split_scale(X, y, seed=42)
        meta = dict(multiclass=False, d=X.shape[1], k=2,
                    model='logreg' if task == 'lr' else 'mlp_bin',
                    name='Breast Cancer Wisconsin (LR)' if task == 'lr' else 'Breast Cancer Wisconsin (MLP)')
        return full, tune, meta
    if task == 'noisy':
        d = load_digits()
        X, y = d.data.astype('float32'), d.target.astype('int64')
        X = X / 16.0
        rng = np.random.default_rng(0)
        X = X + rng.normal(0, noise_std, size=X.shape).astype('float32')   # input noise
        full, tune = _split_scale(X, y, seed=42)
        meta = dict(multiclass=True, d=X.shape[1], k=10, model='mlp_multi',
                    name='Noisy Digits (MLP)')
        return full, tune, meta
    if task == 'mnist':
        cache = os.path.join(RESDIR, 'mnist_cache.npz')
        if os.path.exists(cache):
            z = np.load(cache)
            X, y = z['X'], z['y']
        else:
            mn = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
            X = (mn.data.astype('float32') / 255.0)
            y = mn.target.astype('int64')
            np.savez_compressed(cache, X=X, y=y)
        if mnist_n and mnist_n < X.shape[0]:
            rng = np.random.default_rng(0)
            idx = rng.choice(X.shape[0], size=mnist_n, replace=False)
            X, y = X[idx], y[idx]
        full, tune = _split_scale(X, y, seed=42)
        meta = dict(multiclass=True, d=784, k=10, model='mlp_multi', name='MNIST (MLP)')
        return full, tune, meta
    raise ValueError(task)


def model_factory(meta):
    m = meta['model']
    if m == 'logreg':
        return lambda: L.LogReg(meta['d'])
    if m == 'mlp_bin':
        return lambda: L.MLPBin(meta['d'], h=(64, 32))
    if m == 'mlp_multi':
        h = (256, 128) if meta['d'] == 784 else (128, 64)
        return lambda: L.MLPMulti(meta['d'], meta['k'], h=h)
    raise ValueError(m)


# --------------------------------------------------------------------------- #
#  LR grids (fair tuning)                                                     #
# --------------------------------------------------------------------------- #
LR_GRID = {
    'SGD':                [3e-3, 1e-2, 3e-2, 1e-1, 3e-1],
    'Adam':               [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'AdamW':              [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'RMSprop':            [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    'FractionalGD':       [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    'TFGD_Recursive':     [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'TFGD_Truncated':     [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    'AutoTFGD_Recursive': [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    'AutoTFGD_Truncated': [1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
}


def tune_lr(opt_name, tune_data, meta, epochs, batch_size, W_cap, tune_seeds=(42, 43)):
    Xtr, ytr, Xval, yval = tune_data
    tune_epochs = min(epochs, 40)   # LR selection is stable; cap tuning cost
    if meta['d'] == 784:            # MNIST: reduce tuning cost (CPU)
        tune_epochs = min(epochs, 15)
    best, best_acc = None, -1.0
    for lr in LR_GRID[opt_name]:
        accs = []
        for s in tune_seeds:
            r = L.train_eval(model_factory(meta), opt_name, (Xtr, ytr, Xval, yval),
                             lr=lr, epochs=tune_epochs, batch_size=batch_size,
                             multiclass=meta['multiclass'], seed=s, W_cap=W_cap)
            accs.append(r['acc'])
        m = float(np.mean(accs))
        if m > best_acc:
            best_acc, best = m, lr
    return best, best_acc


# --------------------------------------------------------------------------- #
#  Statistics                                                                 #
# --------------------------------------------------------------------------- #
def paired_stats(a, b):
    """a (reference) vs b, paired over seeds. Returns dict."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    out = dict(mean_ref=float(a.mean()), mean_oth=float(b.mean()),
               mean_diff=float(diff.mean()))
    # paired t
    try:
        t, p_t = stats.ttest_rel(a, b)
        out['t_stat'], out['p_ttest'] = float(t), float(p_t)
    except Exception:
        out['t_stat'], out['p_ttest'] = float('nan'), float('nan')
    # Wilcoxon signed-rank
    try:
        if np.allclose(diff, 0):
            out['w_stat'], out['p_wilcoxon'] = float('nan'), 1.0
        else:
            w, p_w = stats.wilcoxon(a, b)
            out['w_stat'], out['p_wilcoxon'] = float(w), float(p_w)
    except Exception:
        out['w_stat'], out['p_wilcoxon'] = float('nan'), float('nan')
    # Cohen's d_z (paired)
    sd = diff.std(ddof=1)
    out['cohen_dz'] = float(diff.mean() / sd) if sd > 0 else float('nan')
    # Cliff's delta
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    out['cliffs_delta'] = float((gt - lt) / (len(a) * len(b)))
    return out


def holm_correct(pvals):
    """Holm-Bonferroni. pvals: list. Returns list of adjusted p-values (same order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def ci95(x):
    from scipy import stats
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else 0.0
    return float(m), float(sd), float(m - h), float(m + h)


# --------------------------------------------------------------------------- #
#  Main per-task run                                                          #
# --------------------------------------------------------------------------- #
def run_task(task, seeds, epochs, batch_size, W_cap, mnist_n, reference='AutoTFGD_Recursive'):
    full, tune, meta = get_dataset(task, mnist_n=mnist_n)
    Xtr, ytr, Xte, yte = full
    seed_list = list(range(42, 42 + seeds))
    print(f"[{task}] {meta['name']}  d={meta['d']} k={meta['k']} "
          f"train={len(ytr)} test={len(yte)} seeds={seeds} epochs={epochs}", flush=True)

    tuned_lr = {}
    per_opt = {}
    for opt in L.ALL_OPT:
        t0 = time.time()
        lr, vacc = tune_lr(opt, tune, meta, epochs, batch_size, W_cap)
        tuned_lr[opt] = lr
        accs, aucs, teps = [], [], []
        afin, lfin = [], []
        for s in seed_list:
            r = L.train_eval(model_factory(meta), opt, full, lr=lr, epochs=epochs,
                             batch_size=batch_size, multiclass=meta['multiclass'],
                             seed=s, W_cap=W_cap, profile=True)
            accs.append(r['acc']); aucs.append(r['auc']); teps.append(r['time_per_epoch'])
            if 'alpha_final' in r:
                afin.append(r['alpha_final']); lfin.append(r['lam_final'])
        per_opt[opt] = dict(lr=lr, val_acc=vacc, acc=accs, auc=aucs,
                            time_per_epoch=float(np.median(teps)),
                            n_params=r['n_params'],
                            alpha_final=(float(np.mean(afin)) if afin else None),
                            lam_final=(float(np.mean(lfin)) if lfin else None))
        am, asd, alo, ahi = ci95(accs)
        print(f"  {opt:20s} lr={lr:<7g} acc={am:.4f}+/-{asd:.4f} "
              f"CI[{alo:.4f},{ahi:.4f}] t/ep={np.median(teps)*1000:.1f}ms", flush=True)
        # incremental checkpoint so an interruption never loses completed optimizers
        with open(os.path.join(RESDIR, f'{task}_partial.json'), 'w') as f:
            json.dump({'task': task, 'tuned_lr': tuned_lr, 'per_opt': per_opt}, f)

    # paired stats: reference vs each other optimizer (on accuracy, seed-matched)
    comparisons = []
    raw_p = []
    for opt in L.ALL_OPT:
        if opt == reference:
            continue
        st = paired_stats(per_opt[reference]['acc'], per_opt[opt]['acc'])
        st['vs'] = opt
        comparisons.append(st)
        raw_p.append(st['p_wilcoxon'])
    adj = holm_correct([p if not np.isnan(p) else 1.0 for p in raw_p])
    for c, a in zip(comparisons, adj):
        c['p_wilcoxon_holm'] = float(a)

    summary = {}
    for opt in L.ALL_OPT:
        m, sd, lo, hi = ci95(per_opt[opt]['acc'])
        am, asd, _, _ = ci95(per_opt[opt]['auc'])
        summary[opt] = dict(acc_mean=m, acc_std=sd, acc_ci=[lo, hi],
                            auc_mean=am, auc_std=asd,
                            lr=per_opt[opt]['lr'],
                            time_per_epoch=per_opt[opt]['time_per_epoch'],
                            n_params=per_opt[opt]['n_params'],
                            alpha_final=per_opt[opt]['alpha_final'],
                            lam_final=per_opt[opt]['lam_final'])

    out = dict(task=task, meta=meta, seeds=seed_list, epochs=epochs,
               batch_size=batch_size, W_cap=W_cap, reference=reference,
               tuned_lr=tuned_lr, per_opt=per_opt, summary=summary,
               comparisons=comparisons)
    path = os.path.join(RESDIR, f'{task}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"[{task}] saved -> {path}", flush=True)
    return out


# --------------------------------------------------------------------------- #
#  Ablation (on Noisy Digits by default: clearest adaptation signal)          #
# --------------------------------------------------------------------------- #
def run_ablation(seeds, epochs, batch_size, W_cap, task='noisy'):
    full, tune, meta = get_dataset(task)
    seed_list = list(range(42, 42 + seeds))
    variants = {
        'Both (Auto-TFGD)':      dict(adapt_alpha=True,  adapt_lambda=True),
        'Adapt alpha only':      dict(adapt_alpha=True,  adapt_lambda=False),
        'Adapt lambda only':     dict(adapt_alpha=False, adapt_lambda=True),
        'Fixed (no adaptation)': dict(adapt_alpha=False, adapt_lambda=False),
        'Random adaptation':     dict(adapt_alpha=True,  adapt_lambda=True, random_adapt=True),
    }
    # tune one lr for the recursive auto variant, reuse for all ablation cells
    lr, _ = tune_lr('AutoTFGD_Recursive', tune, meta, epochs, batch_size, W_cap)
    print(f"[ablation/{task}] lr={lr}", flush=True)
    res = {}
    for name, kw in variants.items():
        accs = []
        for s in seed_list:
            r = L.train_eval(model_factory(meta), 'AutoTFGD_Recursive', full, lr=lr,
                             epochs=epochs, batch_size=batch_size,
                             multiclass=meta['multiclass'], seed=s, W_cap=W_cap, **kw)
            accs.append(r['acc'])
        m, sd, lo, hi = ci95(accs)
        res[name] = dict(acc=accs, mean=m, std=sd, ci=[lo, hi])
        print(f"  {name:24s} acc={m:.4f}+/-{sd:.4f}", flush=True)
    out = dict(task='ablation', base=task, lr=lr, seeds=seed_list, result=res)
    with open(os.path.join(RESDIR, 'ablation.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print("[ablation] saved", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True,
                    choices=['lr', 'mlp', 'noisy', 'mnist', 'ablation'])
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--epochs', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=0)
    ap.add_argument('--W_cap', type=int, default=50)
    ap.add_argument('--mnist_n', type=int, default=20000)
    args = ap.parse_args()

    defaults = dict(lr=(150, 32), mlp=(150, 32), noisy=(120, 64),
                    mnist=(30, 128), ablation=(120, 64))
    de, db = defaults[args.task]
    epochs = args.epochs or de
    bs = args.batch_size or db

    t0 = time.time()
    if args.task == 'ablation':
        run_ablation(args.seeds, epochs, bs, args.W_cap)
    else:
        run_task(args.task, args.seeds, epochs, bs, args.W_cap, args.mnist_n)
    print(f"[{args.task}] total wall {time.time()-t0:.1f}s", flush=True)
