"""
Regenerate ALL manuscript figures from the new (fair-tuned) runs, writing PNGs
with the exact filenames the LaTeX already references (in ../extracted/).
Single representative seed (42); per-optimizer tuned learning rates loaded from
results/<task>.json so the curves are consistent with the result tables.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

import autotfgd_lib as L
import harness as H

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'extracted'))
SUFFIX = {'lr': 'LR', 'mlp': 'MLP', 'noisy': 'noisy digits', 'mnist': 'MNIST'}
CFG = {'lr': dict(epochs=150, bs=32, W=50, mnist_n=0),
       'mlp': dict(epochs=150, bs=32, W=50, mnist_n=0),
       'noisy': dict(epochs=100, bs=64, W=30, mnist_n=0),
       'mnist': dict(epochs=25, bs=128, W=12, mnist_n=15000)}
PALETTE = {'SGD': 'tab:gray', 'Adam': 'tab:blue', 'AdamW': 'tab:cyan',
           'RMSprop': 'tab:green', 'FractionalGD': 'tab:orange',
           'TFGD_Recursive': 'tab:purple', 'TFGD_Truncated': 'tab:brown',
           'AutoTFGD_Recursive': 'tab:red', 'AutoTFGD_Truncated': 'tab:pink'}


def run_logged(task, opt_name, lr, meta, full, cfg):
    Xtr, ytr, Xte, yte = full
    L.set_seed(42)
    model = H.model_factory(meta)()
    opt = L.make_optimizer(opt_name, model.parameters(), lr, seed=42, W_cap=cfg['W'])
    mc = meta['multiclass']
    lossf = nn.CrossEntropyLoss() if mc else nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long if mc else torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    rng = np.random.default_rng(42)
    n = len(ytr); bs = cfg['bs']
    log = dict(trL=[], teL=[], trA=[], teA=[], teAUC=[])
    for _ in range(cfg['epochs']):
        model.train()
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            bi = torch.tensor(idx[i:i + bs])
            opt.zero_grad(); out = model(Xtr_t[bi]); loss = lossf(out, ytr_t[bi])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            otr = model(Xtr_t); ote = model(Xte_t)
            log['trL'].append(float(lossf(otr, ytr_t)))
            log['teL'].append(float(lossf(ote, yte if False else torch.tensor(yte, dtype=torch.long if mc else torch.float32))))
            if mc:
                ptr = torch.softmax(otr, 1).numpy(); pte = torch.softmax(ote, 1).numpy()
                log['trA'].append(accuracy_score(ytr, ptr.argmax(1)))
                log['teA'].append(accuracy_score(yte, pte.argmax(1)))
                try: log['teAUC'].append(roc_auc_score(yte, pte, multi_class='ovr', average='macro'))
                except Exception: log['teAUC'].append(float('nan'))
            else:
                ptr = torch.sigmoid(otr).numpy(); pte = torch.sigmoid(ote).numpy()
                log['trA'].append(accuracy_score(ytr, (ptr >= .5)))
                log['teA'].append(accuracy_score(yte, (pte >= .5)))
                try: log['teAUC'].append(roc_auc_score(yte, pte))
                except Exception: log['teAUC'].append(float('nan'))
    traj = None
    if hasattr(opt, 'alpha_hist') and opt.alpha_hist:
        spe = max(1, math.ceil(n / bs))
        x = np.arange(len(opt.alpha_hist)) / spe
        traj = dict(x=x, alpha=np.array(opt.alpha_hist), lam=np.array(opt.lam_hist))
    return log, traj


def save_curve(logs, key, ylabel, fname, title):
    plt.figure(figsize=(6, 4))
    for opt, lg in logs.items():
        y = lg[key]
        plt.plot(range(1, len(y) + 1), y, label=opt.replace('_', ' '),
                 color=PALETTE.get(opt), lw=1.6)
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title)
    plt.legend(fontsize=6, ncol=2); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()


def save_traj(traj, comp, fname, ylabel, title):
    plt.figure(figsize=(6, 4))
    plt.plot(traj['x'], traj[comp], color='tab:red', lw=1.6)
    plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.title(title)
    plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()


def main():
    for task, suf in SUFFIX.items():
        cfg = CFG[task]
        meta = H.get_dataset(task, mnist_n=cfg['mnist_n'])[2]
        full = H.get_dataset(task, mnist_n=cfg['mnist_n'])[0]
        tuned = json.load(open(os.path.join(H.RESDIR, f'{task}.json')))['tuned_lr']
        logs, trajs = {}, {}
        for opt in L.ALL_OPT:
            lg, tr = run_logged(task, opt, tuned[opt], meta, full, cfg)
            logs[opt] = lg
            if tr is not None:
                trajs[opt] = tr
            print(f"[fig:{task}] {opt} done", flush=True)
        # curve figures (only those the manuscript references)
        need = {
            'lr': [('teL', 'Test loss', f'test_loss {suf}.png', 'Test loss (LR)')],
            'mlp': [('trL', 'Train loss', f'train_loss {suf}.png', 'Train loss (MLP)'),
                    ('teL', 'Test loss', f'test_loss {suf}.png', 'Test loss (MLP)')],
            'mnist': [('trA', 'Train accuracy', f'train_accuracy {suf}.png', 'Train accuracy (MNIST)'),
                      ('teA', 'Test accuracy', f'test_accuracy {suf}.png', 'Test accuracy (MNIST)'),
                      ('teL', 'Test loss', f'test_loss {suf}.png', 'Test loss (MNIST)'),
                      ('teAUC', 'Test AUC', f'auc_learning_curves {suf}.png', 'AUC learning curves (MNIST)')],
            'noisy': [('trA', 'Train accuracy', f'train_accuracy {suf}.png', 'Train accuracy (Noisy Digits)'),
                      ('trL', 'Train loss', f'train_loss {suf}.png', 'Train loss (Noisy Digits)'),
                      ('teAUC', 'Test AUC', f'auc_learning_curves {suf}.png', 'AUC learning curves (Noisy Digits)')],
        }[task]
        for key, yl, fn, ti in need:
            save_curve(logs, key, yl, fn, ti)
        # adaptation-trajectory figures
        tr_t = trajs.get('AutoTFGD_Truncated'); tr_r = trajs.get('AutoTFGD_Recursive')
        if tr_t is not None:
            save_traj(tr_t, 'alpha', f'autotfgd_alpha {suf}.png', r'$\alpha_k$', f'AutoTFGD-Truncated $\\alpha$ ({suf})')
            save_traj(tr_t, 'lam', f'autotfgd_lambda {suf}.png', r'$\lambda_k$', f'AutoTFGD-Truncated $\\lambda$ ({suf})')
        if tr_r is not None:
            save_traj(tr_r, 'alpha', f'AutoTFGD_recursive_alpha {suf}.png', r'$\alpha_k$', f'AutoTFGD-Recursive $\\alpha$ ({suf})')
            save_traj(tr_r, 'lam', f'AutoTFGD_recursive_lambda {suf}.png', r'$\lambda_k$', f'AutoTFGD-Recursive $\\lambda$ ({suf})')
        print(f"[fig:{task}] figures written", flush=True)


if __name__ == '__main__':
    main()
