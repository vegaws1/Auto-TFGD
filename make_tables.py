"""
Generate LaTeX tables/fragments from results/*.json produced by harness.py.
Outputs to results/tables/*.tex and prints a compact summary.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
OUT = os.path.join(RES, 'tables')
os.makedirs(OUT, exist_ok=True)

ORDER = ['SGD', 'Adam', 'AdamW', 'RMSprop', 'FractionalGD', 'TFGD_Recursive',
         'TFGD_Truncated', 'AutoTFGD_Recursive', 'AutoTFGD_Truncated']
ESC = lambda s: s.replace('_', r'\_')
TASK_NAME = {'lr': 'Breast Cancer Wisconsin (logistic regression)',
             'mlp': 'Breast Cancer Wisconsin (MLP)',
             'noisy': 'Noisy Digits (MLP)',
             'mnist': 'MNIST (MLP)'}


def fmt(m, s):
    return f"${m*100:.2f} \\pm {s*100:.2f}$"


def load(task):
    p = os.path.join(RES, f'{task}.json')
    return json.load(open(p)) if os.path.exists(p) else None


def main_table(task, d):
    s = d['summary']
    best = max(s, key=lambda o: s[o]['acc_mean'])
    rows = []
    for o in ORDER:
        if o not in s:
            continue
        r = s[o]
        name = ESC(o)
        if o == best:
            name = r'\textbf{%s}' % name
            acc = r'$\mathbf{%.2f \pm %.2f}$' % (r['acc_mean']*100, r['acc_std']*100)
        else:
            acc = fmt(r['acc_mean'], r['acc_std'])
        auc = f"${r['auc_mean']*100:.2f}$" if r['auc_mean'] == r['auc_mean'] else '--'
        rows.append(f"{name} & {r['lr']:g} & {acc} & {auc} & {r['time_per_epoch']*1000:.1f} \\\\")
    body = "\n".join(rows)
    cap = f"Fair-comparison results on {TASK_NAME[task]} over {len(d['seeds'])} seeds " \
          f"({d['epochs']} epochs). Learning rate tuned per optimizer on a validation split. " \
          f"Best mean accuracy in bold."
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{cap}}}
\\label{{tab:{task}_main}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Optimizer}} & \\textbf{{lr}} & \\textbf{{Test acc.\\ (\\%)}} & \\textbf{{AUC (\\%)}} & \\textbf{{ms/epoch}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def stats_table(task, d):
    ref = d['reference']
    rows = []
    for c in d['comparisons']:
        sig = 'yes' if (c.get('p_wilcoxon_holm', 1) < 0.05) else 'no'
        rows.append(
            f"{ESC(c['vs'])} & ${c['mean_diff']*100:+.2f}$ & {c['p_ttest']:.3f} & "
            f"{c['p_wilcoxon']:.3f} & {c.get('p_wilcoxon_holm',float('nan')):.3f} & "
            f"${c['cohen_dz']:.2f}$ & ${c['cliffs_delta']:.2f}$ & {sig} \\\\")
    body = "\n".join(rows)
    cap = (f"Paired significance tests on {TASK_NAME[task]}: {ESC(ref)} versus each baseline "
           f"(seed-matched accuracies, {len(d['seeds'])} seeds). $\\Delta$ is the mean accuracy "
           f"difference (ref$-$other, \\%). Holm correction across all comparisons; effect sizes "
           f"are paired Cohen's $d_z$ and Cliff's $\\delta$.")
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{cap}}}
\\label{{tab:{task}_stats}}
\\small
\\begin{{tabular}}{{lccccccc}}
\\hline
\\textbf{{vs.\\ baseline}} & $\\Delta$\\% & $p_{{t}}$ & $p_{{W}}$ & $p_{{W}}^{{\\text{{Holm}}}}$ & $d_z$ & $\\delta$ & sig. \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def cost_table(task, d):
    s = d['summary']
    base = s['SGD']['time_per_epoch'] if 'SGD' in s else 1.0
    rows = []
    for o in ORDER:
        if o not in s:
            continue
        r = s[o]
        rel = r['time_per_epoch'] / base if base else float('nan')
        ad = '' if r.get('alpha_final') is None else f"{r['alpha_final']:.3f}"
        ld = '' if r.get('lam_final') is None else f"{r['lam_final']:.3f}"
        rows.append(f"{ESC(o)} & {r['time_per_epoch']*1000:.1f} & {rel:.2f} & {ad or '--'} & {ld or '--'} \\\\")
    body = "\n".join(rows)
    cap = (f"Computational cost on {TASK_NAME[task]} (median ms/epoch, CPU) and the converged "
           f"adaptive parameters for the proposed methods. Relative cost is normalized to SGD.")
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{cap}}}
\\label{{tab:{task}_cost}}
\\begin{{tabular}}{{lcccc}}
\\hline
\\textbf{{Optimizer}} & \\textbf{{ms/epoch}} & \\textbf{{rel.\\ SGD}} & \\textbf{{$\\alpha_\\infty$}} & \\textbf{{$\\lambda_\\infty$}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


def ablation_table(d):
    r = d['result']
    rows = []
    for name, v in r.items():
        rows.append(f"{name} & {fmt(v['mean'], v['std'])} & [${v['ci'][0]*100:.2f}, {v['ci'][1]*100:.2f}$] \\\\")
    body = "\n".join(rows)
    cap = (f"Ablation of the adaptation mechanism on Noisy Digits ({len(d['seeds'])} seeds, recursive "
           f"variant, shared tuned learning rate {d['lr']:g}). Each row toggles which signals drive "
           f"$(\\alpha_k,\\lambda_k)$; \"Random adaptation\" ignores the gradient statistics.")
    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{{cap}}}
\\label{{tab:ablation}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Variant}} & \\textbf{{Test acc.\\ (\\%)}} & \\textbf{{95\\% CI}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}
"""


if __name__ == '__main__':
    for task in ['lr', 'mlp', 'noisy', 'mnist']:
        d = load(task)
        if not d:
            print(f"[skip] {task} (no results yet)")
            continue
        with open(os.path.join(OUT, f'{task}_main.tex'), 'w') as f:
            f.write(main_table(task, d))
        with open(os.path.join(OUT, f'{task}_stats.tex'), 'w') as f:
            f.write(stats_table(task, d))
        with open(os.path.join(OUT, f'{task}_cost.tex'), 'w') as f:
            f.write(cost_table(task, d))
        print(f"[ok] {task}: wrote main/stats/cost tables")
    ab = load('ablation')
    if ab:
        with open(os.path.join(OUT, 'ablation.tex'), 'w') as f:
            f.write(ablation_table(ab))
        print("[ok] ablation table")
    print("tables ->", OUT)
