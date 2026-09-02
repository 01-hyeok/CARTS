#!/bin/bash
set -uo pipefail

# Compare tau_topk settings for the arms in the sweep.
#
# eff_k = exp(H(alpha)) is the effective number of candidates the weighting keeps:
# 10 means a plain average of the retrieved set, 1 means the top-1 only. It is the
# number to read first - if it does not move, tau did nothing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASETS=("${@}")
[ "$#" -gt 0 ] || DATASETS=(ETTh1 ETTm1)

/data/pjh_workspace/ts-env/bin/python - "${DATASETS[@]}" <<'EOF'
import re, os, sys, math
DS = sys.argv[1:]
ARMS = os.environ.get('ARMS',
    'identity identity_l2 random random_l2 chronos chronos_l2 chronos_eos chronos_tsrag '
    '2stage_ema 2stage_ema_l2 2stage_mse 2stage_mse_l2 '
    'e2e_ema e2e_ema_l2 e2e_mse e2e_mse_l2').split()
TAUS = ['0.10'] + os.environ.get('TAUS', '0.05 0.01').split()
PLS  = [int(x) for x in os.environ.get('PRED_LENS', '96 192 336').split()]

def tag(t):
    return '' if abs(float(t) - 0.10) < 1e-12 else '_tauk' + t.replace('.', 'p')

def read(ds, arm, t, pl):
    f = f'logs/{ds}/self_topk/{arm}{tag(t)}_seq{pl}_pred{pl}.log'
    if not os.path.exists(f): return None
    txt = open(f, errors='ignore').read()
    if 'Stage2 Test Final' not in txt: return None
    tail = txt[txt.rindex('Stage2 Test |'):]
    def last(pat, s=tail):
        # The Stage2 Test Final block prints one metric per line; re.M is needed
        # for the ^-anchored patterns to match anywhere but the first line.
        m = re.findall(pat, s, re.M)
        return float(m[-1]) if m else float('nan')
    ent = [float(x) for x in re.findall(r'topk_weight_entropy_\w+: ([\d.]+)', tail)]
    return dict(
        mse=last(r'^final_mse: ([\d.]+)', txt),
        mae=last(r'^final_mae: ([\d.]+)', txt),
        r10=last(r'^student_relation_oracle_recall_at_10: ([\d.]+)', txt),
        eff=math.exp(sum(ent)/len(ent)) if ent else float('nan'),
        a1=last(r'\| alpha_top1_mean: ([\d.]+)'),
    )

for ds in DS:
    rows = [(a, t, pl, read(ds, a, t, pl)) for a in ARMS for pl in PLS for t in TAUS]
    if not any(r[3] for r in rows): continue
    print(f'\n===== {ds} =====')
    print(f'{"arm":<13}{"pred":>5}{"tau":>7}{"eff_k":>8}{"a_top1":>8}{"MSE":>9}{"MAE":>9}{"R@10":>9}')
    for a in ARMS:
        for pl in PLS:
            for t in TAUS:
                d = read(ds, a, t, pl)
                if d is None:
                    print(f'{a:<13}{pl:>5}{t:>7}{"미완":>8}')
                else:
                    print(f'{a:<13}{pl:>5}{t:>7}{d["eff"]:>8.2f}{d["a1"]:>8.3f}'
                          f'{d["mse"]:>9.4f}{d["mae"]:>9.4f}{d["r10"]:>9.4f}')
            print()
EOF
