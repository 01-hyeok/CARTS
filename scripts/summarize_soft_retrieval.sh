#!/bin/bash
set -uo pipefail

# Soft-retrieval results next to the Top-K baselines they replace.
#
# eff_k is the number to read first: exp(H(alpha)) over the whole bank in soft
# mode, over the k selected entries in Top-K mode. If soft mode lands in the
# hundreds the model is averaging most of the bank and MSE should be worse; if it
# lands near 10-20 it is doing what Top-K did, but differentiably.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(cd "${SCRIPT_DIR}/.." && pwd)"

/data/pjh_workspace/ts-env/bin/python - "$@" <<'EOF'
import re, os, sys, math
DS = sys.argv[1:] or ['ETTh1', 'ETTm1']
PLS = [int(x) for x in os.environ.get('PRED_LENS', '96 192 336').split()]
TAUS = os.environ.get('TAUS', '0.01 0.05 0.10').split()
LAM = os.environ.get('LAMBDA', '0.0')

def read(path):
    if not os.path.exists(path): return None
    t = open(path, errors='ignore').read()
    if 'Stage2 Test Final' not in t: return None
    def last(pat, s=t, flags=re.M):
        m = re.findall(pat, s, flags)
        return float(m[-1]) if m else float('nan')
    tail = t[t.rindex('Stage2 Test |'):] if 'Stage2 Test |' in t else ''
    eff = re.findall(r'\| top_k_effective: ([\d.]+)', tail)
    ent = [float(x) for x in re.findall(r'topk_weight_entropy_\w+: ([\d.]+)', tail)]
    return dict(
        mse=last(r'^final_mse: ([\d.]+)'), mae=last(r'^final_mae: ([\d.]+)'),
        r10=last(r'^student_relation_oracle_recall_at_10: ([\d.]+)'),
        ret=float(re.findall(r'\| ret_mse: ([\d.]+)', tail)[-1]) if 'ret_mse' in tail else float('nan'),
        eff=float(eff[-1]) if eff else float('nan'),
        ent=math.exp(sum(ent)/len(ent)) if ent else float('nan'),
    )

for ds in DS:
    print(f'\n===== {ds} =====')
    print(f'{"setting":<22}{"pred":>5}{"eff_k":>9}{"MSE":>9}{"MAE":>9}{"R@10":>9}{"ret_mse":>10}')
    for pl in PLS:
        for name, path in (
            [('topk  e2e_mse (기존)', f'logs/{ds}/self_topk/e2e_mse_seq{pl}_pred{pl}.log')] +
            [(f'soft  tau={t} lam={LAM}',
              f'logs/{ds}/soft_retrieval/tau{t.replace(".","p")}_lam{LAM.replace(".","p")}_seq{pl}_pred{pl}.log')
             for t in TAUS]
        ):
            d = read(path)
            if d is None:
                print(f'{name:<22}{pl:>5}{"미완":>9}')
            else:
                effk = d['eff'] if not math.isnan(d['eff']) else d['ent']
                print(f'{name:<22}{pl:>5}{effk:>9.1f}{d["mse"]:>9.4f}{d["mae"]:>9.4f}'
                      f'{d["r10"]:>9.4f}{d["ret"]:>10.4f}')
        print()
EOF
