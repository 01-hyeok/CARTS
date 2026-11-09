import sys,json,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_summary',HERE.parent/'summarize.py');mod=importlib.util.module_from_spec(spec)
sys.path.insert(0,str(HERE.parent));spec.loader.exec_module(mod)
mod.main()
note='''

## Canonical score 교정 기록

ETTh1 H192 Full의 마지막 B1에서 기존 all-channel 경로와 직접 공식 경로의 E0 점수 차이를 발견해 중단했습니다(최대 1.6617123037576675e-6, Top10 ID 차이 0). 허용오차를 늘리지 않고 canonical 5-row head normalization과 all-channel batched GEMM을 사용했습니다. H192/336/720 Full B1에 한정되며 각 조건 양 split의 직접 공식 raw cosine 및 E0 점수 bitwise 일치 gate를 통과해야 진행합니다. 이 범위의 과거 raw trace 점수는 exact 근거로 사용하지 않으며, 기존 Top10 ID·후보 MSE는 그대로 대조합니다. COMPLETE.json의 canonical_tail_score_recovery와 개별 GPU_SMOKE_GATE.json이 적용 범위를 명시합니다. H96 완료 결과와 원본 소스는 보존했습니다.
'''
for name in ('FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md'):
    p=mod.OUT/name;p.write_text(p.read_text()+note)
p=mod.OUT/'FINAL_GATE.json';g=json.loads(p.read_text())
for name in ('FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md'):g['files'][name]=mod.sha(mod.OUT/name)
mod.dump(p,g)
