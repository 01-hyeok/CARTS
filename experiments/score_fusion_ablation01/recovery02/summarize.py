import sys,runpy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
runpy.run_path(str(HERE.parent/'recovery01/summarize.py'),run_name='__main__')
from fusion_config import OUT
import hashlib
note='\n\nWeather Full V5의 짧은 query batch에서는 encoder L2 정규화 합산 순서를 원래 B행에 맞추었습니다. 네 horizon 양 split의 raw cosine·E0 확률이 직접 공식 경로와 bitwise 일치하고, 기존 E0 및 head Top10 ID 차이 0임을 확인했습니다. recovery02/*_NORM_PROBE.json과 개별 GPU_SMOKE_GATE.json에 근거를 보존했습니다. 허용오차 완화나 채널 순차 fallback은 없습니다.\n'
for n in ['FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md']:
 p=OUT/n;p.write_text(p.read_text()+note)
p=OUT/'FINAL_GATE.json';g=json.loads(p.read_text())
for n in ['FINAL_REPORT.md','CHATGPT_ANALYSIS_REPORT.md']:g['files'][n]=hashlib.sha256((OUT/n).read_bytes()).hexdigest()
p.write_text(json.dumps(g,indent=2))
