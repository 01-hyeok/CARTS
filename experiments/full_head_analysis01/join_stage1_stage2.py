import csv,json,sys
from pathlib import Path
p=Path('/data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01')/sys.argv[1]
stage1={r['subset']:r for r in json.loads((p/'test_metrics.json').read_text())}
stage2=list(csv.DictReader((p/'HEAD_STAGE2_MSE_MAE.csv').open()))
raw=list(csv.DictReader((p/'CANONICAL_COMPARATORS_RAW.csv').open()))
base=next(r for r in raw if r['model']=='Base')
all5=next(r for r in stage2 if r['subset']=='12345')
out=[]
for r in stage2:
 a=stage1[r['subset']]
 row=dict(cell=sys.argv[1],subset=r['subset'],seed=0,split='test',stage1_aggmse=a['agg_mse10'],stage1_aggmae=a['agg_mae10'])
 for k in ['primary_mse','primary_mae','primary_lambda','secondary_mse','secondary_mae']:row[k]=float(r[k])
 row['primary_mse_improvement_pct_vs_base']=100*(float(base['test_mse'])-float(r['primary_mse']))/float(base['test_mse'])
 row['primary_mae_improvement_pct_vs_base']=100*(float(base['test_mae'])-float(r['primary_mae']))/float(base['test_mae'])
 row['primary_mse_delta_vs_all5']=float(r['primary_mse'])-float(all5['primary_mse'])
 row['primary_mae_delta_vs_all5']=float(r['primary_mae'])-float(all5['primary_mae'])
 out.append(row)
with (p/'HEAD_STAGE1_STAGE2_JOINED.csv').open('w') as f:
 w=csv.DictWriter(f,list(out[0]));w.writeheader();w.writerows(out)
keys=['subset','stage1_aggmse','stage1_aggmae','primary_mse','primary_mae','primary_lambda','primary_mse_improvement_pct_vs_base','primary_mae_improvement_pct_vs_base']
t='# Stage1 및 Stage2 연결 비교\n\nSeed0 test. 개선율은 Base 대비이며 양수가 개선, 음수가 악화다. 12345는 All5, 나머지 문자열은 사용한 head 번호다. 단일 seed 사후 진단이며 이 표로 test 최적 구성을 배포 선택하지 않는다.\n\n'
t+='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
for r in out:t+='| '+' | '.join(str(r[k]) if k=='subset' else f'{r[k]:.6f}' for k in keys)+' |\n'
(p/'HEAD_STAGE1_STAGE2_JOINED.md').write_text(t)
print(p/'HEAD_STAGE1_STAGE2_JOINED.csv')
