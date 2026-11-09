"""CPU summary after all twelve frozen-value Oracle supplement gates."""
import csv,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data/pjh_workspace/CARTS');OUT=ROOT/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-ORACLE-K-SUPPLEMENT01'
OLD=ROOT/'experiments/p100_head_retrieval_quality_legacy_pool02'
sys.path.insert(0,str(OLD));import runtime as rt

def main():
    cells=[f'{d}_{h}' for d in ['ETTh1','Weather','Solar'] for h in [96,192,336,720]]
    rows=[];gates=[]
    for cell in cells:
        gate=json.loads((OUT/cell/'COMPLETE.json').read_text());assert gate['status']=='COMPLETE'
        assert gate['gates']['val']['saved_k10_metrics_exact'] and gate['gates']['test']['saved_k10_metrics_exact']
        with (OUT/cell/'ORACLE_K.csv').open() as f: rr=list(csv.DictReader(f))
        assert len(rr)==8 and {(r['split'],int(r['k'])) for r in rr}=={(s,k) for s in ['val','test'] for k in [1,5,10,20]}
        rows+=rr;gates.append(gate)
    with (OUT/'ORACLE_K_ALL_CELLS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    manifest=json.loads((OLD/'configs/run_manifest.json').read_text())
    for metric in ['retmse','agg_mse','agg_mae']:
        fig,axes=plt.subplots(3,4,figsize=(14,9))
        for ax,cell in zip(axes.flat,cells):
            for model in ['V0','V1','V2','V5']:
                run=next(r for r in manifest if r['run_id']==f'{cell}_{model}_seed0')
                met=json.loads((ROOT/run['output_path']/'test_metrics.json').read_text())
                yy=[next(r for r in met['rows'] if r['name']==('All' if k==10 else f'All_K{k}'))[metric] for k in [1,5,10,20]]
                ax.plot([1,5,10,20],yy,'o-',label=model)
            oracle=[next(r for r in rows if r['cell']==cell and r['split']=='test' and int(r['k'])==k) for k in [1,5,10,20]]
            ax.plot([1,5,10,20],[float(r[metric]) for r in oracle],'kD--',label='Individual-MSE Oracle (observed)')
            if metric!='retmse':ax.axhline(met['last_mse' if metric=='agg_mse' else 'last_mae'],color='gray',ls=':',label='Last-repeat')
            ax.set(title=cell,xlabel='K (official K=10)',ylabel=metric+'; lower is better',xticks=[1,5,10,20])
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=3)
        fig.tight_layout();fig.savefig(OUT/f'K_{metric}_ORACLE_COMPLETE.png',dpi=170,bbox_inches='tight');plt.close(fig)
    lines=['# K별 Oracle 보완 결과','12셀 × val/test × K1/5/10/20, 96행을 원본 frozen candidate future에서 재평가했다. 모델 학습과 Stage2 실행은 없다.',
        '기존 K10 per-query/channel 지표 전체와 exact 일치를 확인했다. K1/5/20은 각 K의 Individual-MSE 최저 후보를 새로 선택한 Oracle 진단이다. AggMSE/MAE set-optimal Oracle이 아니다.',
        '|Cell|K|Test Oracle RetMSE|Test Oracle AggMSE|Test Oracle AggMAE|','|---|---:|---:|---:|---:|']
    for r in rows:
        if r['split']=='test':lines.append('|'+r['cell']+'|'+r['k']+'|'+'|'.join(f'{float(r[k]):.6f}' for k in ['retmse','agg_mse','agg_mae'])+'|')
    lines+=['','원본 48개 결과는 변경하지 않았다. 후속 failure-attribution 및 Solar Full8 완료 여부와 별개다.']
    (OUT/'ORACLE_K_REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    rt.write_json(OUT/'FINAL_GATE.json',dict(status='PASS',cells=12,rows=96,at=rt.now(),all_saved_k10_exact=True,
        stage1_training=False,stage2=False,completed_inputs=[str(OUT/c/'COMPLETE.json') for c in cells]))
    print('ORACLE SUPPLEMENT ALL 12 PASS',flush=True)

if __name__=='__main__':main()
