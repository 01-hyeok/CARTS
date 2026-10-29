#!/usr/bin/env python3
"""Durable sequential-cell controller; only this track runs one GPU child at a time."""
import csv, fcntl, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/TRACK-V5-FULL-HEAD-ANALYSIS01'
LOG=ROOT/'logs/TRACK-V5-FULL-HEAD-ANALYSIS01'
GPU='GPU-91418a73-202a-2a9b-57d5-2538b19e07ec'
CELLS=['ETTh1_96','ETTh1_720','Weather_96','Weather_720']
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
def table(rows,keys):
    text='| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'
    for row in rows:text+='| '+' | '.join(str(row.get(k,'')) for k in keys)+' |\n'
    return text
def aggregate():
    names=['HEAD_QUALITY','HEAD_OVERLAP','HEAD_UNION_COVERAGE','HEAD_ERROR_COMPLEMENTARITY','HEAD_LEAVE_ONE_OUT','HEAD_SUBSET_ANALYSIS','HEAD_STAGE2_MSE_MAE','HEAD_COUNT','INDEPENDENT_MODELS']
    merged={}
    for name in names:
        rows=[]
        for cell in CELLS:
            p=OUT/cell/f'{name}.csv'
            if p.exists():rows+=list(csv.DictReader(open(p)))
        merged[name]=rows
        if rows:
            keys=list(dict.fromkeys(k for row in rows for k in row))
            with open(OUT/f'{name}.csv','w',newline='') as f:
                w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    provenance={c:json.loads((OUT/c/'CHECKPOINT_PROVENANCE.json').read_text()) for c in CELLS if (OUT/c/'CHECKPOINT_PROVENANCE.json').exists()}
    dump(OUT/'CHECKPOINT_PROVENANCE.json',provenance)
    repro={c:json.loads((OUT/c/'baseline_reproduction.json').read_text()) for c in CELLS if (OUT/c/'baseline_reproduction.json').exists()}
    (OUT/'BASELINE_REPRODUCTION.md').write_text('# 기존 Full V5 재현\n\n'+table([dict(cell=c,status=r['status'],stage2_mse=r['stage2']['mse'],stage2_mae=r['stage2']['mae']) for c,r in repro.items()],['cell','status','stage2_mse','stage2_mae'])+'\n저장된 원래 lambda로 forward 검증. 신규 All5 캐시 일치 검증은 셀별 newcache_stored_lambda_verification.json.\n')
    complete=[c for c in CELLS if (OUT/c/'COMPLETE.json').exists()]
    text='# Full Candidate Multi-Head Retrieval 분석\n\n상태: '+('COMPLETED' if len(complete)==4 else 'PARTIAL')+'; seed0만 실행.\n\n'
    text+='## 연구 질문과 방법\n기존 Full V5의 head를 제거하거나 결합할 때 실제 Stage2 MSE/MAE가 어떻게 바뀌는가? Encoder와 projection은 동결하고, 원래 full train memory/RAFT mask를 유지했다. 31개 subset은 validation/test에서 평가하며 11개 단일·제거·All5 구성은 기존 global lambda를 train에서 적합하고 validation으로 선택했다. Secondary는 기존 All5 lambda를 고정했다.\n\n'
    text+='## 독립 학습 V0/V1/V2/V5 및 Base\n'+table(merged['INDEPENDENT_MODELS'],['cell','model','status','test_mse','test_mae','source_kind'])+'\nV0→V1에는 query projection 추가가 포함된다. 이 표는 과거 결과를 동일 base/split forward로 검증해 재사용한 것이며 새 학습 결과가 아니다. V5 내부 subset은 독립 학습 V1/V2와 같지 않다.\n\n'
    text+='## 실제 head의 Stage2 기여\n'+table(merged['HEAD_STAGE2_MSE_MAE'],['cell','subset','primary_mse','primary_mae','primary_lambda','secondary_mse','secondary_mae'])+'\nsubset은 사용한 head 번호다. 12345는 All5. Primary는 lambda 재적합, Secondary는 canonical All5 lambda 고정. Test상 최선 구성을 배포 선택으로 취급하지 않는다.\n\n'
    for c in complete:
        rows=[r for r in merged['HEAD_STAGE2_MSE_MAE'] if r['cell']==c];all5=next(r for r in rows if r['subset']=='12345')
        better=[r['subset'] for r in rows if float(r['primary_mse'])<float(all5['primary_mse'])]
        text+=f'### {c}\n관찰: All5 MSE={float(all5["primary_mse"]):.6f}, MAE={float(all5["primary_mae"]):.6f}. All5보다 test MSE가 낮은 진단 구성: {better}. 이는 사후 진단이며 validation-only 배포 선택 및 seed 반복 검증 없이 최적 구성이라고 확정하지 않는다.\n'
        selection=json.loads((OUT/c/'subset_selection.json').read_text())
        text+=f'Validation 선택 subset={selection["val_selected_subset"]}; 이 subset의 test AggMSE={selection["test_selected_subset"]["agg_mse10"]:.6f}. Test future oracle AggMSE={selection["oracle_test_aggmse"]:.6f}, 비배포 upper bound.\n\n'
    text+='## 다양성·오류·기여의 구분\nHEAD_OVERLAP/HEAD_UNION_COVERAGE는 후보 발견, HEAD_SUBSET_ANALYSIS는 선택 품질, HEAD_STAGE2_MSE_MAE는 최종 예측 기여를 측정한다. Head error cosine과 선택 후보 집합의 D/C는 다른 단위다. D/C는 독립 pairwise raw dot로 검증했다. 낮은 overlap만으로 오류 상쇄나 의미적 전문화를 입증하지 않는다.\n\n'
    text+='## 연구 판단 및 다음 실험\n현재 seed0의 조건부 ablation 증거이며 의미적 역할의 인과적 증명은 NOT ESTABLISHED. 최종 분류는 결과 해석 완료 전까지 PENDING으로 두며 셀별 primary/secondary와 MSE/MAE의 방향을 함께 읽는다. 새로운 loss/architecture는 추가하지 않았다. 가장 작은 후속 작업은 validation에서 고정한 subset 정책의 독립 seed 검증이지만 사용자 지시에 따라 seed1/2는 미실행이다.\n\n'
    text+='## 발표 한 줄 결론\nHead별 검색 다양성과 최종 forecasting 기여를 구분하고, 동일한 Full checkpoint에서 head 제거와 lambda 고정/재적합 대조로 기여를 측정했다.\n\n## 발표 스크립트\n이번 실험에서는 새 retriever를 학습하지 않고 기존 Full V5의 다섯 head를 하나씩 제거하거나 단독으로 사용했습니다. 각 구성은 확률을 다시 평균한 뒤 Top10을 재선택했으며, 최종 MSE와 MAE를 우선 비교했습니다. Lambda를 다시 적합한 결과와 고정한 결과를 나누어 gate 변화의 영향을 분리했습니다. 후보가 다양하다는 사실만으로 head가 도움이 된다고 판단하지 않고, 표에 제시된 Stage2 변화로 조건부 기여를 해석해야 합니다.\n\n'
    text+='## 상태와 제한\n'+table([dict(cell=c,status='Completed' if c in complete else 'Not complete') for c in CELLS],['cell','status'])+'\n실행 시간은 두 트랙이 동일 GPU1을 공유하는 경합 환경으로 독립 속도 비교가 아니다. 대용량 cache/checkpoint는 git에 넣지 않았다. GitHub 업로드는 controller가 수행하지 않는다.\n'
    (OUT/'FINAL_REPORT.md').write_text(text)
def main():
    assert sys.executable == '/usr/bin/python3', 'Use verified historical /usr/bin/python3 runtime'
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=open(OUT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['CUDA_VISIBLE_DEVICES']=GPU
    mapping=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv'],text=True)
    assert any(line.split(',')[0].strip()=='1' and line.split(',')[1].strip()==GPU for line in mapping.splitlines()[1:])
    (OUT/'gpu_at_launch.csv').write_text(mapping)
    dump(OUT/'run_manifest.json',dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_uuid=GPU,physical_gpu=1,cells=CELLS,seeds=[0],concurrent_gpu_timing=True,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    for cell in CELLS:
        if (OUT/cell/'COMPLETE.json').exists():continue
        success=False
        for attempt,(batch,chunk) in enumerate([(32,0),(16,0),(8,1024)],1):
            log=LOG/f'{cell}_attempt{attempt}.log'
            dump(OUT/'STATUS.json',dict(status='RUNNING',cell=cell,attempt=attempt,batch=batch,chunk=chunk,log=str(log)))
            with open(log,'w') as f:
                p=subprocess.Popen([sys.executable,'-u','experiments/full_head_analysis01/full_analysis.py','--cell',cell,'--batch',str(batch),'--chunk',str(chunk)],stdout=f,stderr=subprocess.STDOUT)
                dump(OUT/'active_process.json',dict(controller_pid=os.getpid(),worker_pid=p.pid,cell=cell,gpu_uuid=GPU));code=p.wait()
            if code==0:success=True;break
            tail=log.read_text()[-12000:]
            if 'out of memory' not in tail.lower():break
            failroot=OUT/'failed_attempts';failroot.mkdir(exist_ok=True)
            dest=failroot/f'{cell}_{int(time.time())}_attempt{attempt}'
            (OUT/cell).rename(dest)
            print(f'[OOM RECOVERY] archived {dest}; reduce eval batch/chunk; FULL support unchanged',flush=True)
        aggregate()
        if not success:
            dump(OUT/'STATUS.json',dict(status='FAILED',cell=cell,log=str(log),returncode=code));raise SystemExit(code or 1)
        print('[PIPELINE CELL COMPLETED]',cell,flush=True)
    dump(OUT/'STATUS.json',dict(status='COMPLETED',cells=CELLS,seeds=[0]));aggregate()
if __name__=='__main__':main()
