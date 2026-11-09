"""Read-only immutable completed snapshot audit; writes only a NEW local directory.

Does not imply the remote server currently has only these five complete cells.
Never substitutes P100_CONTROLLED for P100_NATIVE.
"""
import argparse,csv,hashlib,io,json
from pathlib import Path
import zipfile

EXPECTED='9cabe23448ea8602d33599d6f4a902853829fa7d33dcec4a97dc3f1782ec280f'
CELLS=['ETTh1_96','ETTh1_192','ETTh1_336','ETTh1_720','Weather_96']


def digest(raw):return hashlib.sha256(raw).hexdigest()


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def run(archive,out):
    archive,out=Path(archive),Path(out)
    assert digest(archive.read_bytes())==EXPECTED,'source export archive changed'
    assert not out.exists(),'never overwrite a prior audit snapshot'
    files={};inventory=[];normalizers=[];d1=[];mass=[];loo=[];gates=[]
    with zipfile.ZipFile(archive) as z:
        for cell in CELLS:
            pre='cells/'+cell+'/'
            gate=json.loads(z.read(pre+'CELL_ARTIFACT_GATE.json'))
            assert gate['status']=='CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING'
            gates.append(dict(cell=cell,status=gate['status'],at=gate['at']))
            for name in ['HEAD_UNIQUE_RELEVANCE.csv','HEAD_UTILITY.csv','NORMALIZER_CONTROL.csv','HEAD_PROBABILITY_MASS.csv','CROSS_SUPPORT.csv']:
                raw=z.read(pre+name);hashval=digest(raw)
                assert gate['files'][name]==hashval,(cell,name)
                rows=list(csv.DictReader(io.StringIO(raw.decode('utf8'))))
                files[pre+name]=raw
                for row in rows:
                    if name=='HEAD_UNIQUE_RELEVANCE.csv':
                        d1.append(dict(**row,near_threshold_definition='LEGACY: d10 + max(1e-8,.01*median_validation_d10_per_channel); not new query-relative .01/.05/.10',reuse='EXISTS_VALID_SNAPSHOT'))
                    elif name=='HEAD_PROBABILITY_MASS.csv':mass.append(row)
                    elif name=='HEAD_UTILITY.csv' and row['method'].startswith('WithoutH'):
                        loo.append(dict(**row,requested_delta_all5_minus_without=-float(row['delta_ret_vs_all5']),definition='positive: removal improves RetMSE; fixed checkpoint intervention'))
                inventory.append(dict(cell=cell,file=name,rows=len(rows),sha256=hashval,
                  status='EXISTS_VALID_SNAPSHOT',gate_at=gate['at'],source_archive=str(archive.resolve()),source_member=pre+name,
                  current_remote_status='UNCONFIRMED_SSH_TIMEOUT',native_p100_head_detail='NOT_IN_THIS_CACHE' if name.startswith('HEAD_') else 'SEPARATE_VARIANT'))
            cross=list(csv.DictReader(io.StringIO(files[pre+'CROSS_SUPPORT.csv'].decode('utf8'))))
            for support in ['P100','FULL']:
                for model in ['V0','V1','V5']:
                    pick=lambda v:next(r for r in cross if r['training_support']==support and r['model']==model and r['variant']==v)
                    a,b,c=pick('P100_LOCAL_NORMALIZER'),pick('P100_FULL_NORMALIZER'),pick('FULL')
                    assert a['checkpoint_sha256']==b['checkpoint_sha256']==c['checkpoint_sha256']
                    ra,rb,rc=map(lambda r:float(r['retmse']),[a,b,c])
                    assert abs((rb-ra)+(rc-rb)-(rc-ra))<1e-12
                    normalizers.append(dict(cell=cell,model=model,training_support=support,checkpoint_sha256=a['checkpoint_sha256'],
                      queries=a['queries'],channels=a['channels'],A_variant=a['variant'],B_variant=b['variant'],C_variant=c['variant'],
                      R_A=a['retmse'],R_B=b['retmse'],R_C=c['retmse'],normalization_effect=rb-ra,candidate_expansion_effect=rc-rb,
                      total_path_change=rc-ra,scope='aggregate exact path decomposition; per-query gate retained upstream; not causal decomposition',
                      pending='head mass variance, per-query rank Spearman and sensitivity tails require raw blocks'))
    out.mkdir(parents=True)
    for name,raw in files.items():
        p=out/'reused_sources'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    write_csv(out/'PHASE0_ARTIFACT_INVENTORY.csv',inventory)
    write_csv(out/'D1_LEGACY_HEAD_UNIQUE_REUSE.csv',d1)
    write_csv(out/'D2_HEAD_LEAVE_ONE_OUT_REUSED.csv',loo)
    write_csv(out/'D3_NORMALIZER_ONLY_RESULTS.csv',normalizers)
    write_csv(out/'D3_HEAD_PROBABILITY_MASS.csv',mass)
    status=dict(status='LOCAL_SNAPSHOT_REUSE_PASS_NOT_REMOTE_COMPLETION',archive_sha256=EXPECTED,cells=CELLS,
                source_files_verified=len(files),D1_rows=len(d1),D2_LOO_rows=len(loo),D3_path_rows=len(normalizers),D3_mass_rows=len(mass),
                gates=gates,remote_refreshed=False,gpu_run=False,originals_modified=False,
                remaining=['current remote cells/checkpoints/gates','D1 query-relative epsilon and native head gaps','D2 exact loss and lost-candidate rank/probabilities','D3 raw query sensitivities','D4 implementation/runtime integration','SoftLag cache/teacher/GPU/runtime integration'])
    (out/'STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.archive,a.out)
