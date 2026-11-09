"""CPU-only immutable inventory and comparator coverage verification."""
import csv,json,hashlib,datetime,sys
from pathlib import Path
import numpy as np
import torch
from fusion_config import ROOT,HERE,OUT,POP,CELLS,PROTOCOL

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
def audit():
    rows=list(csv.DictReader((ROOT/'results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/00_Audit/CHECKPOINT_AUDIT.csv').open()))
    v5=[];comparators=[]
    for cell in CELLS:
        for training in ('FULL','P100'):
            gatepath=POP/cell/('V5_'+training)/'GATE.json';g=json.loads(gatepath.read_text());ident=g['identity'];data=ident['data']
            row=next(r for r in rows if r['cell']==cell and r['architecture']=='V5' and r['training_support']==training)
            assert ident['checkpoint']==row and sha(row['checkpoint'])==row['checkpoint_sha256']
            assert g['status']=='PASS_FULL_QUERY_D123_RAW' and g['queries']=={s:data['counts'][s] for s in ('val','test')}
            assert g['channels']==(7 if cell.startswith('ETTh1') else 21)
            v5.append(dict(**row,validated_query_counts=g['queries'],channels=g['channels'],source_gate=str(gatepath),source_gate_sha256=sha(gatepath),validated_runtime=ident['current_torch']))
            for arm in ('V0','V1'):
                row=next(r for r in rows if r['cell']==cell and r['architecture']==arm and r['training_support']==training)
                base=dict(cell=cell,architecture=arm,training_support=training,checkpoint=row['checkpoint'],checkpoint_sha256=row['checkpoint_sha256'],selected_epoch=row['selected_epoch'],selection_criterion=row['selection_criterion'])
                try:
                    assert sha(row['checkpoint'])==row['checkpoint_sha256']
                    if training=='P100':
                        evidence=Path(row['evidence']);seen=[]
                        for c in range(data['channels']):
                            p=evidence/'queries/test'/('channel%03d.npz'%c)
                            with np.load(p) as z:
                                assert len(z['query_start'])==data['counts']['test']
                                assert z['query_start'][0]==data['starts']['test'][0] and z['query_start'][-1]==data['starts']['test'][1]
                                assert z['candidate_ids'].shape[0]==data['counts']['test'] and z['candidate_ids'].shape[-1]>=10
                            seen.append(dict(path=str(p),sha256=sha(p)))
                        value=float(row['canonical_retmse']);source=str(evidence);source_hashes=seen
                    else:
                        ds,h=cell.rsplit('_',1);cache=ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{ds}/H{h}/seed0/{arm}/cache'
                        p=cache/'test.pt';z=torch.load(p,map_location='cpu',weights_only=False)
                        q=z['query_start_idx'].numpy();assert len(q)==data['counts']['test']
                        assert q[0]==data['starts']['test'][0] and q[-1]==data['starts']['test'][1]
                        assert tuple(z['relation_outputs'].shape)==(data['counts']['test'],data['pred_len'],data['channels'])
                        del z
                        mp=cache/'stage1_metrics.json';value=float(json.loads(mp.read_text())['retmse10']);source=str(mp)
                        source_hashes=[dict(path=str(p),sha256=sha(p)),dict(path=str(mp),sha256=sha(mp))]
                    comparators.append(dict(**base,status='VALID_REUSED_OFFICIAL_ALL_QUERY_NATIVE',retmse=value,source=source,source_hashes=source_hashes,queries=data['counts']['test'],channels=data['channels']))
                except Exception as e:comparators.append(dict(**base,status='MISSING_OR_INCOMPATIBLE',reason=repr(e)))
    dump(OUT/'CHECKPOINT_AUDIT.json',dict(v5=v5,comparators=comparators,protocol=PROTOCOL,source_request_sha256=sha(HERE/'SOURCE_REQUEST.md')))
    with (OUT/'CHECKPOINT_INVENTORY.csv').open('w',newline='') as f:
        cols=['cell','training_support','checkpoint','checkpoint_sha256','selected_epoch','selection_criterion','validated_runtime','channels','source_gate_sha256']
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(v5)
    dump(OUT/'PROTOCOL.json',PROTOCOL)
    print('CPU_AUDIT',len(v5),'V5',sum(x['status'].startswith('VALID') for x in comparators),'comparators',flush=True)
if __name__=='__main__':audit()
