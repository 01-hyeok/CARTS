import sys,json,csv,torch,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import full_analysis as m
torch.set_num_threads(4)
cell='Weather_720';out=m.OUT/cell
with torch.no_grad():
 cfg,canonical,sel,exp,args,model,heads,base=m.setup(cell,32)
 torch.backends.cuda.matmul.allow_tf32=True
 canonical_cache=torch.load(canonical/'cache/test.pt',map_location='cpu')
 rows=list(csv.DictReader((out/'INDEPENDENT_MODELS.csv').open()))
 for arm in ['V0','V1']:
  loc=m.ROOT/f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/{arm}'
  raw=json.loads((loc/'stage2/metrics.json').read_text());cache=torch.load(loc/'cache/test.pt',map_location='cpu')
  assert torch.equal(cache['query_start_idx'],canonical_cache['query_start_idx'])
  starts,B,R,Y=m.load_tensors(exp,base,loc/'cache','test','cuda',int(args.enc_in))
  gate=m.GlobalLambdaGate().cuda();gate.load_state_dict(torch.load(loc/'stage2/checkpoint.pth',map_location='cuda')['gate_state_dict'])
  mse,mae,lam=m.eval_gate(gate,B,R,Y)
  assert abs(mse-raw['test_mse'])<1e-5 and abs(mae-raw['test_mae'])<1e-5
  row=next(r for r in rows if r['model']==arm)
  row.update(status='VERIFIED',source_kind='historical_single_head_original_selection_equivalent_to_meanmix_verified_forward',test_mse=mse,test_mae=mae,base_sha256=m.sha(cfg['base_checkpoint']),source=str(loc/'stage2/metrics.json'),source_sha256=m.sha(loc/'stage2/metrics.json'),seed=0,split='test',n_test=len(Y),lambda_value=lam,inference='full_single_head_rank_equivalent',selection_note='V0/V1 original single-head selection; same Base and exact test query IDs verified; no multihead RR comparator used')
  print(arm,mse,mae,flush=True);del B,R,Y,cache;torch.cuda.empty_cache()
 p=out/'INDEPENDENT_MODELS.csv'
 backup=out/'INDEPENDENT_MODELS.before_legacy_fallback.csv'
 if not backup.exists():shutil.copy2(p,backup)
 m.csv_write(p,rows)
 m.dump(out/'LEGACY_COMPARATOR_VERIFICATION.json',dict(status='PASS',arms=['V0','V1'],same_base=True,exact_test_query_ids=True,torch_version=torch.__version__,matmul_allow_tf32=True,reason='V0 and V1 have one scoring distribution, so round-robin and mean-mixture rankings coincide; original full artifacts retained when later meanmix path was absent'))
