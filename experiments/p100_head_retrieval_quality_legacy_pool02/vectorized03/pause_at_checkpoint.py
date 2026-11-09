import pathlib,json,os,signal,time,datetime,hashlib,shutil
v=pathlib.Path(__file__).parent;p=v.parent;r=p.parents[1];o=r/'results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02'
a=json.loads((v/'handoff_before.json').read_text());cur=a['current'];pid=cur['worker_pid'];target=a['target_epoch'];rid=cur['target'];cp=r/'checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/Solar_96/seed0/V2'/f'epoch{target:02d}.pth';dest=o/'runs/Solar_96/seed0/V2'
expected=str(p/'runtime.py').encode();stat0=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
while True:
 proc=pathlib.Path(f'/proc/{pid}');assert proc.exists(),'Worker ended before checkpoint handoff'
 assert proc.joinpath('stat').read_text().split(') ',1)[1].split()[19]==stat0
 assert expected in proc.joinpath('cmdline').read_bytes()
 history=json.loads((dest/'epochs.json').read_text())
 if cp.exists() and history[-1]['epoch']>=target:
  os.kill(pid,signal.SIGSTOP)
  frozen=json.loads((dest/'current.json').read_text())
  archive=o/'vectorized03_prechange/Solar_96_V2_seed0';archive.parent.mkdir(exist_ok=True);shutil.copytree(dest,archive)
  before=dict(status='PAUSED_AT_SAVED_EPOCH',run_id=rid,saved_epoch=target,checkpoint=str(cp),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),old_worker_pid=pid,old_worker_start_ticks=stat0,progress_when_stopped=frozen,stopped_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),partial_results_archive=str(archive),reason='User authorized all-channel vectorization; resume requires tested numerical/RNG policy')
  (v/'PAUSED.json').write_text(json.dumps(before,indent=2)+'\n')
  (o/'current.json').write_text(json.dumps(dict(before,total=48,cancelled_by_user=96),indent=2)+'\n')
  os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
  print(json.dumps(before),flush=True);break
 time.sleep(.2)
