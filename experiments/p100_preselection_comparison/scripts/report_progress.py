#!/usr/bin/env python3
"""Read-only experiment inspection; writes only P100 reports."""
import csv,json,re,os
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parents[1]
CELLS=("ETTh1_96","ETTh1_720","Weather_96","Weather_720")
METHODS=("baseline_cosine","paa16_cosine","saraf_time_aligned")
def read(p):
    return json.loads(p.read_text()) if p.exists() else {}
def table(rows,cols):
    return ["|"+"|".join(cols)+"|","|"+"|".join("---" for _ in cols)+"|"]+["|"+"|".join(str(r.get(c,"—")) for c in cols)+"|" for r in rows]
def main():
    current=read(R/"reports/current.json")
    alive=bool(current.get("pid") and Path("/proc/"+str(current["pid"])).exists())
    allrows=[];eff=[]
    for cell in CELLS:
      for method in METHODS:
       for arm in (0,5):
        out=R/"results/runs"/cell/method/("V"+str(arm))
        done=read(out/"COMPLETE.json"); started=read(out/"STARTED.json")
        state="Completed" if done else ("Running" if started and alive else ("Failed/Interrupted" if started else "Not Run"))
        row=dict(Cell=cell,Model="Original KL" if arm==0 else "5-Head V5",Method=method,Status=state)
        if done:
          s1=read(out/"cache/stage1_metrics.json");s2=read(out/"stage2/metrics.json")
          manifest=read(out/"PROVENANCE.json")
          assert manifest["cell"]==cell and manifest["seed"]==0 and manifest["method"]==method and manifest["model"]=="V"+str(arm)
          assert s2["seed"]==0 and done["independent_stage2"]["passed"]
          row.update(MSE=s2["test_mse"],MAE=s2["test_mae"],Recall10=s1["recall10"],RetMSE=s1["retmse10"],AggMSE=s1["agg_mse10"],Checkpoint=manifest["checkpoint"])
          row["HeadAnalysis"]="Completed" if arm==5 and (out/"head_metrics_test.json").exists() else ("Pending" if arm==5 else "N/A")
        log=R/"logs"/(cell+"_"+method+"_V"+str(arm)+".log")
        row["Log"]=str(log)
        resources=[]
        if log.exists():
          for line in log.read_text().splitlines():
            if line.startswith("RESOURCE_METRICS "):resources.append(json.loads(line[len("RESOURCE_METRICS "):]))
        if resources:
          er=dict(Cell=cell,Model=row["Model"],Method=method,Status=state)
          er["stage1_seconds"]=sum(x["seconds"] for x in resources if x["module"]=="train_v_sharedtop100_01")
          er["stage2_seconds"]=sum(x["seconds"] for x in resources if x["module"]=="train_r_stage2_lambda01")
          er["recorded_subprocess_seconds"]=sum(x["seconds"] for x in resources)
          if done and started:
            er["run_wall_seconds_through_stage2"]=out.joinpath("COMPLETE.json").stat().st_mtime-datetime.strptime(started["started_utc"],"%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
          er["peak_allocated_bytes"]=max(x["peak_allocated"] for x in resources)
          er["peak_reserved_bytes"]=max(x["peak_reserved"] for x in resources)
          er["measurement"]="CUDA synchronized subprocess interval; concurrent GPU workload; incomplete until Completed"
          eff.append(er)
        allrows.append(row)
    for name,rows in (("verified_results.csv",allrows),("efficiency.csv",eff)):
      if rows:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        with (R/"reports"/name).open("w") as f:
          w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    text=["# P100 Candidate Preselection Comparison — seed0","",
      "Primary outcomes: Stage2 test MSE and MAE. Sequence: ETTh1 96 → ETTh1 720 → Weather 96 → Weather 720; all six settings finish Stage2 before advancing.",
      "Completed counts only runs with independent Stage2 validation and matching method/seed provenance. No unrun result is estimated.",
      "Timing is measured with concurrent GPU use and is not an independent speed benchmark.",""]
    for cell in CELLS:
      rows=[dict(r) for r in allrows if r["Cell"]==cell]
      for metric in ("MSE","MAE","RetMSE","AggMSE","Recall10"):
        unique=sorted({r[metric] for r in rows if metric in r},reverse=metric=="Recall10")
        for r in rows:
          if metric in r:
            v=r[metric];s=f"{v:.6f}"
            if len(unique)>=1 and v==unique[0]:s="**"+s+"**"
            elif len(unique)>=2 and v==unique[1]:s="<u>"+s+"</u>"
            r[metric]=s
      text+=["## "+cell,""]+table(rows,["Model","Method","Status","Recall10","RetMSE","AggMSE","MSE","MAE"])+[""]
      q=read(R/"results/quality"/cell/"metrics.json")
      if q:
        qr=[dict(Method=m,**v) for m,v in q["methods"].items()]
        text+=["Validation diagnostics: identical fixed query subset, all channels. Oracle future values are diagnostic-only. Set oracle is greedy, approximate, without replacement.",""]+table(qr,["Method","oracle_coverage100","mixture_retention100","best_individual_mse","individual_oracle_top10_mse","approximate_greedy_set_oracle_aggmse","baseline_overlap100"])+[""]
      complete=[r for r in allrows if r["Cell"]==cell and r["Status"]=="Completed"]
      if len(complete)==6:
        best=min(complete,key=lambda x:x["MSE"]); bestmae=min(complete,key=lambda x:x["MAE"])
        text += [f"Lowest Stage2 MSE: {best['Model']} / {best['Method']} ({best['MSE']:.6f}); lowest MAE: {bestmae['Model']} / {bestmae['Method']} ({bestmae['MAE']:.6f})."]
        for model in ("Original KL","5-Head V5"):
          rr=[r for r in complete if r["Model"]==model];base=next(r for r in rr if r["Method"]=="baseline_cosine")
          span=max(r["MSE"] for r in rr)-min(r["MSE"] for r in rr)
          text += [f"{model}: Stage2 MSE range across preselectors {span:.6f}; baseline {base['MSE']:.6f}."]
          for r in rr:
            if r["Method"]!="baseline_cosine":text += [f"{r['Method']} MSE change relative to baseline: {100*(r['MSE']/base['MSE']-1):+.2f}%."]
        text+=["These are seed0 observations, not evidence of statistical reliability or causal head specialization.",""]
    for model,name in (("Original KL","ORIGINAL_KL_RESULTS.md"),("5-Head V5","V5_RESULTS.md")):
      rows=[r for r in allrows if r["Model"]==model]
      (R/"reports"/name).write_text("\n".join(["# "+model+" — seed0",""]+table(rows,["Cell","Method","Status","Recall10","RetMSE","AggMSE","MSE","MAE"])))
    text+=["## Retrieval and forecasting agreement",""]
    for cell in CELLS:
      q=read(R/"results/quality"/cell/"metrics.json")
      done=[r for r in allrows if r["Cell"]==cell and r["Status"]=="Completed"]
      if len(done)!=6 or not q:continue
      for metric in ("RetMSE","AggMSE","Recall10"):
        best=(max if metric=="Recall10" else min)(done,key=lambda r:r[metric])
        text.append(f"{cell} best {metric}: {best['Model']} / {best['Method']} = {best[metric]:.6f}.")
      for model in ("Original KL","5-Head V5"):
        base=next(r for r in done if r["Model"]==model and r["Method"]=="baseline_cosine")
        for r in done:
          if r["Model"]!=model or r["Method"]=="baseline_cosine":continue
          dc=q["methods"][r["Method"]]["oracle_coverage100"]-q["methods"]["baseline_cosine"]["oracle_coverage100"]
          dm=r["MSE"]-base["MSE"]
          text.append(f"{cell} {model} {r['Method']}: coverage delta {dc:+.6f}, Stage2 MSE delta {dm:+.6f}; coverage is a fixed validation subset diagnostic, forecasting is full test.")
    text+=["","## Head statistics",""]
    headrows=[]
    for cell in CELLS:
      for method in METHODS:
        h=read(R/"results/runs"/cell/method/"V5/head_metrics_test.json")
        if not h:continue
        off=lambda key:sum(h[key][i][j] for i in range(5) for j in range(5) if i!=j)/20
        headrows.append(dict(Cell=cell,Method=method,MeanHeadErrorCosine=off("head_error_cosine"),MeanHeadTop10Overlap=off("head_top10_overlap"),MixtureAggMSE=h["mixture_aggMSE"],HeadRetMSE=h["head_retMSE"]))
    if headrows:text+=table(headrows,["Cell","Method","MeanHeadErrorCosine","MeanHeadTop10Overlap","MixtureAggMSE","HeadRetMSE"])
    text+=["","These are descriptive diversity/complementarity diagnostics; no causal head-removal intervention was run.","","## Observed resource measurements",""]
    if eff:text+=table(eff,["Cell","Model","Method","Status","stage1_seconds","stage2_seconds","run_wall_seconds_through_stage2","peak_allocated_bytes","peak_reserved_bytes"])
    text+=["","Do not infer an isolated speed ranking from concurrent workloads. The most accurate arm and lowest observed runtime/memory may differ; timing excludes separate diagnostic head passes. Feature/search preselection timings are stored per method in results/pools/<cell>/<method>/COMPLETE.json, with timing-boundary limitations in IMPLEMENTATION_AUDIT.md.",""]
    text+=["## Provenance and limitations","",
      "See IMPLEMENTATION_AUDIT.md for Original KL/V5/Stage2 definitions, official SARAF commit, timestamp handling and singleton baseline regression.",
      "Each run's PROVENANCE.json records Stage1 checkpoint/hash and all pool hashes. COMPLETE.json contains independent Stage2 verification and base checkpoint SHA.",
      "Head metrics are in each V5 run's head_metrics_test.json. Head error cosine and overlap do not establish causal complementarity.",
      "Full raw paths and status appear below.",""]+table(allrows,["Cell","Model","Method","Status","Checkpoint","Log"])
    (R/"reports/VERIFIED_ANALYSIS.md").write_text("\n".join(text))
    print(json.dumps(dict(completed=sum(r["Status"]=="Completed" for r in allrows),total=24,current=current,alive=alive)))
if __name__=="__main__":main()
