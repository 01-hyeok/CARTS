import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pipeline as p
# Exact canonical cosine function, original PyTorch/CUDA numerical runtime.
p.baseline_scores=p.compute_coarse_delta_last_cosine_scores
out=p.precompute(sys.argv[1],"baseline_cosine")
import torch,json
meta=json.loads((out/"COMPLETE.json").read_text())
meta.update(precompute_python=sys.executable,torch_version=torch.__version__,cuda_version=torch.version.cuda,baseline_ordered_id_parity="ALL_SPLITS_EXACT")
p.dump(out/"COMPLETE.json",meta)
