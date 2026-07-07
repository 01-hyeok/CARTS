# Cross-channel Aligned Retrieval for Time Series (CARTS)

Cross-channel Aligned Retrieval for Time Series (CARTS) is a relation-retrieval
time-series forecasting project built on top of
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) and adapted
from RAFT-style retrieval forecasting.


### Required Packages
* python == 3.9.13
* numpy == 1.24.3
* torch == 1.10.0
* tqdm == 4.65.0

### Usage
1. Activate the environment.

```bash
source /data/pjh_workspace/ts-env/bin/activate
```

2. Create the script and log directories if they do not exist.

```bash
mkdir -p scripts logs
```

3. Place the ETT dataset files under the Time-Series-Library dataset path used by
the scripts.

```text
../Dataset/Time-Series-Library_dataset/ETT-small/
```

Expected files include:

```text
ETTh1.csv
ETTm1.csv
```

4. Run an experiment script. For example:

```bash
bash scripts/ETTh1/run_mlp_repeat_ETTh1_96.sh 2>&1 | tee logs/ETTh1/run_mlp_repeat_ETTh1_96.log
```

The current experiment scripts are organized by dataset:

```text
scripts/ETTh1/
scripts/ETTm1/
```
