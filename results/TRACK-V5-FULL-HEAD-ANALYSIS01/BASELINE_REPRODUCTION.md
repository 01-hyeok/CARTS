# 기존 Full V5 재현

| cell | status | stage2_mse | stage2_mae |
| --- | --- | --- | --- |
| ETTh1_96 | PASS | 0.3747245967388153 | 0.3994608223438263 |
| ETTh1_720 | PASS | 0.5036835670471191 | 0.5058333277702332 |
| Weather_96 | PASS | 0.16817615926265717 | 0.21593798696994781 |
| Weather_720 | PASS | 0.34229227900505066 | 0.3475555181503296 |

저장된 원래 lambda로 forward 검증. 신규 All5 캐시 일치 검증은 셀별 newcache_stored_lambda_verification.json.
