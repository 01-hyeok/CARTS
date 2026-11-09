# P100 Candidate Preselection Comparison — seed0

Primary outcomes: Stage2 test MSE and MAE. Sequence: ETTh1 96 → ETTh1 720 → Weather 96 → Weather 720; all six settings finish Stage2 before advancing.
Completed counts only runs with independent Stage2 validation and matching method/seed provenance. No unrun result is estimated.
Timing is measured with concurrent GPU use and is not an independent speed benchmark.

## ETTh1_96

|Model|Method|Status|Recall10|RetMSE|AggMSE|MSE|MAE|
|---|---|---|---|---|---|---|---|
|Original KL|baseline_cosine|Completed|0.238984|0.682849|**0.408743**|**0.369243**|**0.394206**|
|5-Head V5|baseline_cosine|Completed|0.257522|**0.663855**|<u>0.409808</u>|<u>0.369597</u>|0.395469|
|Original KL|paa16_cosine|Completed|0.244965|0.696640|0.414580|0.370967|<u>0.394461</u>|
|5-Head V5|paa16_cosine|Completed|**0.263847**|<u>0.675279</u>|0.414585|0.370898|0.395283|
|Original KL|saraf_time_aligned|Completed|0.243837|0.695348|0.415535|0.371283|0.396044|
|5-Head V5|saraf_time_aligned|Completed|<u>0.262729</u>|0.681370|0.419784|0.372044|0.398562|

Validation diagnostics: identical fixed query subset, all channels. Oracle future values are diagnostic-only. Set oracle is greedy, approximate, without replacement.

|Method|oracle_coverage100|mixture_retention100|best_individual_mse|individual_oracle_top10_mse|approximate_greedy_set_oracle_aggmse|baseline_overlap100|
|---|---|---|---|---|---|---|
|baseline_cosine|0.1709821460036827|0.37187500540832324|0.3950217849841075|0.5482257743632155|0.33109614939894527|1.0|
|paa16_cosine|0.13125000245470023|0.31339286028274466|0.4170436443361853|0.5673364182668073|0.3479433548137812|0.7044642700680664|
|saraf_time_aligned|0.16160714589724584|0.31875000415103777|0.3824659254668014|0.5442859836454902|0.3218518937438993|0.5348214166504996|

Lowest Stage2 MSE: Original KL / baseline_cosine (0.369243); lowest MAE: Original KL / baseline_cosine (0.394206).
Original KL: Stage2 MSE range across preselectors 0.002040; baseline 0.369243.
paa16_cosine MSE change relative to baseline: +0.47%.
saraf_time_aligned MSE change relative to baseline: +0.55%.
5-Head V5: Stage2 MSE range across preselectors 0.002447; baseline 0.369597.
paa16_cosine MSE change relative to baseline: +0.35%.
saraf_time_aligned MSE change relative to baseline: +0.66%.
These are seed0 observations, not evidence of statistical reliability or causal head specialization.

## ETTh1_720

|Model|Method|Status|Recall10|RetMSE|AggMSE|MSE|MAE|
|---|---|---|---|---|---|---|---|
|Original KL|baseline_cosine|Completed|0.222701|1.020577|0.640548|0.521939|0.513005|
|5-Head V5|baseline_cosine|Completed|<u>0.321286</u>|**0.942633**|<u>0.620570</u>|<u>0.514917</u>|<u>0.508438</u>|
|Original KL|paa16_cosine|Completed|0.312620|1.117467|0.746275|0.536487|0.525711|
|5-Head V5|paa16_cosine|Completed|**0.370384**|1.040949|0.701014|0.527924|0.520175|
|Original KL|saraf_time_aligned|Completed|0.198909|1.021385|0.643660|0.526369|0.517363|
|5-Head V5|saraf_time_aligned|Completed|0.258234|<u>0.960586</u>|**0.604064**|**0.509770**|**0.507301**|

Validation diagnostics: identical fixed query subset, all channels. Oracle future values are diagnostic-only. Set oracle is greedy, approximate, without replacement.

|Method|oracle_coverage100|mixture_retention100|best_individual_mse|individual_oracle_top10_mse|approximate_greedy_set_oracle_aggmse|baseline_overlap100|
|---|---|---|---|---|---|---|
|baseline_cosine|0.09732142945618502|0.1861607178580016|1.2575232301439558|1.467890867697341|1.0851282122145807|1.0|
|paa16_cosine|0.015178571721272809|0.06651785826709654|1.3968402514500278|1.5863974961851324|1.2165183491472686|0.3578571353905967|
|saraf_time_aligned|0.07500000101780253|0.12366071695994053|1.1354221885225602|1.3466401829251222|0.9479515680244991|0.4240624905963029|

Lowest Stage2 MSE: 5-Head V5 / saraf_time_aligned (0.509770); lowest MAE: 5-Head V5 / saraf_time_aligned (0.507301).
Original KL: Stage2 MSE range across preselectors 0.014548; baseline 0.521939.
paa16_cosine MSE change relative to baseline: +2.79%.
saraf_time_aligned MSE change relative to baseline: +0.85%.
5-Head V5: Stage2 MSE range across preselectors 0.018154; baseline 0.514917.
paa16_cosine MSE change relative to baseline: +2.53%.
saraf_time_aligned MSE change relative to baseline: -1.00%.
These are seed0 observations, not evidence of statistical reliability or causal head specialization.

## Weather_96

|Model|Method|Status|Recall10|RetMSE|AggMSE|MSE|MAE|
|---|---|---|---|---|---|---|---|
|Original KL|baseline_cosine|Completed|0.250195|0.430873|0.241648|0.174460|0.214232|
|5-Head V5|baseline_cosine|Completed|**0.266581**|0.455618|0.264330|0.176649|0.213072|
|Original KL|paa16_cosine|Completed|0.244958|**0.403567**|<u>0.227845</u>|0.171986|0.213534|
|5-Head V5|paa16_cosine|Completed|<u>0.261272</u>|0.416574|0.243635|0.174263|0.212299|
|Original KL|saraf_time_aligned|Completed|0.219682|0.418474|0.232973|<u>0.169536</u>|<u>0.209730</u>|
|5-Head V5|saraf_time_aligned|Completed|0.228926|<u>0.403597</u>|**0.224169**|**0.168068**|**0.208819**|

Validation diagnostics: identical fixed query subset, all channels. Oracle future values are diagnostic-only. Set oracle is greedy, approximate, without replacement.

|Method|oracle_coverage100|mixture_retention100|best_individual_mse|individual_oracle_top10_mse|approximate_greedy_set_oracle_aggmse|baseline_overlap100|
|---|---|---|---|---|---|---|
|baseline_cosine|0.08854166725428686|0.10877976325961451|0.3464197999886069|0.38676751501645323|0.3184095480467996|1.0|
|paa16_cosine|0.08794642889517404|0.1061011919352625|0.34519233586576903|0.38421067734444003|0.3162944107330185|0.7810416491258712|
|saraf_time_aligned|0.033333334007433484|0.08005952512446259|0.3191022713471111|0.36183103538652567|0.29372579456834763|0.32784225259508404|

Lowest Stage2 MSE: 5-Head V5 / saraf_time_aligned (0.168068); lowest MAE: 5-Head V5 / saraf_time_aligned (0.208819).
Original KL: Stage2 MSE range across preselectors 0.004924; baseline 0.174460.
paa16_cosine MSE change relative to baseline: -1.42%.
saraf_time_aligned MSE change relative to baseline: -2.82%.
5-Head V5: Stage2 MSE range across preselectors 0.008581; baseline 0.176649.
paa16_cosine MSE change relative to baseline: -1.35%.
saraf_time_aligned MSE change relative to baseline: -4.86%.
These are seed0 observations, not evidence of statistical reliability or causal head specialization.

## Weather_720

|Model|Method|Status|Recall10|RetMSE|AggMSE|MSE|MAE|
|---|---|---|---|---|---|---|---|
|Original KL|baseline_cosine|Completed|0.229534|0.781558|0.592618|<u>0.372792</u>|0.370115|
|5-Head V5|baseline_cosine|Completed|<u>0.255354</u>|0.787918|0.617802|0.385526|<u>0.368637</u>|
|Original KL|paa16_cosine|Completed|0.239745|0.773078|0.594958|0.379613|0.371083|
|5-Head V5|paa16_cosine|Completed|**0.265367**|<u>0.759172</u>|<u>0.571207</u>|0.374329|0.370758|
|Original KL|saraf_time_aligned|Completed|0.202118|0.806625|0.613331|0.378074|0.370003|
|5-Head V5|saraf_time_aligned|Completed|0.229134|**0.720501**|**0.566542**|**0.370556**|**0.365744**|

Validation diagnostics: identical fixed query subset, all channels. Oracle future values are diagnostic-only. Set oracle is greedy, approximate, without replacement.

|Method|oracle_coverage100|mixture_retention100|best_individual_mse|individual_oracle_top10_mse|approximate_greedy_set_oracle_aggmse|baseline_overlap100|
|---|---|---|---|---|---|---|
|baseline_cosine|0.032291667081326955|0.04642857228671866|0.5656188373579338|0.6175633729641655|0.4547270258545246|1.0|
|paa16_cosine|0.029017857486559523|0.03824404848828202|0.5610258291037878|0.6096126822341743|0.44543543649888306|0.6735862938332415|
|saraf_time_aligned|0.02514880977659708|0.038839286259774654|0.5369281791084047|0.6012649357577348|0.44001125353419673|0.4583928486598389|

Lowest Stage2 MSE: 5-Head V5 / saraf_time_aligned (0.370556); lowest MAE: 5-Head V5 / saraf_time_aligned (0.365744).
Original KL: Stage2 MSE range across preselectors 0.006821; baseline 0.372792.
paa16_cosine MSE change relative to baseline: +1.83%.
saraf_time_aligned MSE change relative to baseline: +1.42%.
5-Head V5: Stage2 MSE range across preselectors 0.014969; baseline 0.385526.
paa16_cosine MSE change relative to baseline: -2.90%.
saraf_time_aligned MSE change relative to baseline: -3.88%.
These are seed0 observations, not evidence of statistical reliability or causal head specialization.

## Retrieval and forecasting agreement

ETTh1_96 best RetMSE: 5-Head V5 / baseline_cosine = 0.663855.
ETTh1_96 best AggMSE: Original KL / baseline_cosine = 0.408743.
ETTh1_96 best Recall10: 5-Head V5 / paa16_cosine = 0.263847.
ETTh1_96 Original KL paa16_cosine: coverage delta -0.039732, Stage2 MSE delta +0.001724; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_96 Original KL saraf_time_aligned: coverage delta -0.009375, Stage2 MSE delta +0.002040; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_96 5-Head V5 paa16_cosine: coverage delta -0.039732, Stage2 MSE delta +0.001301; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_96 5-Head V5 saraf_time_aligned: coverage delta -0.009375, Stage2 MSE delta +0.002447; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_720 best RetMSE: 5-Head V5 / baseline_cosine = 0.942633.
ETTh1_720 best AggMSE: 5-Head V5 / saraf_time_aligned = 0.604064.
ETTh1_720 best Recall10: 5-Head V5 / paa16_cosine = 0.370384.
ETTh1_720 Original KL paa16_cosine: coverage delta -0.082143, Stage2 MSE delta +0.014548; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_720 Original KL saraf_time_aligned: coverage delta -0.022321, Stage2 MSE delta +0.004430; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_720 5-Head V5 paa16_cosine: coverage delta -0.082143, Stage2 MSE delta +0.013006; coverage is a fixed validation subset diagnostic, forecasting is full test.
ETTh1_720 5-Head V5 saraf_time_aligned: coverage delta -0.022321, Stage2 MSE delta -0.005148; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_96 best RetMSE: Original KL / paa16_cosine = 0.403567.
Weather_96 best AggMSE: 5-Head V5 / saraf_time_aligned = 0.224169.
Weather_96 best Recall10: 5-Head V5 / baseline_cosine = 0.266581.
Weather_96 Original KL paa16_cosine: coverage delta -0.000595, Stage2 MSE delta -0.002475; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_96 Original KL saraf_time_aligned: coverage delta -0.055208, Stage2 MSE delta -0.004924; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_96 5-Head V5 paa16_cosine: coverage delta -0.000595, Stage2 MSE delta -0.002386; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_96 5-Head V5 saraf_time_aligned: coverage delta -0.055208, Stage2 MSE delta -0.008581; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_720 best RetMSE: 5-Head V5 / saraf_time_aligned = 0.720501.
Weather_720 best AggMSE: 5-Head V5 / saraf_time_aligned = 0.566542.
Weather_720 best Recall10: 5-Head V5 / paa16_cosine = 0.265367.
Weather_720 Original KL paa16_cosine: coverage delta -0.003274, Stage2 MSE delta +0.006821; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_720 Original KL saraf_time_aligned: coverage delta -0.007143, Stage2 MSE delta +0.005282; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_720 5-Head V5 paa16_cosine: coverage delta -0.003274, Stage2 MSE delta -0.011197; coverage is a fixed validation subset diagnostic, forecasting is full test.
Weather_720 5-Head V5 saraf_time_aligned: coverage delta -0.007143, Stage2 MSE delta -0.014969; coverage is a fixed validation subset diagnostic, forecasting is full test.

## Head statistics

|Cell|Method|MeanHeadErrorCosine|MeanHeadTop10Overlap|MixtureAggMSE|HeadRetMSE|
|---|---|---|---|---|---|
|ETTh1_96|baseline_cosine|0.843512624502182|0.3754126787185669|0.4098076494134032|[0.6747603416442871, 0.6749831438064575, 0.7248733043670654, 0.696394145488739, 0.7008069753646851]|
|ETTh1_96|paa16_cosine|0.8452604353427887|0.37934490144252775|0.41458458640311935|[0.7295249700546265, 0.6806991100311279, 0.7365086078643799, 0.7020812630653381, 0.686476469039917]|
|ETTh1_96|saraf_time_aligned|0.8167483985424042|0.3520564138889313|0.4197840657024987|[0.6874238848686218, 0.7630841135978699, 0.7427740097045898, 0.7113057971000671, 0.6844842433929443]|
|ETTh1_720|baseline_cosine|0.9517355620861053|0.7315309762954711|0.6205701582759783|[0.9724923372268677, 0.9428905844688416, 0.9378695487976074, 0.9563637971878052, 0.9366835355758667]|
|ETTh1_720|paa16_cosine|0.9569777011871338|0.7489614844322204|0.7010141999217013|[1.0669035911560059, 1.075087070465088, 1.024519443511963, 1.04210364818573, 1.0625978708267212]|
|ETTh1_720|saraf_time_aligned|0.9433025479316711|0.7119653671979904|0.6040645199221724|[1.0095953941345215, 0.9848325848579407, 0.9322341680526733, 0.9563934206962585, 0.967932403087616]|
|Weather_96|baseline_cosine|0.649662709236145|0.25256859362125395|0.26432987318967344|[0.6071714162826538, 0.6535886526107788, 0.4683806598186493, 0.43047210574150085, 0.529029905796051]|
|Weather_96|paa16_cosine|0.6312069833278656|0.21616000086069107|0.24363466595002717|[0.4363723397254944, 0.5432496070861816, 0.4386257529258728, 0.6287158131599426, 0.37668493390083313]|
|Weather_96|saraf_time_aligned|0.6493936359882355|0.1992853119969368|0.2241693929843168|[0.4578542709350586, 0.5773080587387085, 0.5738454461097717, 0.3658367097377777, 0.5074260234832764]|
|Weather_720|baseline_cosine|0.8085419952869415|0.43746939599514006|0.6178019317519308|[0.8590858578681946, 0.7430852055549622, 0.7327743768692017, 0.7725602984428406, 0.7684114575386047]|
|Weather_720|paa16_cosine|0.8135168612003326|0.45430236160755155|0.5712068005788599|[0.836641252040863, 0.8239715695381165, 0.8324136734008789, 0.772132158279419, 0.790067732334137]|
|Weather_720|saraf_time_aligned|0.7846837341785431|0.40411231219768523|0.5665417178680096|[0.7192167639732361, 0.8160115480422974, 0.7708646059036255, 0.6919437646865845, 0.8225926160812378]|

These are descriptive diversity/complementarity diagnostics; no causal head-removal intervention was run.

## Observed resource measurements

|Cell|Model|Method|Status|stage1_seconds|stage2_seconds|run_wall_seconds_through_stage2|peak_allocated_bytes|peak_reserved_bytes|
|---|---|---|---|---|---|---|---|---|
|ETTh1_96|Original KL|baseline_cosine|Completed|230.02479324489832|5.480823564343154|264.9140646457672|235463680|281018368|
|ETTh1_96|5-Head V5|baseline_cosine|Completed|459.0878614662215|26.86310850456357|738.9386751651764|237255680|283115520|
|ETTh1_96|Original KL|paa16_cosine|Completed|728.6121871732175|39.436893694102764|840.9198877811432|235463680|281018368|
|ETTh1_96|5-Head V5|paa16_cosine|Completed|955.7376939114183|10.924124002456665|1260.428935289383|237255680|283115520|
|ETTh1_96|Original KL|saraf_time_aligned|Completed|921.5235480507836|29.39028251171112|1024.6552641391754|235463680|281018368|
|ETTh1_96|5-Head V5|saraf_time_aligned|Completed|897.6128630386665|29.447784743271768|1232.4575898647308|237255680|283115520|
|ETTh1_720|Original KL|baseline_cosine|Completed|915.6125623825938|16.003680259920657|1050.4387843608856|1506881536|1644167168|
|ETTh1_720|5-Head V5|baseline_cosine|Completed|671.0629188632593|9.526526615023613|862.5691194534302|1508673536|1646264320|
|ETTh1_720|Original KL|paa16_cosine|Completed|568.3287777528167|23.499733313918114|659.9231951236725|1506881536|1644167168|
|ETTh1_720|5-Head V5|paa16_cosine|Completed|576.0546564608812|11.096677541732788|773.3474533557892|1508673536|1646264320|
|ETTh1_720|Original KL|saraf_time_aligned|Completed|545.8393125971779|24.06853566877544|626.8580005168915|1506881536|1644167168|
|ETTh1_720|5-Head V5|saraf_time_aligned|Completed|414.08745072968304|17.58627152722329|566.2363142967224|1508673536|1646264320|
|Weather_96|Original KL|baseline_cosine|Completed|5597.989142943174|56.312167829833925|6012.534792423248|1786658816|2587885568|
|Weather_96|5-Head V5|baseline_cosine|Completed|5358.092168268748|39.18583854287863|6188.840266704559|1786658816|2587885568|
|Weather_96|Original KL|paa16_cosine|Completed|3972.8345970511436|18.13124083355069|4135.685547590256|1786658816|2587885568|
|Weather_96|5-Head V5|paa16_cosine|Completed|2223.606033705175|16.77323060017079|2577.572655916214|1786658816|2587885568|
|Weather_96|Original KL|saraf_time_aligned|Completed|2124.763611056842|18.079622655175626|2286.841461420059|1786658816|2587885568|
|Weather_96|5-Head V5|saraf_time_aligned|Completed|2264.8937672730535|18.279501876793802|2680.179137945175|1786658816|2587885568|
|Weather_720|Original KL|baseline_cosine|Completed|3225.321045288816|20.97490438912064|3389.5677313804626|12974104064|18765316096|
|Weather_720|5-Head V5|baseline_cosine|Completed|2633.482985998504|20.994483333081007|2975.6944518089294|12974104064|18765316096|
|Weather_720|Original KL|paa16_cosine|Completed|2557.733605758287|22.60285794734955|2723.9638409614563|12974104064|18765316096|
|Weather_720|5-Head V5|paa16_cosine|Completed|2628.3617551745847|22.630312910303473|2971.4312188625336|12974104064|18765316096|
|Weather_720|Original KL|saraf_time_aligned|Completed|2573.007101853378|22.608313901349902|2740.2917239665985|12974104064|18765316096|
|Weather_720|5-Head V5|saraf_time_aligned|Completed|2777.4141739634797|22.747882201336324|3172.3946883678436|12974104064|18765316096|

Do not infer an isolated speed ranking from concurrent workloads. The most accurate arm and lowest observed runtime/memory may differ; timing excludes separate diagnostic head passes. Feature/search preselection timings are stored per method in results/pools/<cell>/<method>/COMPLETE.json, with timing-boundary limitations in IMPLEMENTATION_AUDIT.md.

## Provenance and limitations

See IMPLEMENTATION_AUDIT.md for Original KL/V5/Stage2 definitions, official SARAF commit, timestamp handling and singleton baseline regression.
Each run's PROVENANCE.json records Stage1 checkpoint/hash and all pool hashes. COMPLETE.json contains independent Stage2 verification and base checkpoint SHA.
Head metrics are in each V5 run's head_metrics_test.json. Head error cosine and overlap do not establish causal complementarity.
Full raw paths and status appear below.

|Cell|Model|Method|Status|Checkpoint|Log|
|---|---|---|---|---|---|
|ETTh1_96|Original KL|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/baseline_cosine/V0/ETTh1_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_baseline_cosine_V0.log|
|ETTh1_96|5-Head V5|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/baseline_cosine/V5/ETTh1_96/V5/checkpoint_epoch10.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_baseline_cosine_V5.log|
|ETTh1_96|Original KL|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/paa16_cosine/V0/ETTh1_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_paa16_cosine_V0.log|
|ETTh1_96|5-Head V5|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/paa16_cosine/V5/ETTh1_96/V5/checkpoint_epoch10.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_paa16_cosine_V5.log|
|ETTh1_96|Original KL|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/saraf_time_aligned/V0/ETTh1_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_saraf_time_aligned_V0.log|
|ETTh1_96|5-Head V5|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_96/saraf_time_aligned/V5/ETTh1_96/V5/checkpoint_epoch10.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_96_saraf_time_aligned_V5.log|
|ETTh1_720|Original KL|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/baseline_cosine/V0/ETTh1_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_baseline_cosine_V0.log|
|ETTh1_720|5-Head V5|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/baseline_cosine/V5/ETTh1_720/V5/checkpoint_epoch9.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_baseline_cosine_V5.log|
|ETTh1_720|Original KL|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/paa16_cosine/V0/ETTh1_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_paa16_cosine_V0.log|
|ETTh1_720|5-Head V5|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/paa16_cosine/V5/ETTh1_720/V5/checkpoint_epoch10.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_paa16_cosine_V5.log|
|ETTh1_720|Original KL|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/saraf_time_aligned/V0/ETTh1_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_saraf_time_aligned_V0.log|
|ETTh1_720|5-Head V5|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/ETTh1_720/saraf_time_aligned/V5/ETTh1_720/V5/checkpoint_epoch6.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/ETTh1_720_saraf_time_aligned_V5.log|
|Weather_96|Original KL|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/baseline_cosine/V0/Weather_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_baseline_cosine_V0.log|
|Weather_96|5-Head V5|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/baseline_cosine/V5/Weather_96/V5/checkpoint_epoch4.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_baseline_cosine_V5.log|
|Weather_96|Original KL|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/paa16_cosine/V0/Weather_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_paa16_cosine_V0.log|
|Weather_96|5-Head V5|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/paa16_cosine/V5/Weather_96/V5/checkpoint_epoch6.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_paa16_cosine_V5.log|
|Weather_96|Original KL|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/saraf_time_aligned/V0/Weather_96/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_saraf_time_aligned_V0.log|
|Weather_96|5-Head V5|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_96/saraf_time_aligned/V5/Weather_96/V5/checkpoint_epoch8.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_96_saraf_time_aligned_V5.log|
|Weather_720|Original KL|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/baseline_cosine/V0/Weather_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_baseline_cosine_V0.log|
|Weather_720|5-Head V5|baseline_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/baseline_cosine/V5/Weather_720/V5/checkpoint_epoch6.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_baseline_cosine_V5.log|
|Weather_720|Original KL|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/paa16_cosine/V0/Weather_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_paa16_cosine_V0.log|
|Weather_720|5-Head V5|paa16_cosine|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/paa16_cosine/V5/Weather_720/V5/checkpoint_epoch7.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_paa16_cosine_V5.log|
|Weather_720|Original KL|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/saraf_time_aligned/V0/Weather_720/V0/checkpoint_best_retmse.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_saraf_time_aligned_V0.log|
|Weather_720|5-Head V5|saraf_time_aligned|Completed|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/checkpoints/Weather_720/saraf_time_aligned/V5/Weather_720/V5/checkpoint_epoch6.pth|/data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/logs/Weather_720_saraf_time_aligned_V5.log|