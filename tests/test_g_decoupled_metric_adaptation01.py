"""Sanity checks for TRACK-G-DECOUPLED-METRIC-ADAPTATION01.

Fast checks only: identity-init math, arm/head wiring, and result-artifact
consistency against the actually-completed 18-run batch. Does not
re-launch training (that is expensive and already recorded in
results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720/).
"""
import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.train_g_decoupled_metric_adaptation01 import (
    ARMS, FROZEN_ARMS, HAS_METRIC_HEAD, AsymMetric, PatchMeanMetric, build_metric_head, score_fn,
)

RESULT_DIR = Path("results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720")
SEEDS = (0, 1, 2)


def test_arm_partition_is_consistent():
    assert set(FROZEN_ARMS) | {"A4_continue_encoder_nohead", "A5_joint_encoder_freshasym"} == set(ARMS)
    assert set(FROZEN_ARMS) & {"A4_continue_encoder_nohead", "A5_joint_encoder_freshasym"} == set()
    assert set(HAS_METRIC_HEAD) <= set(ARMS)
    assert "A0_frozen_final_cosine" not in HAS_METRIC_HEAD
    assert "A4_continue_encoder_nohead" not in HAS_METRIC_HEAD


def test_build_metric_head_returns_none_for_a0_a4():
    assert build_metric_head("A0_frozen_final_cosine", 128) is None
    assert build_metric_head("A4_continue_encoder_nohead", 128) is None
    assert isinstance(build_metric_head("A1_frozen_final_freshasym", 128), AsymMetric)
    assert isinstance(build_metric_head("A3_frozen_patchmean_freshasym", 128), PatchMeanMetric)


def test_identity_init_asym_metric_equals_cosine():
    """A1's Wq/Wk at init must reproduce plain cosine similarity exactly."""
    torch.manual_seed(0)
    d = 32
    head = AsymMetric(d)
    zq = torch.nn.functional.normalize(torch.randn(5, d), dim=-1)
    zk = torch.nn.functional.normalize(torch.randn(7, d), dim=-1)
    cosine = zq @ zk.T
    head_score = score_fn("A1_frozen_final_freshasym", head, zq, zk)
    assert torch.allclose(cosine, head_score, atol=1e-6)


def test_a0_a4_score_fn_ignores_head():
    """A0/A4 use plain cosine via arm_score regardless of a (None) head."""
    zq = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    zk = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    s0 = score_fn("A0_frozen_final_cosine", None, zq, zk)
    s4 = score_fn("A4_continue_encoder_nohead", None, zq, zk)
    assert torch.allclose(s0, zq @ zk.T, atol=1e-6)
    assert torch.allclose(s4, zq @ zk.T, atol=1e-6)


@pytest.mark.skipif(not RESULT_DIR.exists(), reason="TRACK-G results not present in this checkout")
class TestResultArtifacts:
    def _load(self, arm, seed):
        return json.loads((RESULT_DIR / f"{arm}_seed{seed}_metrics.json").read_text())

    def test_all_18_runs_present(self):
        for arm in ARMS:
            for seed in SEEDS:
                f = RESULT_DIR / f"{arm}_seed{seed}_metrics.json"
                assert f.exists(), f"missing {f}"

    def test_a0_baseline_reproduces_exactly_across_seeds(self):
        vals = [self._load("A0_frozen_final_cosine", s)["test_retmse10"] for s in SEEDS]
        assert vals[0] == pytest.approx(0.969165454248431, abs=1e-9)
        assert vals[0] == vals[1] == vals[2]

    def test_frozen_arms_have_zero_param_displacement(self):
        for arm in FROZEN_ARMS:
            for seed in SEEDS:
                m = self._load(arm, seed)
                assert m["encoder_param_displacement"] == 0.0, f"{arm} seed{seed} trunk drifted"

    def test_trainable_arms_have_nonzero_param_displacement(self):
        for arm in ("A4_continue_encoder_nohead", "A5_joint_encoder_freshasym"):
            for seed in SEEDS:
                m = self._load(arm, seed)
                assert m["encoder_param_displacement"] > 0.0, f"{arm} seed{seed} trunk did not move"

    def test_a2_a3_reproduce_control01_b0_b1(self):
        expected_a2 = [0.8149259850582774, 0.8403324266651724, 0.8085792432058635]
        expected_a3 = [0.8496652092362786, 0.8218177769687995, 0.8348129770764752]
        a2 = [self._load("A2_frozen_rawcls_freshasym", s)["test_retmse10"] for s in SEEDS]
        a3 = [self._load("A3_frozen_patchmean_freshasym", s)["test_retmse10"] for s in SEEDS]
        assert a2 == pytest.approx(expected_a2, abs=1e-9)
        assert a3 == pytest.approx(expected_a3, abs=1e-9)

    def test_batch_order_hash_identical_across_trained_arms_per_seed(self):
        trained_arms = [a for a in ARMS if a != "A0_frozen_final_cosine"]
        for seed in SEEDS:
            hashes = {a: self._load(a, seed)["batch_order_hashes"]["epoch1"] for a in trained_arms}
            assert len(set(hashes.values())) == 1, f"seed{seed} batch order mismatch: {hashes}"

    def test_no_nan_or_inf_in_any_summary_metric(self):
        import math
        numeric_keys = ("test_retmse10", "test_uniform_aggregate_mse10", "test_weighted_aggregate_mse10",
                        "test_recall10", "test_binary_oracle_ndcg10", "test_oracle_mean_rank",
                        "test_oracle_median_rank", "test_rank_fraction_mean")
        for arm in ARMS:
            for seed in SEEDS:
                m = self._load(arm, seed)
                for k in numeric_keys:
                    v = m[k]
                    assert not (math.isnan(v) or math.isinf(v)), f"{arm} seed{seed} {k}={v}"

    def test_per_query_csv_reconstructs_summary_mean(self):
        """Per-query CSV channel-mean over all queries must reproduce the
        summary JSON's test_retmse10 (both are computed by the same
        eval_epoch call; this checks the CSV write didn't silently drop
        or duplicate rows)."""
        arm, seed = "A1_frozen_final_freshasym", 0
        m = self._load(arm, seed)
        f = RESULT_DIR / f"per_query_{arm}_seed{seed}.csv"
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        vals = [float(r["individual_retmse10"]) for r in rows]
        reconstructed = sum(vals) / len(vals)
        assert reconstructed == pytest.approx(m["test_retmse10"], abs=1e-6)

    def test_per_query_csv_has_7_channels_per_query_start(self):
        arm, seed = "A1_frozen_final_freshasym", 0
        f = RESULT_DIR / f"per_query_{arm}_seed{seed}.csv"
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        from collections import Counter
        counts = Counter(r["query_start_idx"] for r in rows)
        assert set(counts.values()) == {7}, "expected exactly 7 channels per query_start_idx"
