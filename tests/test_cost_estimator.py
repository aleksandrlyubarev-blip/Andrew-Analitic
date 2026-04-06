"""
tests/test_cost_estimator.py
=============================
Tests for CostEstimator and BatchCostEstimate.

CE-series:
  CE1.  veo3.1 scene → backend="veo", non-zero cost.
  CE2.  grok4.2 scene → backend="grok", non-zero cost.
  CE3.  ltx2.3 scene → backend="comfyui", cost=0.
  CE4.  empty preferred_model → backend="comfyui", cost=0.
  CE5.  cost = duration_sec × price_per_sec for Veo.
  CE6.  cost = duration_sec × price_per_sec for Grok.
  CE7.  total_cost = sum of all scene costs.
  CE8.  sequential_time = sum of all gen_time_sec.
  CE9.  parallel_time ≤ sequential_time for multi-scene batch.
  CE10. by_backend groups cost correctly.
  CE11. estimate_scenario() parses scenario text correctly.
  CE12. fallback_to_comfyui=True adds ComfyUI overhead to cloud scenes.
  CE13. format_report() contains scene_id, cost, backend.
  CE14. veo aliases (veo-3.1, veo3) map to "veo" backend.
  CE15. grok aliases (grok-4.2, grok4) map to "grok" backend.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.cost_estimator import (
    CostEstimator,
    _PRICE_PER_SEC,
    _TIME_FACTOR,
    _backend_key,
)
from bridge.schemas import LtxGenerationConfig, LtxKeyframe, LtxSceneJob


def _make_job(model: str, duration: float = 5.0, scene_id: str = "scene_01") -> LtxSceneJob:
    return LtxSceneJob(
        job_id=f"proj_{scene_id}",
        scene_id=scene_id,
        scene_index=0,
        workflow="first_last",
        duration_sec=duration,
        prompts=["test"],
        preferred_model=model,
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="t"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="t"),
        ],
        model_config_ltx={"variant": "fp8", "resolution": "1920x1080",
                          "aspect_ratio": "16:9", "vram_budget_gb": 12},
    )


def test_ce1_veo_backend():
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("veo3.1")])
    assert result.scenes[0].backend == "veo"
    assert result.scenes[0].cost_usd > 0


def test_ce2_grok_backend():
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("grok4.2")])
    assert result.scenes[0].backend == "grok"
    assert result.scenes[0].cost_usd > 0


def test_ce3_ltx_backend_free():
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("ltx2.3")])
    assert result.scenes[0].backend == "comfyui"
    assert result.scenes[0].cost_usd == 0.0


def test_ce4_empty_model_comfyui():
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("")])
    assert result.scenes[0].backend == "comfyui"
    assert result.scenes[0].cost_usd == 0.0


def test_ce5_veo_cost_formula():
    duration = 7.0
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("veo3.1", duration)])
    expected = round(duration * _PRICE_PER_SEC["veo"], 4)
    assert result.scenes[0].cost_usd == pytest.approx(expected)


def test_ce6_grok_cost_formula():
    duration = 13.0
    est = CostEstimator()
    result = est.estimate_jobs([_make_job("grok4.2", duration)])
    expected = round(duration * _PRICE_PER_SEC["grok"], 4)
    assert result.scenes[0].cost_usd == pytest.approx(expected)


def test_ce7_total_cost_is_sum():
    jobs = [
        _make_job("veo3.1",  5.0, "s1"),
        _make_job("grok4.2", 13.0, "s2"),
        _make_job("",        7.0, "s3"),
    ]
    est = CostEstimator()
    result = est.estimate_jobs(jobs)
    assert result.total_cost_usd == pytest.approx(
        sum(s.cost_usd for s in result.scenes)
    )


def test_ce8_sequential_time_is_sum():
    jobs = [_make_job("veo3.1", 5.0, "s1"), _make_job("grok4.2", 10.0, "s2")]
    est = CostEstimator()
    result = est.estimate_jobs(jobs)
    assert result.sequential_time_sec == pytest.approx(
        sum(s.gen_time_sec for s in result.scenes)
    )


def test_ce9_parallel_time_le_sequential():
    jobs = [_make_job("veo3.1", d, f"s{i}") for i, d in enumerate([5, 7, 8, 10])]
    est = CostEstimator(max_concurrent=4)
    result = est.estimate_jobs(jobs)
    assert result.parallel_time_sec <= result.sequential_time_sec


def test_ce10_by_backend_grouping():
    jobs = [
        _make_job("veo3.1",  5.0, "s1"),
        _make_job("veo3.1",  7.0, "s2"),
        _make_job("grok4.2", 10.0, "s3"),
    ]
    est = CostEstimator()
    result = est.estimate_jobs(jobs)
    # "veo" bucket should be sum of two veo scenes
    expected_veo = round(
        5.0 * _PRICE_PER_SEC["veo"] + 7.0 * _PRICE_PER_SEC["veo"], 4
    )
    assert result.by_backend["veo"] == pytest.approx(expected_veo)
    assert "grok" in result.by_backend
    assert "comfyui" not in result.by_backend  # $0 backends omitted


def test_ce11_estimate_scenario_parses_correctly():
    scenario = """\
## Scene 1 (0:00–0:05)
[VISUAL: forest at dusk]
[MODEL: veo3.1]

## Scene 2 (0:05–0:18)
[VISUAL: sword fight]
[MODEL: grok4.2]
"""
    est = CostEstimator()
    result = est.estimate_scenario(scenario)
    assert len(result.scenes) == 2
    assert result.scenes[0].backend == "veo"
    assert result.scenes[1].backend == "grok"


def test_ce12_fallback_adds_comfyui_overhead():
    job = _make_job("veo3.1", 5.0)
    est_no_fb  = CostEstimator(fallback_to_comfyui=False)
    est_with_fb = CostEstimator(fallback_to_comfyui=True)
    r_no = est_no_fb.estimate_jobs([job])
    r_fb = est_with_fb.estimate_jobs([job])
    # fallback adds ComfyUI time (free, so cost stays same; time increases)
    assert r_fb.scenes[0].gen_time_sec > r_no.scenes[0].gen_time_sec


def test_ce13_format_report_contains_scene_info():
    jobs = [_make_job("veo3.1", 5.0, "scene_01"), _make_job("grok4.2", 10.0, "scene_02")]
    est = CostEstimator()
    result = est.estimate_jobs(jobs)
    report = est.format_report(result)
    assert "scene_01" in report
    assert "scene_02" in report
    assert "Veo" in report
    assert "Grok" in report
    assert "$" in report


def test_ce14_veo_aliases():
    for alias in ["veo-3.1", "veo3", "veo-3"]:
        assert _backend_key(alias) == "veo", f"alias {alias!r} not mapped to veo"


def test_ce15_grok_aliases():
    for alias in ["grok-4.2", "grok4", "grok-4"]:
        assert _backend_key(alias) == "grok", f"alias {alias!r} not mapped to grok"
