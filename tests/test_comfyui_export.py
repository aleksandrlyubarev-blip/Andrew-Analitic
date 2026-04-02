"""
tests/test_comfyui_export.py
============================
Tests for the ComfyUI workflow exporter.

Coverage:
  E1.  first_last job → workflow has all required node class types.
  E2.  multi_keyframe job → same required node types.
  E3.  Node IDs are unique sequential strings.
  E4.  Positive prompt appears in a CLIPTextEncode node.
  E5.  VAEDecode feeds SaveVideo (output chain integrity).
  E6.  double_upscale job → ImageScaleBy node is present.
  E7.  No double_upscale → ImageScaleBy absent.
  E8.  Keyframe with image_path → LoadImage node present.
  E9.  Keyframe without image_path → no LoadImage node.
  E10. wrap_for_api adds 'prompt' and 'client_id' keys.
  E11. export_batch returns one workflow per job.
  E12. fp8 + distilled → ckpt_name contains 'distilled-fp8'.
  E13. bf16 no-lora → ckpt_name contains 'bf16'.
  E14. Distilled lora → steps == 8; full model → steps == 25.
  E15. Resolution parsed correctly from generation_resolution.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.comfyui_export import (
    NODE_CLIP_ENCODE,
    NODE_LOAD_IMAGE,
    NODE_MODEL_LOADER,
    NODE_SAMPLER,
    NODE_SAVE_VIDEO,
    NODE_VAE_DECODE,
    ComfyUIWorkflowExporter,
)
from bridge.schemas import LtxGenerationConfig, LtxKeyframe, LtxSceneJob

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_job(
    workflow: str = "first_last",
    prompts: list[str] | None = None,
    keyframes: list[LtxKeyframe] | None = None,
    model_config_ltx: dict | None = None,
) -> LtxSceneJob:
    if keyframes is None:
        keyframes = [
            LtxKeyframe(index=0, frame_type="first", source_prompt="test scene"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="test scene"),
        ]
    return LtxSceneJob(
        job_id="test_proj_scene_01_abc123",
        scene_id="scene_01",
        scene_index=0,
        workflow=workflow,
        duration_sec=8.0,
        prompts=prompts or ["neon corridor, cinematic neon-tech"],
        keyframes=keyframes,
        model_config_ltx=model_config_ltx or {
            "variant": "fp8",
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "vram_budget_gb": 12,
            "lora": "distilled",
            "upscale_pass": "double",
            "generation_resolution": "960x544",
        },
    )


def _class_types(workflow: dict) -> set[str]:
    return {node["class_type"] for node in workflow.values()}


def _nodes_of_type(workflow: dict, class_type: str) -> list[dict]:
    return [n for n in workflow.values() if n["class_type"] == class_type]


EXPORTER = ComfyUIWorkflowExporter()
DEFAULT_CONFIG = LtxGenerationConfig()


# ── E1: first_last required nodes ─────────────────────────────────────────────

def test_e1_first_last_required_nodes():
    job = _make_job(workflow="first_last")
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    types = _class_types(wf)
    assert NODE_MODEL_LOADER in types
    assert NODE_CLIP_ENCODE in types
    assert NODE_SAMPLER in types
    assert NODE_VAE_DECODE in types
    assert NODE_SAVE_VIDEO in types


# ── E2: multi_keyframe required nodes ─────────────────────────────────────────

def test_e2_multi_keyframe_required_nodes():
    kfs = [
        LtxKeyframe(index=0, frame_type="first", source_prompt="wide shot"),
        LtxKeyframe(index=1, frame_type="middle", source_prompt="mid action"),
        LtxKeyframe(index=-1, frame_type="last", source_prompt="end frame"),
    ]
    job = _make_job(workflow="multi_keyframe", keyframes=kfs)
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    types = _class_types(wf)
    assert NODE_MODEL_LOADER in types
    assert NODE_SAMPLER in types
    assert NODE_VAE_DECODE in types
    assert NODE_SAVE_VIDEO in types


# ── E3: unique sequential node IDs ────────────────────────────────────────────

def test_e3_unique_sequential_node_ids():
    job = _make_job()
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    ids = list(wf.keys())
    assert len(ids) == len(set(ids)), "Duplicate node IDs"
    int_ids = [int(i) for i in ids]
    assert int_ids == sorted(int_ids), "IDs not sequential"
    assert int_ids[0] == 1


# ── E4: positive prompt in CLIPTextEncode ─────────────────────────────────────

def test_e4_positive_prompt_in_clip_encode():
    job = _make_job(prompts=["robot assembling itself, hard sci-fi"])
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    clip_nodes = _nodes_of_type(wf, NODE_CLIP_ENCODE)
    texts = [n["inputs"]["text"] for n in clip_nodes]
    assert any("robot assembling itself" in t for t in texts)


# ── E5: decode feeds save ─────────────────────────────────────────────────────

def test_e5_decode_feeds_save():
    # No upscale so decode → save is direct
    job = _make_job(
        model_config_ltx={
            "variant": "fp8",
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "vram_budget_gb": 12,
        }
    )
    config = LtxGenerationConfig(double_upscale=False)
    wf = EXPORTER.export_job(job, config)

    # Find decode node id
    decode_id = next(k for k, v in wf.items() if v["class_type"] == NODE_VAE_DECODE)
    save_node = next(v for v in wf.values() if v["class_type"] == NODE_SAVE_VIDEO)
    images_link = save_node["inputs"]["images"]
    assert images_link[0] == decode_id


# ── E6: double_upscale adds ImageScaleBy ─────────────────────────────────────

def test_e6_double_upscale_has_imagescaleby():
    job = _make_job()  # default has upscale_pass=double
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    assert "ImageScaleBy" in _class_types(wf)


# ── E7: no double_upscale → no ImageScaleBy ──────────────────────────────────

def test_e7_no_upscale_no_imagescaleby():
    job = _make_job(
        model_config_ltx={
            "variant": "fp8",
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "vram_budget_gb": 12,
        }
    )
    config = LtxGenerationConfig(double_upscale=False)
    wf = EXPORTER.export_job(job, config)
    assert "ImageScaleBy" not in _class_types(wf)


# ── E8: keyframe with image_path → LoadImage ─────────────────────────────────

def test_e8_keyframe_with_image_path_has_loadimage():
    kfs = [
        LtxKeyframe(index=0, frame_type="first", source_prompt="p", image_path="frames/first.png"),
        LtxKeyframe(index=-1, frame_type="last", source_prompt="p"),
    ]
    job = _make_job(keyframes=kfs)
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    load_nodes = _nodes_of_type(wf, NODE_LOAD_IMAGE)
    assert len(load_nodes) == 1
    assert load_nodes[0]["inputs"]["image"] == "frames/first.png"


# ── E9: keyframes without image_path → no LoadImage ──────────────────────────

def test_e9_no_image_path_no_loadimage():
    job = _make_job()  # default keyframes have no image_path
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    assert _nodes_of_type(wf, NODE_LOAD_IMAGE) == []


# ── E10: wrap_for_api structure ───────────────────────────────────────────────

def test_e10_wrap_for_api():
    job = _make_job()
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    wrapped = EXPORTER.wrap_for_api(wf, client_id="test-client-001")
    assert "prompt" in wrapped
    assert "client_id" in wrapped
    assert wrapped["client_id"] == "test-client-001"
    assert wrapped["prompt"] is wf


# ── E11: export_batch returns one workflow per job ────────────────────────────

def test_e11_export_batch_count():
    jobs = [_make_job(), _make_job(workflow="multi_keyframe")]
    results = EXPORTER.export_batch(jobs, DEFAULT_CONFIG)
    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)


# ── E12: fp8 + distilled ckpt name ───────────────────────────────────────────

def test_e12_fp8_distilled_ckpt():
    config = LtxGenerationConfig(model_variant="fp8", use_distilled_lora=True)
    job = _make_job()
    wf = EXPORTER.export_job(job, config)
    loader = _nodes_of_type(wf, NODE_MODEL_LOADER)[0]
    assert "distilled-fp8" in loader["inputs"]["ckpt_name"]


# ── E13: bf16 no-lora ckpt name ──────────────────────────────────────────────

def test_e13_bf16_nolora_ckpt():
    config = LtxGenerationConfig(model_variant="bf16", use_distilled_lora=False)
    job = _make_job(model_config_ltx={"variant": "bf16", "resolution": "1920x1080", "aspect_ratio": "16:9", "vram_budget_gb": 20})
    wf = EXPORTER.export_job(job, config)
    loader = _nodes_of_type(wf, NODE_MODEL_LOADER)[0]
    assert "bf16" in loader["inputs"]["ckpt_name"]
    assert "distilled" not in loader["inputs"]["ckpt_name"]


# ── E14: steps (distilled vs full) ───────────────────────────────────────────

def test_e14_steps_distilled_vs_full():
    distilled_config = LtxGenerationConfig(use_distilled_lora=True)
    full_config = LtxGenerationConfig(use_distilled_lora=False)
    job = _make_job()

    wf_dist = EXPORTER.export_job(job, distilled_config)
    wf_full = EXPORTER.export_job(job, full_config)

    sampler_dist = _nodes_of_type(wf_dist, NODE_SAMPLER)[0]
    sampler_full = _nodes_of_type(wf_full, NODE_SAMPLER)[0]

    assert sampler_dist["inputs"]["steps"] == 8
    assert sampler_full["inputs"]["steps"] == 25


# ── E15: generation_resolution parsed into latent dims ───────────────────────

def test_e15_generation_resolution_latent():
    job = _make_job(
        model_config_ltx={
            "variant": "fp8",
            "resolution": "1920x1080",
            "aspect_ratio": "16:9",
            "vram_budget_gb": 12,
            "lora": "distilled",
            "upscale_pass": "double",
            "generation_resolution": "960x544",
        }
    )
    wf = EXPORTER.export_job(job, DEFAULT_CONFIG)
    latent_node = _nodes_of_type(wf, "LTXVEmptyLatentVideo")[0]
    assert latent_node["inputs"]["width"] == 960
    assert latent_node["inputs"]["height"] == 544
