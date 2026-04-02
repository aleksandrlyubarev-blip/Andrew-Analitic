"""
bridge/comfyui_export.py
========================
Converts LtxSceneJob descriptors into ComfyUI API-compatible workflow JSON.

The output of `ComfyUIWorkflowExporter.export_job()` can be submitted directly
to a running ComfyUI instance via:

    POST http://127.0.0.1:8188/prompt
    Content-Type: application/json
    Body: {"prompt": <export_job output>, "client_id": "<uuid>"}

Node class names follow the ComfyUI-LTXVideo extension
(https://github.com/Lightricks/ComfyUI-LTXVideo).  Update NODE_CLASS_* constants
if the extension renames nodes in a future release.

Supported workflows
-------------------
first_last     — text + optional first-frame image + optional last-frame image.
multi_keyframe — text + N ordered keyframe images at evenly-spaced positions.
"""
from __future__ import annotations

import uuid
from typing import Any

from bridge.schemas import LtxGenerationConfig, LtxKeyframe, LtxSceneJob

# ── ComfyUI node class names (ComfyUI-LTXVideo extension) ────────────────────

NODE_MODEL_LOADER = "LTXVLoader"
NODE_CLIP_ENCODE = "CLIPTextEncode"
NODE_LOAD_IMAGE = "LoadImage"
NODE_LATENT_VIDEO = "LTXVEmptyLatentVideo"
NODE_CONDITIONING = "LTXVConditioning"
NODE_KEYFRAME_APPEND = "LTXVScheduledCFGKeyframe"
NODE_SAMPLER = "LTXVSampler"
NODE_VAE_DECODE = "LTXVDecode"
NODE_SAVE_VIDEO = "VHS_VideoCombine"

# ── Default generation parameters ────────────────────────────────────────────

DEFAULT_STEPS = 25
DEFAULT_CFG = 3.5
DEFAULT_FPS = 24
NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"


class _NodeGraph:
    """Mutable graph that assigns sequential string IDs to nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._counter = 1

    def add(self, class_type: str, inputs: dict[str, Any]) -> str:
        node_id = str(self._counter)
        self._counter += 1
        self._nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id

    def link(self, node_id: str, output_index: int = 0) -> list:
        """Return a ComfyUI link reference: [node_id, output_index]."""
        return [node_id, output_index]

    def build(self) -> dict[str, Any]:
        return dict(self._nodes)


class ComfyUIWorkflowExporter:
    """
    Convert a single LtxSceneJob into a ComfyUI prompt-API payload.

    Usage::

        exporter = ComfyUIWorkflowExporter()
        payload = exporter.export_job(job, config)
        # POST payload to http://127.0.0.1:8188/prompt
    """

    def export_job(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Return a dict suitable for the ComfyUI ``/prompt`` API body."""
        config = config or LtxGenerationConfig()
        g = _NodeGraph()

        # ── 1. Model ─────────────────────────────────────────────────────────
        model_node = g.add(
            NODE_MODEL_LOADER,
            {
                "ckpt_name": self._ckpt_name(config),
                "dtype": config.model_variant,
            },
        )
        # model output slots: 0=model, 1=vae, 2=clip

        # ── 2. Text conditioning ─────────────────────────────────────────────
        positive_text = job.prompts[0] if job.prompts else ""
        pos_node = g.add(
            NODE_CLIP_ENCODE,
            {"text": positive_text, "clip": g.link(model_node, 2)},
        )
        neg_node = g.add(
            NODE_CLIP_ENCODE,
            {"text": NEGATIVE_PROMPT, "clip": g.link(model_node, 2)},
        )

        # ── 3. Latent canvas ─────────────────────────────────────────────────
        width, height = self._parse_resolution(
            job.model_config_ltx.get("generation_resolution")
            or job.model_config_ltx.get("resolution", "1920x1080")
        )
        n_frames = int(job.duration_sec * DEFAULT_FPS)
        latent_node = g.add(
            NODE_LATENT_VIDEO,
            {"width": width, "height": height, "length": n_frames, "batch_size": 1},
        )

        # ── 4. Keyframe image nodes ───────────────────────────────────────────
        image_nodes = self._add_keyframe_images(g, job.keyframes)

        # ── 5. LTX conditioning (stitches keyframes into latent) ──────────────
        conditioning_node = g.add(
            NODE_CONDITIONING,
            {
                "positive": g.link(pos_node),
                "negative": g.link(neg_node),
                "latent": g.link(latent_node),
                **self._keyframe_conditioning_inputs(image_nodes, job.keyframes, n_frames),
            },
        )

        # ── 6. Sampler ───────────────────────────────────────────────────────
        steps = self._steps(config)
        sampler_node = g.add(
            NODE_SAMPLER,
            {
                "model": g.link(model_node, 0),
                "conditioning": g.link(conditioning_node),
                "steps": steps,
                "cfg": DEFAULT_CFG,
                "seed": 0,
                "sampler_name": "euler",
                "scheduler": "linear",
            },
        )

        # ── 7. Decode ────────────────────────────────────────────────────────
        decode_node = g.add(
            NODE_VAE_DECODE,
            {"samples": g.link(sampler_node), "vae": g.link(model_node, 1)},
        )

        # ── 8. Optional upscale pass ─────────────────────────────────────────
        output_node_ref = g.link(decode_node)
        if job.model_config_ltx.get("upscale_pass") == "double":
            upscale_node = g.add(
                "ImageScaleBy",
                {
                    "image": g.link(decode_node),
                    "upscale_method": "lanczos",
                    "scale_by": self._upscale_factor(
                        job.model_config_ltx.get("generation_resolution"),
                        job.model_config_ltx.get("resolution", "1920x1080"),
                    ),
                },
            )
            output_node_ref = g.link(upscale_node)

        # ── 9. Save video ─────────────────────────────────────────────────────
        g.add(
            NODE_SAVE_VIDEO,
            {
                "images": output_node_ref,
                "frame_rate": DEFAULT_FPS,
                "loop_count": 0,
                "filename_prefix": job.scene_id,
                "format": "video/h264-mp4",
                "save_output": True,
            },
        )

        return g.build()

    def export_batch(
        self,
        jobs: list[LtxSceneJob],
        config: LtxGenerationConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Export multiple jobs, one ComfyUI workflow per scene."""
        return [self.export_job(job, config) for job in jobs]

    def wrap_for_api(
        self,
        workflow: dict[str, Any],
        client_id: str | None = None,
    ) -> dict[str, Any]:
        """Wrap a workflow dict in the ComfyUI /prompt request envelope."""
        return {
            "prompt": workflow,
            "client_id": client_id or str(uuid.uuid4()),
        }

    # ── private helpers ───────────────────────────────────────────────────────

    def _ckpt_name(self, config: LtxGenerationConfig) -> str:
        if config.model_variant == "fp8":
            suffix = "fp8" if not config.use_distilled_lora else "distilled-fp8"
        else:
            suffix = "distilled" if config.use_distilled_lora else "bf16"
        return f"ltxv-2b-0.9.6-{suffix}.safetensors"

    def _parse_resolution(self, res: str) -> tuple[int, int]:
        """Parse 'WxH' or 'W×H' string into (width, height) integers."""
        for sep in ("x", "X", "\u00d7"):
            if sep in res:
                w, h = res.split(sep, 1)
                return int(w.strip()), int(h.strip())
        return 960, 544  # safe fallback for double_upscale generation pass

    def _steps(self, config: LtxGenerationConfig) -> int:
        # Distilled LoRA converges faster — fewer steps needed.
        return 8 if config.use_distilled_lora else DEFAULT_STEPS

    def _add_keyframe_images(
        self,
        g: _NodeGraph,
        keyframes: list[LtxKeyframe],
    ) -> dict[int, str]:
        """
        Add LoadImage nodes for keyframes that have an image_path.
        Returns {keyframe.index: node_id}.
        """
        image_nodes: dict[int, str] = {}
        for kf in keyframes:
            if kf.image_path:
                node_id = g.add(
                    NODE_LOAD_IMAGE,
                    {"image": kf.image_path},
                )
                image_nodes[kf.index] = node_id
        return image_nodes

    def _keyframe_conditioning_inputs(
        self,
        image_nodes: dict[int, str],
        keyframes: list[LtxKeyframe],
        n_frames: int,
    ) -> dict[str, Any]:
        """
        Build the image conditioning inputs for LTXVConditioning.

        Maps keyframe positions to frame indices in [0, n_frames-1].
        """
        extra: dict[str, Any] = {}
        for kf in keyframes:
            if kf.index not in image_nodes:
                continue
            frame_idx = 0 if kf.index == 0 else (n_frames - 1 if kf.index == -1 else
                int((kf.index / max(1, len(keyframes) - 1)) * (n_frames - 1)))
            slot_key = f"keyframe_image_{kf.frame_type}"
            extra[slot_key] = [image_nodes[kf.index], 0]
            extra[f"keyframe_frame_{kf.frame_type}"] = frame_idx
        return extra

    def _upscale_factor(
        self,
        gen_res: str | None,
        target_res: str,
    ) -> float:
        """Compute scale factor from generation resolution to target resolution."""
        if not gen_res:
            return 2.0
        gw, _ = self._parse_resolution(gen_res)
        tw, _ = self._parse_resolution(target_res)
        if gw == 0:
            return 2.0
        return round(tw / gw, 4)
