"""
bridge/service.py
=================
AndrewMoltisBridge — orchestration layer between the Moltis runtime and the
Andrew/Romeo analytical+educational agents.

Responsibilities:
- Memory recall from Moltis before query execution
- Running the SwarmSupervisor pipeline (Andrew + Romeo)
- HITL gate for low-confidence results
- Memory storage after successful analysis
- Channel-aware response formatting
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

from bridge.client import MoltisClient, MoltisConfig
from bridge.hitl import HitlGate

logger = logging.getLogger("bridge_service")


class AndrewMoltisBridge:
    """
    Integration layer between Moltis (Rust runtime) and Andrew/Romeo agents.

    Lifecycle of a query:
    1. Moltis receives user message via Telegram/Discord/Web
    2. Moltis forwards to this bridge via webhook
    3. Bridge enriches query with relevant past analyses (memory recall)
    4. Bridge runs SwarmSupervisor (routes to Andrew, Romeo, or both)
    5. HITL gate reviews result if confidence < threshold
    6. Bridge returns structured result to Moltis for delivery
    7. Bridge optionally stores successful result in Moltis memory
    """

    def __init__(
        self,
        moltis_config: Optional[MoltisConfig] = None,
        andrew_db_url: Optional[str] = None,
        store_results_in_memory: bool = True,
    ):
        self.moltis = MoltisClient(moltis_config)
        self.store_results = store_results_in_memory
        self._andrew_executor = None
        self._scene_reviewer = None
        self._scene_ops_aggregator = None
        self._db_url = andrew_db_url or os.getenv("DATABASE_URL", "")
        self.hitl = HitlGate()

    def _get_executor(self):
        """Lazy-load SwarmSupervisor (routes to Andrew, Romeo, or both)."""
        if self._andrew_executor is None:
            from core.supervisor import SwarmSupervisor
            self._andrew_executor = SwarmSupervisor(db_url=self._db_url)
            logger.info("SwarmSupervisor v1.0.0 initialized (Andrew + Romeo)")
        return self._andrew_executor

    def _get_scene_reviewer(self):
        """Lazy-load the deterministic PinoCut scene reviewer."""
        if self._scene_reviewer is None:
            from bridge.pinocut_review import PinoCutSceneReviewer

            self._scene_reviewer = PinoCutSceneReviewer()
            logger.info("PinoCut scene reviewer initialized")
        return self._scene_reviewer

    def _get_scene_ops_aggregator(self):
        """Lazy-load the frontend-facing SceneOps aggregator."""
        if self._scene_ops_aggregator is None:
            from bridge.scene_ops import PinoCutSceneOpsAggregator

            self._scene_ops_aggregator = PinoCutSceneOpsAggregator(self._get_scene_reviewer())
            logger.info("SceneOps aggregator initialized")
        return self._scene_ops_aggregator

    async def handle_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point: receive a query, run the agent pipeline, return results.

        Args:
            query: Natural language analytical or educational question
            context: Optional dict with channel, user_id, session_id, etc.

        Returns:
            Structured result dict with narrative, confidence, cost, warnings
        """
        context = context or {}
        start_time = time.time()
        logger.info(f"Bridge received: {query[:80]}...")

        # 1. Recall relevant past analyses from Moltis memory
        memory_context = ""
        if self.store_results:
            try:
                memories = await self.moltis.recall_memory(query, limit=3)
                if memories:
                    memory_context = "\n".join(
                        f"- Prior analysis (relevance {m.get('score', 0):.2f}): {m.get('content', '')[:200]}"
                        for m in memories
                    )
                    logger.info(f"Found {len(memories)} relevant memories")
            except Exception as e:
                logger.warning(f"Memory recall failed (non-fatal): {e}")

        # 2. Run the agent pipeline (synchronous — offload to thread pool)
        executor = self._get_executor()
        enriched_query = query
        if memory_context:
            enriched_query = f"{query}\n\n[Context from prior analyses:\n{memory_context}]"

        result = await asyncio.to_thread(executor.execute, enriched_query)
        elapsed = time.time() - start_time

        # 3. Build structured response
        response = {
            "query": query,
            "narrative": result.output if hasattr(result, "output") else str(result.raw_data or ""),
            "sql_query": result.sql_query,
            "confidence": result.confidence,
            "cost_usd": result.cost_usd if hasattr(result, "cost_usd") else result.cost,
            "warnings": result.warnings if hasattr(result, "warnings") else [],
            "success": result.success,
            "error": result.error_message if hasattr(result, "error_message") else result.error,
            "elapsed_seconds": round(elapsed, 2),
            "routing": getattr(result, "routing", "unknown"),
            "model_used": getattr(result, "model_used", "unknown"),
            "agent_used": getattr(result, "agent_used", "andrew"),
            "channel": context.get("channel", "api"),
        }

        # 4. HITL gate — review low-confidence results before delivery
        hitl_outcome = await self.hitl.check(
            query=query,
            output=response["narrative"],
            confidence=response["confidence"],
            routing=response["routing"],
            agent_used=response["agent_used"],
            warnings=response["warnings"],
            cost_usd=response["cost_usd"],
            sql_query=response.get("sql_query"),
        )
        if hitl_outcome.triggered:
            response["narrative"] = hitl_outcome.output
            response["warnings"] = hitl_outcome.warnings
            response["hitl_decision"] = hitl_outcome.decision
            response["hitl_review_id"] = hitl_outcome.review_id
            if hitl_outcome.decision == "reject":
                response["success"] = False
                response["error"] = "Result rejected by human reviewer"
        else:
            response["hitl_decision"] = "skipped"
            response["hitl_review_id"] = None

        # 5. Format for Moltis channel delivery
        response["formatted_message"] = self._format_for_channel(response)

        # 6. Store in Moltis memory for future context
        if self.store_results and result.success:
            try:
                await self.moltis.store_memory(
                    content=f"Query: {query}\nResult: {response['narrative'][:500]}",
                    metadata={
                        "confidence": response["confidence"],
                        "sql": result.sql_query,
                        "timestamp": time.time(),
                        "cost": response["cost_usd"],
                    },
                )
                logger.info("Analysis stored in Moltis memory")
            except Exception as e:
                logger.warning(f"Memory store failed (non-fatal): {e}")

        logger.info(
            f"Bridge completed: confidence={response['confidence']:.2f}, "
            f"cost=${response['cost_usd']:.4f}, elapsed={elapsed:.1f}s"
        )
        return response

    async def handle_scene_review(
        self,
        scene_payload: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Review a PinoCut scene bundle and return Andrew-style QA output.

        This path is deterministic and uses the existing HITL gate for
        low-confidence scene reviews.
        """
        from bridge.schemas import SceneReviewRequest

        request = SceneReviewRequest(**scene_payload)
        response = await self._run_scene_review(request, context=context)
        response["formatted_message"] = self._format_scene_review_for_channel(response)
        return response

    def _get_ltx_pipeline(self):
        """Lazy-load the LTX video generation pipeline."""
        if not hasattr(self, "_ltx_pipeline") or self._ltx_pipeline is None:
            from bridge.ltx_video import LtxVideoPipeline
            self._ltx_pipeline = LtxVideoPipeline()
            logger.info("LtxVideoPipeline initialized")
        return self._ltx_pipeline

    async def handle_ltx_generate(
        self,
        payload: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Parse a Bassito ТЗ/scenario and return a ComfyUI-ready LTX job queue.

        Wraps LtxVideoPipeline.run() so the bridge can call it the same way
        it calls handle_scene_review / handle_scene_ops.
        """
        from bridge.schemas import LtxVideoJobRequest
        request = LtxVideoJobRequest(**payload)
        pipeline = self._get_ltx_pipeline()
        result = pipeline.run(request)
        return result.model_dump()

    async def handle_scene_ops(
        self,
        scene_payload: Dict[str, Any],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate PinoCut scene data, Andrew review, and Bassito job state into
        the frontend-facing SceneOps snapshot used by RomeoFlexVision.
        """
        from bridge.schemas import SceneOpsAggregateRequest

        request = SceneOpsAggregateRequest(**scene_payload)
        review = await self._run_scene_review(request, context=context)
        aggregator = self._get_scene_ops_aggregator()
        return aggregator.build_snapshot(request, review=review)

    async def _run_scene_review(
        self,
        request,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        channel = (context or {}).get("channel", "api")
        start_time = time.time()
        reviewer = self._get_scene_reviewer()
        response = reviewer.review(request)
        response["elapsed_seconds"] = round(time.time() - start_time, 2)
        response["agent_used"] = "andrew_scene_review"
        response["routing"] = "pinocut_scene_review"
        response["channel"] = channel

        hitl_outcome = await self.hitl.check(
            query=f"Review scene {request.scene_id}: {request.scene_goal}",
            output=response["summary"],
            confidence=response["confidence"],
            routing=response["routing"],
            agent_used=response["agent_used"],
            warnings=response["warnings"],
            cost_usd=0.0,
            sql_query=None,
        )
        response["summary"] = hitl_outcome.output
        response["warnings"] = hitl_outcome.warnings
        response["hitl_decision"] = hitl_outcome.decision
        response["hitl_review_id"] = hitl_outcome.review_id
        if hitl_outcome.decision == "reject":
            response["success"] = False

        return response

    def _format_for_channel(self, response: Dict) -> str:
        """Format output for messaging channels (Telegram/Discord ~4000 char limit)."""
        parts = []

        if response.get("success"):
            parts.append(f"**Analysis** (confidence: {response['confidence']:.0%})")
            narrative = response.get("narrative", "")
            if len(narrative) > 2000:
                narrative = narrative[:2000] + "...\n_(truncated — full report available via API)_"
            parts.append(narrative)
        else:
            parts.append("**Analysis failed**")
            parts.append(f"Error: {response.get('error', 'Unknown')}")

        if response.get("warnings"):
            parts.append("\n**Warnings:**")
            for w in response["warnings"][:3]:
                parts.append(f"- {w}")

        parts.append(
            f"\n_Cost: ${response.get('cost_usd', 0):.4f} | "
            f"Time: {response.get('elapsed_seconds', 0)}s | "
            f"Route: {response.get('routing', '?')}_"
        )
        return "\n".join(parts)

    def _format_scene_review_for_channel(self, response: Dict[str, Any]) -> str:
        """Format a PinoCut scene review for Telegram/Discord/Web delivery."""
        parts = [
            f"**Scene Review** `{response.get('scene_id', '?')}` "
            f"(confidence: {response.get('confidence', 0):.0%})",
            response.get("summary", ""),
        ]

        recommended_actions = response.get("recommended_actions", [])
        if recommended_actions:
            parts.append("\n**Recommended actions:**")
            for action in recommended_actions[:5]:
                parts.append(f"- {action}")

        warnings = response.get("warnings", [])
        if warnings:
            parts.append("\n**Warnings:**")
            for warning in warnings[:5]:
                parts.append(f"- {warning}")

        breakdown = response.get("quality_breakdown", {})
        if breakdown:
            parts.append(
                "\n_"
                + " | ".join(
                    f"{key.replace('_', ' ')}: {value:.1f}/5"
                    for key, value in breakdown.items()
                )
                + "_"
            )

        parts.append(
            f"\n_Route: {response.get('routing', '?')} | "
            f"Time: {response.get('elapsed_seconds', 0)}s_"
        )
        return "\n".join(parts)

    async def handle_scheduled_task(self, task: str) -> Dict[str, Any]:
        """Handle a cron-triggered analytical task."""
        logger.info(f"Scheduled task triggered: {task[:80]}")
        return await self.handle_query(task, context={"channel": "cron", "scheduled": True})

    async def use_moltis_sandbox(self, code: str) -> Dict[str, Any]:
        """Execute Python code via Moltis's Docker sandbox (replaces E2B)."""
        return await self.moltis.execute_in_sandbox(code, language="python")

    async def close(self):
        await self.moltis.close()
