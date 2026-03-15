"""
Human-in-the-Loop (HITL) Gate — bridge/hitl.py
===================================================
Intercepts low-confidence agent results and requests human review before
the response is returned to the user.

Review protocol (webhook-based):
  1. POST the result to HITL_WEBHOOK_URL with a unique review_id.
  2. Poll HITL_WEBHOOK_URL/<review_id> for a decision (approve/reject/modify).
  3. On timeout use HITL_ON_TIMEOUT behaviour (approve or reject).
  4. Return a HitlOutcome describing what happened.

Webhook contract
----------------
POST  {webhook_url}
      Content-Type: application/json
      Body: ReviewRequest (see dataclass below)

GET   {webhook_url}/{review_id}
      Response 200: ReviewDecision JSON  — decision ready
      Response 202/404: still pending    — keep polling

ReviewDecision fields
---------------------
  decision        : "approve" | "reject" | "modify"
  modified_output : str | None   (only used when decision == "modify")
  reviewer_note   : str | None   (optional human note, appended to warnings)

Environment variables
---------------------
  HITL_ENABLED               : "true" / "false"  (default: false)
  HITL_WEBHOOK_URL           : full URL for POST/GET  (required if enabled)
  HITL_CONFIDENCE_THRESHOLD  : float 0-1  (default: 0.5)
  HITL_TIMEOUT_SECONDS       : int        (default: 120)
  HITL_POLL_INTERVAL_SECONDS : float      (default: 3.0)
  HITL_ON_TIMEOUT            : "approve" / "reject"  (default: "approve")
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("hitl")


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class HitlConfig:
    enabled: bool = False
    webhook_url: str = ""
    confidence_threshold: float = 0.5
    timeout_seconds: int = 120
    poll_interval_seconds: float = 3.0
    on_timeout: str = "approve"   # "approve" | "reject"

    @classmethod
    def from_env(cls) -> "HitlConfig":
        return cls(
            enabled=os.getenv("HITL_ENABLED", "false").lower() == "true",
            webhook_url=os.getenv("HITL_WEBHOOK_URL", ""),
            confidence_threshold=float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.5")),
            timeout_seconds=int(os.getenv("HITL_TIMEOUT_SECONDS", "120")),
            poll_interval_seconds=float(os.getenv("HITL_POLL_INTERVAL_SECONDS", "3.0")),
            on_timeout=os.getenv("HITL_ON_TIMEOUT", "approve").lower(),
        )


# ============================================================
# 2. WIRE TYPES
# ============================================================

@dataclass
class ReviewRequest:
    """Payload sent to the reviewer webhook."""
    review_id: str
    timestamp: float
    query: str
    output: str
    confidence: float
    routing: str
    agent_used: str
    warnings: List[str]
    cost_usd: float
    sql_query: Optional[str] = None


@dataclass
class ReviewDecision:
    """Response from the reviewer webhook."""
    decision: str                   # "approve" | "reject" | "modify"
    modified_output: Optional[str] = None
    reviewer_note: Optional[str] = None


@dataclass
class HitlOutcome:
    """Result of the HITL gate check, returned to the bridge."""
    triggered: bool                 # Was HITL invoked at all?
    decision: str                   # "approve" | "reject" | "modify" | "skipped" | "timeout_approve" | "timeout_reject"
    output: str                     # Final output (may be modified)
    warnings: List[str]             # Warnings (may include reviewer note)
    timed_out: bool = False
    review_id: Optional[str] = None


# ============================================================
# 3. HITL GATE
# ============================================================

class HitlGate:
    """
    Checks agent confidence and conditionally requests human review.
    Thread-safe: each call gets its own review_id and httpx client.
    """

    def __init__(self, config: Optional[HitlConfig] = None):
        self.config = config or HitlConfig.from_env()

    def needs_review(self, confidence: float) -> bool:
        """True if HITL is enabled and confidence is below threshold."""
        return self.config.enabled and confidence < self.config.confidence_threshold

    async def check(
        self,
        query: str,
        output: str,
        confidence: float,
        routing: str = "unknown",
        agent_used: str = "unknown",
        warnings: Optional[List[str]] = None,
        cost_usd: float = 0.0,
        sql_query: Optional[str] = None,
    ) -> HitlOutcome:
        """
        Main entry point.  Returns immediately if HITL not triggered.
        Blocks (async) up to timeout_seconds waiting for human decision if triggered.
        """
        warnings = list(warnings or [])

        if not self.needs_review(confidence):
            return HitlOutcome(
                triggered=False,
                decision="skipped",
                output=output,
                warnings=warnings,
            )

        if not self.config.webhook_url:
            logger.warning("HITL enabled but HITL_WEBHOOK_URL not set — passing through")
            warnings.append(f"HITL: low confidence ({confidence:.2f}) but no webhook configured")
            return HitlOutcome(
                triggered=True,
                decision="approve",
                output=output,
                warnings=warnings,
            )

        review_id = str(uuid.uuid4())
        request = ReviewRequest(
            review_id=review_id,
            timestamp=time.time(),
            query=query,
            output=output,
            confidence=confidence,
            routing=routing,
            agent_used=agent_used,
            warnings=warnings,
            cost_usd=cost_usd,
            sql_query=sql_query,
        )

        logger.info(
            f"HITL triggered: confidence={confidence:.2f} < {self.config.confidence_threshold:.2f} "
            f"| review_id={review_id[:8]}"
        )

        decision = await self._request_review(request)
        return self._apply_decision(decision, output, warnings, review_id)

    async def _request_review(self, request: ReviewRequest) -> ReviewDecision:
        """POST the review request and poll for a decision."""
        payload = {
            "review_id": request.review_id,
            "timestamp": request.timestamp,
            "query": request.query,
            "output": request.output[:3000],  # truncate for webhook readability
            "confidence": request.confidence,
            "routing": request.routing,
            "agent_used": request.agent_used,
            "warnings": request.warnings,
            "cost_usd": request.cost_usd,
            "sql_query": request.sql_query,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Step 1: POST the review
            try:
                post_resp = await client.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                post_resp.raise_for_status()
                logger.info(f"HITL review POSTed: {post_resp.status_code}")

                # If the webhook returns a synchronous decision (200 + JSON), use it
                if post_resp.status_code == 200:
                    body = post_resp.json()
                    if "decision" in body:
                        logger.info(f"HITL synchronous decision: {body['decision']}")
                        return ReviewDecision(
                            decision=body.get("decision", "approve"),
                            modified_output=body.get("modified_output"),
                            reviewer_note=body.get("reviewer_note"),
                        )

            except Exception as e:
                logger.warning(f"HITL webhook POST failed: {e} — using on_timeout behavior")
                return self._timeout_decision()

            # Step 2: Poll for async decision
            poll_url = f"{self.config.webhook_url.rstrip('/')}/{request.review_id}"
            deadline = time.monotonic() + self.config.timeout_seconds
            interval = self.config.poll_interval_seconds

            while time.monotonic() < deadline:
                await asyncio.sleep(interval)
                try:
                    poll_resp = await client.get(poll_url, timeout=5.0)
                    if poll_resp.status_code == 200:
                        body = poll_resp.json()
                        if "decision" in body:
                            logger.info(
                                f"HITL decision received: {body['decision']} "
                                f"(review_id={request.review_id[:8]})"
                            )
                            return ReviewDecision(
                                decision=body.get("decision", "approve"),
                                modified_output=body.get("modified_output"),
                                reviewer_note=body.get("reviewer_note"),
                            )
                        # 200 but no decision yet — keep polling
                    elif poll_resp.status_code in (202, 404):
                        pass  # still pending
                    else:
                        logger.warning(f"HITL poll unexpected status: {poll_resp.status_code}")
                except Exception as e:
                    logger.warning(f"HITL poll error: {e}")

                # Reduce poll interval over time (back off slightly)
                interval = min(interval * 1.2, 15.0)

            logger.warning(
                f"HITL timed out after {self.config.timeout_seconds}s "
                f"(review_id={request.review_id[:8]}) — {self.config.on_timeout}"
            )
            return self._timeout_decision()

    def _timeout_decision(self) -> ReviewDecision:
        decision = self.config.on_timeout if self.config.on_timeout in ("approve", "reject") else "approve"
        return ReviewDecision(decision=decision)

    def _apply_decision(
        self,
        decision: ReviewDecision,
        original_output: str,
        warnings: List[str],
        review_id: str,
    ) -> HitlOutcome:
        warnings = list(warnings)
        timed_out = decision.decision in ("approve", "reject") and decision.reviewer_note is None

        if decision.reviewer_note:
            warnings.append(f"Reviewer note: {decision.reviewer_note}")

        if decision.decision == "reject":
            return HitlOutcome(
                triggered=True,
                decision="reject",
                output="",
                warnings=warnings + ["Result rejected by human reviewer"],
                timed_out=False,
                review_id=review_id,
            )

        if decision.decision == "modify" and decision.modified_output:
            warnings.append("Output modified by human reviewer")
            return HitlOutcome(
                triggered=True,
                decision="modify",
                output=decision.modified_output,
                warnings=warnings,
                timed_out=False,
                review_id=review_id,
            )

        # approve (or modify with no modified_output — treat as approve)
        if decision.decision.startswith("timeout_"):
            # came from our timeout path
            timed_out = True
            warnings.append(
                f"HITL: low-confidence result auto-{'approved' if 'approve' in decision.decision else 'rejected'} "
                f"after timeout (review_id={review_id[:8]})"
            )

        return HitlOutcome(
            triggered=True,
            decision="approve",
            output=original_output,
            warnings=warnings,
            timed_out=timed_out,
            review_id=review_id,
        )
