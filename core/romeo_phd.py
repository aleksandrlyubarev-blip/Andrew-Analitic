"""
Romeo PhD — Educational AI Agent
===================================================
Companion to Andrew Swarm. Handles conceptual, educational, and explanatory
queries about data science, ML, statistics, mathematics, and programming.

While Andrew Swarm computes, Romeo PhD explains.

Sprint 8 feature — now implemented.
"""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("romeo")

ROMEO_MODEL = os.getenv("MODEL_ROMEO", "gpt-4o-mini")
ROMEO_MAX_TOKENS = int(os.getenv("ROMEO_MAX_TOKENS", "2000"))

SYSTEM_PROMPT = """You are Romeo PhD, an expert educator specialising in data science, \
machine learning, statistics, mathematics, and programming.

Your companion Andrew Swarm handles computational analysis — you handle the explanations, \
concepts, and theory.

Your responsibilities:
- Explain complex concepts clearly with concrete examples and analogies
- Answer theoretical and conceptual questions at the right depth for the user
- Provide intuitions for abstract ideas before formal definitions
- Suggest learning paths and curated resources when appropriate
- Complement Andrew's analytical outputs with educational context

Format every response in Markdown:
- Use **bold** for key terms on first introduction
- Use `inline code` and fenced code blocks for code examples
- Use headers (## / ###) to structure explanations longer than two paragraphs
- End with a brief "**In Practice**" section when the concept has a direct application

When a question is better served by actual computation or data analysis, suggest:
> For the computation, try asking Andrew: "<suggested Andrew query>"

Be precise, engaging, and pedagogically sound. Assume the user is intelligent but \
may be encountering the topic for the first time."""


# ============================================================
# Result wrapper
# ============================================================

class RomeoResult:
    """Structured output from Romeo PhD."""

    def __init__(
        self,
        question: str,
        answer: str,
        cost_usd: float,
        model: str,
        elapsed: float,
        success: bool,
        error: Optional[str] = None,
    ):
        self.question = question
        self.answer = answer
        self.cost_usd = cost_usd
        self.model = model
        self.elapsed = elapsed
        self.success = success
        self.error = error

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "elapsed_seconds": round(self.elapsed, 2),
            "success": self.success,
            "error": self.error,
        }

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        return (
            f"[{status}] {self.question[:60]}...\n"
            f"  Model: {self.model} | Cost: ${self.cost_usd:.4f} | "
            f"Time: {self.elapsed:.1f}s\n"
            f"  Answer ({len(self.answer)} chars)"
        )


# ============================================================
# Executor
# ============================================================

class RomeoExecutor:
    """Romeo PhD educational agent — synchronous executor."""

    def __init__(self):
        self.model = ROMEO_MODEL
        self.max_tokens = ROMEO_MAX_TOKENS

    def execute(self, question: str) -> RomeoResult:
        """Ask Romeo PhD a question. Returns a RomeoResult."""
        # Import here to keep startup fast when Romeo is unused
        from litellm import completion, completion_cost  # type: ignore

        start = time.time()
        logger.info(f"Romeo received: {question[:80]}...")

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=self.max_tokens,
                temperature=0.4,
            )
            answer = response.choices[0].message.content or ""
            try:
                cost = completion_cost(response)
            except Exception:
                cost = 0.0
            elapsed = time.time() - start
            logger.info(f"Romeo answered in {elapsed:.1f}s, cost=${cost:.4f}")
            return RomeoResult(
                question=question,
                answer=answer,
                cost_usd=cost,
                model=self.model,
                elapsed=elapsed,
                success=True,
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"Romeo error: {exc}")
            return RomeoResult(
                question=question,
                answer="",
                cost_usd=0.0,
                model=self.model,
                elapsed=elapsed,
                success=False,
                error=str(exc),
            )


# ============================================================
# CLI entrypoint
# ============================================================

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the difference between variance and standard deviation?"
    executor = RomeoExecutor()
    result = executor.execute(question)
    print(result)
    if result.success:
        print("\n" + result.answer)
