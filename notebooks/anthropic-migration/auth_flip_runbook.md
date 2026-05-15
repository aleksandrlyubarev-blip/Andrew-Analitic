# Auth flip runbook (Phase 1, 15.06.2026)

> **Purpose.** Make the 15.06 auth flip executable as a sequence of pre-decided
> steps, not improvisation. Every "we'll figure it out on the day" item below
> is moved into the **Sandbox validation** section so it can be settled by
> 14.06 against a throwaway account.

## 0. Goal & assumptions

| Item | Value |
| ---- | ----- |
| Auth flip window | 15.06.2026, 1 working day (target: complete by 18:00 local) |
| Code freeze for demo | 29.06–01.07 (no flips, no Claude-side experiments) |
| Repos in scope | `Andrew-Analitic`, `Romeo_PHD` (= PinoCut) |
| Branch | `claude/anthropic-sdk-migration-mPKYD` (already pushed) |
| Rollback toggle | `ANTHROPIC_AUTH_MODE=api_key` (default) + `ANTHROPIC_API_KEY` re-set in env |
| Rollback target time | < 5 min from decision to rolled-back prod |

Assumptions that **must** be true at T-0 (validate on 14.06):

1. `claude login` works under the Max 5x account on the prod host(s) where
   Andrew Swarm and PinoCut API server run.
2. The cached OAuth credentials produced by `claude login` either
   (a) populate `ANTHROPIC_AUTH_TOKEN` env via a wrapper that LiteLLM and
   `@anthropic-ai/sdk` will accept, **or**
   (b) the Anthropic-published path for plan-auth uses a different runtime
   (Agent SDK / `claude -p`) that we wrap explicitly. Decide which by 14.06.
3. The $100 SDK credit was claimed (ТЗ 1.1) in the Anthropic UI.

If any of these is false at T-0, **do not start the flip** — see "Abort &
defer" at the end.

## 1. Pre-flight (08.06–14.06)

### 1.1 Sandbox validation (the critical week)

These five experiments must be run against a **throwaway** Anthropic account
(or a separate billing project) with a tiny budget. The results decide the
route for the real flip. Document outcomes in this file as you go.

- [ ] **S1: Does `claude login` populate something LiteLLM can consume?**
  - Run `claude login` under the sandbox account.
  - Inspect `~/.claude/.credentials.json`. Note the field name of the bearer
    token (likely `access_token` or `accessToken`).
  - Try: `ANTHROPIC_AUTH_TOKEN=<that bearer> python -c "from litellm import completion; print(completion(model='anthropic/claude-sonnet-4-20250514', messages=[{'role':'user','content':'ping'}]).choices[0].message.content)"`
  - **Pass criterion:** call returns content, no 401. If it 401s, plan-auth
    OAuth bearers are not accepted on the standard Anthropic API endpoint
    and we must use the Agent SDK path instead.
  - **Record result here:** `__________`

- [ ] **S2: Does `@anthropic-ai/sdk` accept the same bearer?**
  - In `lib/integrations-anthropic-ai/src/client.ts`, temporarily swap
    `apiKey: process.env.AI_INTEGRATIONS_ANTHROPIC_API_KEY` for
    `authToken: process.env.ANTHROPIC_AUTH_TOKEN` (the SDK accepts both since
    v0.40+).
  - Run a one-off TS smoke script (see § 4.2). If 200 OK → S2 passes.
  - **Record result here:** `__________`

- [ ] **S3: Does plan-auth survive process restart and 24h?**
  - After S1 succeeds, leave the sandbox running. Kick a call every hour for
    24h via cron. Look for token expiry / refresh behavior.
  - **Pass criterion:** no auth failures, OR a documented refresh command
    that can be scheduled (TZ R2).
  - **Record result here:** `__________`

- [ ] **S4: Does `claude credits status` exist on the installed CLI version?**
  - Run `claude credits status` (or whatever Anthropic ships ~15.06).
  - Capture the raw output to `notebooks/anthropic-migration/sample_credits_status.txt`.
  - **Record format:** plain text / JSON / other: `__________`
  - This unblocks ТЗ Phase 3.1 (credit_monitor parser).

- [ ] **S5: Does the GHA action `anthropics/claude-code-action@beta` accept
      plan auth, or does it still require `ANTHROPIC_API_KEY` as a secret?**
  - Run a manual `workflow_dispatch` on the sandbox repo's
    `claude-pr-review.yml` (Phase 0.3 committed it disabled).
  - **Pass criterion:** action completes with cached creds OR with an
    explicit OAuth token flow documented in Anthropic's email (~08.06).
  - **Record result here:** `__________`

### 1.2 Outcome → routing decision

After S1–S5, pick **exactly one** of these three paths and write it down here
before 14.06 EOD:

- [ ] **Path A — env-var bridge.** S1+S2 passed. `ANTHROPIC_AUTH_TOKEN` carries
      the bearer; LiteLLM and `@anthropic-ai/sdk` use it transparently. Only
      change in repos: a wrapper that copies the bearer from
      `~/.claude/.credentials.json` into env at process start.
- [ ] **Path B — Agent SDK adapter.** S1 failed but S5 passed or Anthropic
      provided an Agent SDK Python/Node package that talks to plan-auth.
      Build a tiny adapter that the `ANTHROPIC_AUTH_MODE=sdk` branch calls,
      bypassing LiteLLM for Anthropic-bound traffic.
- [ ] **Path C — subprocess via `claude -p`.** Both above failed. Wrap each
      Anthropic call in a `claude -p` subprocess. Acceptable only as a
      fallback — no streaming, awkward tool-use, slow.

**Chosen path:** `_____`
**Reasoning:** `__________________________________________________`

### 1.3 14.06 final pre-flight

- [ ] `pytest tests/ -q` on Andrew-Analitic main → `389 passed, 2 skipped`
      (current baseline; update this line if the count drifts before 15.06).
- [ ] `pnpm run typecheck` on Romeo_PHD main → no errors.
- [ ] M5 (claude.ai subscription utilization snapshot) captured into
      `baseline_2026-05.md`. **This is the last clean pre-migration metric.**
- [ ] Anthropic Console: confirm $5 spending limit configured (ТЗ 1.8 safety
      net). Do **not** delete the API key yet — rollback needs it.
- [ ] Telegram bot / email alert channel for credit-monitor verified live
      (test ping). Even though monitor lands Phase 3, the channel is shared.
- [ ] One operator on-call from 09:00 to 20:00 on 15.06. Name: `__________`
- [ ] Backup operator: `__________`

## 2. Flip — 15.06 hour-by-hour

All times local. Assumes Path A or B chosen (Path C requires a different
runbook — write it during sandbox week if C looks likely).

### T+0 (09:00) — claim & local validation

1. Operator logs into Anthropic UI under Max 5x account.
2. Claim the $100 SDK credit (ТЗ 1.1). Screenshot the dashboard showing
   `Remaining: $100.00` into `notebooks/anthropic-migration/claim_screenshot.png`.
3. On the prod host(s), run `claude login`. Confirm
   `cat ~/.claude/.credentials.json` shows the bearer.
4. Set up the env bridge (Path A) or the adapter (Path B) — see § 3.

### T+1h (10:00) — Andrew-Analitic smoke

1. On a non-prod replica or a `git checkout` of the migration branch:
   ```bash
   git fetch origin claude/anthropic-sdk-migration-mPKYD
   git checkout claude/anthropic-sdk-migration-mPKYD
   git pull
   ```
2. Edit env: `ANTHROPIC_AUTH_MODE=sdk`, `ANTHROPIC_API_KEY=` (empty),
   `ANTHROPIC_AUTH_TOKEN=<bearer>` (Path A) or leave alone (Path B).
3. Run `pytest tests/ -q`. **Expect:** `389 passed, 2 skipped`. If the
   `test_anthropic_auth` suite fails, the guard caught a missing rewire
   (Phase 1.3 deliverable below isn't merged).
4. Run the live smoke script in § 4.1. Expect 3 lanes × 3 calls = 9 successful
   200s with content. Cost should appear on the SDK credit, **not** on the
   Anthropic Console API meter.
5. Eyeball the Anthropic Console "API usage" page after the smoke: it must
   stay flat (or only show the $5 limit being unused). The SDK dashboard
   should tick down by a few cents.

### T+3h (12:00) — Romeo_PHD smoke

1. Same branch checkout in Romeo_PHD repo.
2. Env: same overrides as above, plus existing
   `AI_INTEGRATIONS_ANTHROPIC_BASE_URL` left intact.
3. `pnpm install --frozen-lockfile && pnpm run typecheck` → no errors.
4. Run the live smoke script in § 4.2 (creates one synthetic pipeline node,
   calls `executeNodeWithLLM`, asserts 200 and parses JSON).
5. Confirm SDK credit ticks down; Console API meter does not.

### T+5h (14:00) — production merge & restart

If both smokes are green:

1. Merge `claude/anthropic-sdk-migration-mPKYD` → `main` in both repos.
   (Phase 1 deliverable in TZ 1.6 / 1.7.)
2. Update prod env vars (set `ANTHROPIC_AUTH_MODE=sdk`, clear the API key)
   on whichever host runs Andrew Swarm bridge and PinoCut api-server.
3. Restart services. Capture restart timestamps for the 24h watch window.
4. In Anthropic Console: lower API spend limit from $50 to $5 (ТЗ 1.8). **Do
   not delete the key yet** — it's our rollback handle.

### T+6h–T+24h — watch window

- Telegram alerts on for: credit burn > $4/24h, any 4xx auth error from
  Anthropic, service-level 5xx spikes.
- No code merges to either repo unrelated to the migration during this
  window. Quiet hours.

### T+24h (16.06, 09:00) — acceptance check

ТЗ AC 1.9 deliverable. Operator confirms:

- [ ] No regression vs M3 (379+ tests still passing on `main`).
- [ ] No regression on M4 (latency p50/p95 within ±15% of baseline).
- [ ] Anthropic Console API meter burn for last 24h = $0 on new traffic
      (residual M1 baseline tail is acceptable).
- [ ] SDK credit burn rate is in the expected envelope (target $3.30/day,
      acceptable corridor $2.50–$5.00 for first 7 days).

If all four green → Phase 1 is done; proceed to Phase 2 (16.06).

## 3. Wiring map — what the flip actually changes

Concrete files touched on flip day. Phase 0 already added the toggle; flip
day wires it up.

### 3.1 Andrew-Analitic

All Anthropic-bound `litellm.completion()` calls live in **5 call sites**
that already import `litellm.completion` directly:

| File                | Line | Caller |
| ------------------- | ---- | ------ |
| `lcb/runner.py`     | ~302 | LCB single-call runner |
| `lcb/pipeline.py`   | ~106 | LCB pipeline step |
| `core/romeo_swarm.py` | ~218 | Romeo educator agent |
| `core/romeo_phd.py` | ~118 | Romeo PhD agent |
| `core/memory.py`    | ~243 | Consolidation engine |

**Path A wiring (preferred).** No file edits on the day — only env. Add a
one-shot init script `scripts/load_plan_auth.sh` that:
1. Reads bearer from `~/.claude/.credentials.json` (`jq -r '.access_token'`).
2. `export ANTHROPIC_AUTH_TOKEN="$bearer"`
3. `unset ANTHROPIC_API_KEY`
Source it from the service unit (systemd `EnvironmentFile=`).
**Then** the 5 call sites need no change at all — LiteLLM will see
`ANTHROPIC_AUTH_TOKEN` and route via plan auth.

**Path B wiring.** Add `core/anthropic_client.py` with:
```python
def claude_complete(model, messages, **params): ...
```
that delegates to the Agent SDK when `is_sdk_mode()` else to `litellm.completion`.
Each of the 5 call sites changes from
```python
from litellm import completion
response = completion(model=model, messages=msgs, **params)
```
to
```python
from core.anthropic_client import claude_complete
response = claude_complete(model=model, messages=msgs, **params)
```
when (and only when) `model` starts with `anthropic/`. Non-Anthropic lanes
(Grok, OpenAI) still go through `litellm.completion`. This means each call
site needs a 4-line conditional. Total diff: ~30 LOC. Doable on the day but
**must be implemented and merged on 13.06**, not improvised.

### 3.2 Romeo_PHD

A single active call site lives in
`artifacts/api-server/src/lib/pipeline-executor.ts:63`, going through
`lib/integrations-anthropic-ai/src/client.ts`. The batch utility template
in `lib/integrations-anthropic-ai/src/batch/utils.ts` is documentation
only — no live wiring.

**Path A wiring.** In `client.ts`, alongside the existing `apiKey`, also
pass `authToken: process.env.ANTHROPIC_AUTH_TOKEN`. SDK uses whichever is
truthy. Remove the `requireApiKeyMode` guard (Phase 0 placeholder).

**Path B wiring.** Replace the `Anthropic` client constructor with an
adapter that branches on `isSdkMode()` — but the Node Agent SDK package
may not exist yet (validation S2/S5). If it doesn't, fall back to keeping
PinoCut on API keys for one more cycle and document the gap in the
post-mortem (ТЗ 5.1).

## 4. Smoke scripts

Both scripts are checked into the migration branch as part of this runbook
so the operator just runs them.

### 4.1 Andrew-Analitic — `scripts/auth_flip_smoke.py`

> To be added on 13.06 alongside Path A/B wiring. Skeleton:

```python
"""Smoke test for the 15.06 auth flip. Runs 3 lanes × 3 calls = 9 requests."""
import os, time
from litellm import completion

assert os.getenv("ANTHROPIC_AUTH_MODE") == "sdk", "Set ANTHROPIC_AUTH_MODE=sdk first"
assert not os.getenv("ANTHROPIC_API_KEY"), "Clear ANTHROPIC_API_KEY first"
assert os.getenv("ANTHROPIC_AUTH_TOKEN"), "ANTHROPIC_AUTH_TOKEN missing (Path A)"

LANES = {
    "standard": "anthropic/claude-sonnet-4-20250514",
    "analytics": "gpt-4o-mini",        # control: should keep using OpenAI key
    "reasoning": "grok-4",             # control: should keep using Grok key
}

for lane, model in LANES.items():
    for i in range(3):
        t = time.time()
        r = completion(model=model, messages=[{"role":"user","content":f"smoke {lane} {i}: say OK"}], temperature=0)
        latency_ms = int((time.time() - t) * 1000)
        ok = "OK" in (r.choices[0].message.content or "")
        print(f"[{lane}] call {i}: {'pass' if ok else 'FAIL'} ({latency_ms}ms)")
```

Expected: 9 lines all `pass`.

### 4.2 Romeo_PHD — `scripts/src/auth-flip-smoke.ts`

> To be added on 13.06. Skeleton:

```typescript
import { anthropic } from "@workspace/integrations-anthropic-ai";

if (process.env.ANTHROPIC_AUTH_MODE !== "sdk") throw new Error("set ANTHROPIC_AUTH_MODE=sdk");

const t0 = Date.now();
const msg = await anthropic.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 64,
  messages: [{ role: "user", content: "smoke test: say OK" }],
});
const block = msg.content[0];
const text = block.type === "text" ? block.text : "";
console.log(`Romeo_PHD smoke: ${text.includes("OK") ? "pass" : "FAIL"} (${Date.now() - t0}ms)`);
```

Run with `pnpm --filter scripts exec tsx src/auth-flip-smoke.ts`.

## 5. Rollback

Triggers — roll back **immediately** if any of these is true after the flip:

- More than 2 consecutive auth failures (401/403) from Anthropic.
- Test suite regression on `main` (drop below 389 passed).
- p95 latency for any lane goes above 1.5× baseline for >15 min.
- Operator gut-call: anything looks weird that we can't explain in 10 min.

Procedure (target: 5 min from decision):

1. On prod host(s):
   ```bash
   export ANTHROPIC_AUTH_MODE=api_key
   export ANTHROPIC_API_KEY=<the key we kept for exactly this moment>
   unset ANTHROPIC_AUTH_TOKEN
   systemctl restart andrew-bridge pinocut-api  # or whatever the unit names are
   ```
2. Telegram-announce the rollback.
3. In Anthropic Console: raise spend limit back to $50 for 72h (do not lift
   to $200 — we want any explosion to be capped).
4. Open a post-mortem doc under `notebooks/anthropic-migration/`. The
   operator on call writes it, not the next-day team.

The migration branch on `main` is **not** reverted — only the env is
flipped. This keeps Phase 2/3 work merge-able once the root cause is found.

## 6. Abort & defer

Conditions to **not start** the flip on 15.06:

- Sandbox week (S1–S5) left any of S1, S2, S5 unresolved.
- Anthropic's expected email (~08.06) hasn't arrived or contradicts our
  assumptions.
- A demo-critical PR for 01.07 is in-flight in either repo and the flip
  would block it.

If aborted, the new target is 22.06 (one-week slip), which still leaves a
1-week soak before the 29.06 freeze. If aborted again on 22.06, demo runs
on API keys and the migration shifts to post-demo (02.07).

## 7. Open questions to settle during sandbox week

| # | Question | Owner | Resolved by |
| - | -------- | ----- | ----------- |
| Q1 | LiteLLM 1.84.0 honours `ANTHROPIC_AUTH_TOKEN` — does the standard Anthropic API endpoint accept an OAuth bearer from `claude login`? | sandbox S1 | 12.06 |
| Q2 | Does `@anthropic-ai/sdk` v0.78 accept `authToken` constructor param, and does prod endpoint accept it? | sandbox S2 | 12.06 |
| Q3 | Token refresh: does the bearer expire? If yes, what's the refresh command? | sandbox S3 | 13.06 |
| Q4 | `claude credits status` output format (JSON / plain / non-existent)? | sandbox S4 | 13.06 |
| Q5 | GHA `anthropics/claude-code-action@beta` and plan auth: does it need a special CI runner, or self-hosted only? | sandbox S5 | 13.06 |
| Q6 | If Path B is chosen — does Anthropic ship a Node Agent SDK package (`@anthropic-ai/claude-agent-sdk` or similar) by 15.06? | Anthropic email | 08.06 |

Every "yes/no" lands in this file with the date. **No verbal-only answers.**

## 8. Sign-off

- [ ] Runbook reviewed by: `__________` on `__________`
- [ ] Path A/B/C chosen (record in § 1.2)
- [ ] Sandbox checklist S1–S5 complete and recorded
- [ ] Phase 1 acceptance (TZ 1.9) signed off on `__________` (T+24h)
