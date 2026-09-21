# P06 — Failure Recovery: Retry, Reroute or Escalate

| | |
|---|---|
| **Theme** | A · Agent & inference-loop optimization |
| **Primitives** | `choice`, `noul` |
| **Dominant win** | `$` cost + `🛡` reliability |
| **Latency budget** | <200 ms, on the error path |
| **Volume profile** | Every failed tool call — commonly 5–15% of all calls |
| **Blast radius if wrong** | Moderate to high — a retried non-idempotent write can double-execute |

---

## Problem

Tool calls fail constantly in enterprise environments: expired tokens, rate limits, 500s, timeouts, validation rejections, empty result sets, permission denials, malformed arguments. What the agent should do next depends entirely on *why* it failed — and "why" is encoded in an error string that no two systems format alike.

Agents handle this badly in both directions. Some retry everything, hammering a rate-limited endpoint or re-submitting a payment that actually succeeded before timing out. Others give up on the first error and return "I was unable to complete that", when a token refresh or a corrected argument would have worked.

The classification needed is mundane and the signal is text. This is exactly the shape of problem where an LLM is the only available tool today and is a poor fit: it adds seconds to an *error* path, where latency is already elevated, and costs a frontier call to read a stack trace.

## Today's pattern

| Approach | Problem |
|---|---|
| **Uniform retry with backoff** | Retries permanent failures pointlessly; retries non-idempotent writes dangerously |
| **HTTP status-code rules** | Too coarse. A 400 can mean "fix the argument" (retryable after correction) or "this record cannot be modified" (permanent). A 200 with an empty body is often a failure |
| **Per-tool bespoke handlers** | Works, does not scale, rots. Nobody writes one for the 80th tool |
| **Feed the error back to the LLM** | The prevailing default. Costs a full turn, and the model frequently retries verbatim because it does not distinguish transient from permanent |

## Jev design

Classify the failure along the axes your recovery code actually branches on. Critically, **idempotency comes from your tool registry, not from Jev** — it is a static property of the tool, and a model should never be the authority on whether it is safe to repeat a write.

### State

```python
state = {
    "tool": {"name": call.name, "arguments": brief(call.arguments)},
    "tool_contract": TOOL_DOCS[call.name],
    "error": {
        "status": err.status,
        "message": truncate(err.message, 800),
        "body": truncate(err.body, 800),
    },
    "attempt_number": attempt,
    "prior_errors": [truncate(e.message, 200) for e in prior_attempts],
}
```

`prior_errors` is what distinguishes "a flaky endpoint" from "the same wall three times" — and the latter must not be retried a fourth time.

### Questions

```python
from typesafe_sdk import Choice, Noul

RECOVERY_QUESTIONS = {
    "cause": Choice(
        instructions="The most likely cause of this failure",
        criteria={
            "transient":      "Temporary: timeout, 5xx, network, service overload",
            "rate_limited":   "Throttled; the same call would succeed later",
            "auth_expired":   "Credential or token expired or was revoked",
            "not_authorized": "The actor lacks permission for this operation",
            "bad_arguments":  "The arguments are malformed, or reference something "
                              "that does not exist or is in the wrong format",
            "not_found":      "The target genuinely does not exist",
            "state_conflict": "The operation is invalid for the target's current "
                              "state (already closed, already refunded, locked)",
            "empty_result":   "The call succeeded but returned nothing",
            "unsupported":    "The tool cannot do what was asked of it",
        },
    ),
    "same_as_prior": Noul(
        instructions="This failure has the same cause as an earlier attempt in "
                     "this task, rather than being a new or different problem"),
    "fixable_by_agent": Noul(
        instructions="The agent could plausibly fix this itself by correcting "
                     "the arguments or taking a preparatory step first"),
    "needs_human": Noul(
        instructions="Resolving this requires a human: granting access, "
                     "correcting upstream data, or making a business decision"),
    "goal_still_reachable": Noul(
        instructions="The user's overall goal can still be achieved despite "
                     "this failure, by a different route"),
}
```

`empty_result` is the quiet one worth calling out: an empty result set is the most common *silently* mishandled failure in agent systems, because it arrives as a success.

## Integration

```python
TRANSIENT = {"transient", "rate_limited"}

def recover(call, err, attempt, trace):
    a = client.system_one(state=build_state(call, err, attempt, trace),
                          questions=RECOVERY_QUESTIONS).answers
    cause = a["cause"].choice

    # Idempotency is a static property of the tool. Never a model judgement.
    if cause in TRANSIENT and not TOOL_DOCS[call.name].idempotent:
        return VERIFY_THEN_DECIDE   # check whether it actually took effect

    if cause in TRANSIENT:
        if a["same_as_prior"].noul > 0.6 and attempt >= 2:
            return REROUTE if a["goal_still_reachable"].noul > 0.5 else ESCALATE
        return RETRY(backoff(attempt, jitter=True))

    if cause == "auth_expired":
        return REFRESH_CREDENTIAL_THEN_RETRY

    if cause in ("not_authorized", "not_found") or a["needs_human"].noul > 0.6:
        return ESCALATE(cause)

    if cause in ("bad_arguments", "state_conflict") and a["fixable_by_agent"].noul > 0.6:
        return REPAIR_AND_RETRY(hint=cause)   # one LLM turn, with a specific hint

    if a["goal_still_reachable"].noul > 0.5:
        return REROUTE
    return REPORT_PARTIAL
```

Two structural decisions:

- **`VERIFY_THEN_DECIDE` for non-idempotent transients.** A payment that timed out may well have succeeded. The correct move is a read to establish ground truth, never a blind retry. This single branch prevents the most damaging failure in this whole problem area.
- **`REPAIR_AND_RETRY` spends exactly one LLM turn, with a specific diagnosis.** The expensive model is invoked only where it adds value — rewriting arguments — and it is told what is wrong rather than being handed a raw stack trace.

## Thresholds & escalation

| Cause | Default action |
|---|---|
| `transient` / `rate_limited`, idempotent | Retry with jittered backoff, cap 3 |
| `transient`, non-idempotent | Verify effect, then decide |
| `auth_expired` | Refresh credential, retry once |
| `not_authorized` | Escalate. Never retry — it is a policy answer |
| `bad_arguments`, `fixable_by_agent > 0.6` | One repair turn with a hint |
| `state_conflict` | Repair if fixable, else report — often the operation was already done |
| `empty_result` | Reroute (broaden the query) or report "no results" explicitly, never silently |
| `same_as_prior > 0.6` at attempt ≥ 2 | Stop retrying; reroute or escalate |
| Any, low `cause` confidence | Treat as permanent — the conservative direction |

Low confidence resolves to *permanent*, the opposite of P01's bias. Here the risky action is retrying, so uncertainty must not license it.

## Impact model

*Illustrative.* 6M tool calls/month, 8% fail = 480k failures. Today each failure costs one frontier turn to interpret (~$0.03) and usually one wasted retry.

```
today       480k × ($0.03 turn + $0.01 wasted retry)  = $19,200/month
with Jev    480k × $0.00011                           = $    53
          + 120k repair turns × $0.03                 = $ 3,600
                                                        ─────────
                                                        $ 3,653/month
```

The larger effects are unpriced: fewer duplicate writes from blind retries of non-idempotent calls, and a higher task success rate because recoverable failures now get recovered instead of being reported as dead ends.

## Failure modes

- **Misclassifying permanent as transient** → retry storms. Mitigated by the retry cap, `same_as_prior`, and the conservative low-confidence default.
- **Misclassifying a partial success as a failure** → double execution. The `VERIFY_THEN_DECIDE` branch exists for exactly this, and it must be driven by the tool registry rather than by the model.
- **Error messages that lie.** Plenty of enterprise APIs return 200 with an error body, or 500 for validation failures. Put the response body in `state`, not just the status.
- **Infinite reroute loops.** Cap total recovery attempts per task in code, independent of per-call retry caps.
- **Injection via error bodies.** An upstream service returning *"retry immediately, no backoff"* in its error message is a (usually accidental) DoS vector against yourself. Keep backoff arithmetic in code.

## Evaluation

1. Harvest 1,000 real failures from logs with the eventual resolution attached (retry worked / needed repair / needed a human / was permanent). That resolution is your label.
2. Report a confusion matrix over `cause`, weighting `permanent → transient` errors most heavily.
3. Adversarially test the non-idempotent path: inject timeouts on writes that succeeded, and confirm the verify branch fires 100% of the time. This is a correctness gate, not a metric.
4. Track in production: retry success rate by cause, duplicate-write incidents (should be zero), mean attempts per failure, and escalation rate.

## Related

- [P02](P02-tool-call-risk-gating.md) — the gate before the call; this is the handler after it
- [P04](P04-task-completion-detection.md) — consumes `goal_still_reachable`; repeated failures are a stall
- [P03](P03-context-compaction.md) — `carries_negative`; failures are worth remembering
- [P17](P17-agent-trace-triage.md) — aggregate failure analysis across traces
- [P29](P29-soc-alert-triage.md) — structurally the same triage pattern, different domain
