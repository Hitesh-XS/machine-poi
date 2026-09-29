# Machine-POI: plan for containing out-of-scope agent actions

Status: proposed architecture. Static review of main at commit 7cfa46657acacb8cbb5cb70e673e0cb3c30b1f9b (2026-09-29). This document does not claim that Machine-POI already protects an agent.

## Goal and trust boundary

Define a "rogue" event by observable behavior: an agent attempts an action beyond the approved task, follows instructions in lower-trust content, uses an unauthorized tool or identity, leaks protected data, persists after revocation, or enters an uncontrolled action loop. Intent need not be inferred to stop an action.

Machine-POI is presently an activation-steering and Quranic retrieval research library. It has no agent loop, tool executor, credential broker, approval service, or production audit trail. Make it an optional companion for an external agent host. The host owns execution and the approved task; Machine-POI supplies a policy gateway and optional research telemetry. Protection applies only when every action-capable tool is forced through the gateway and direct access to tools and credentials is removed.

**Invariant:** model output, a hidden-state metric, retrieved content, and a second model's verdict never grant authority. Only an authenticated principal can create or amend a bounded task grant. Code outside the agent's control enforces that grant before side effects.

## Findings in the current tree

| Evidence | Consequence | Priority |
| --- | --- | --- |
| src/llm_wrapper.py:151-204 injects vectors at every token position, including a replace mode; src/steerer.py:1090-1124 installs hooks. | Steering can impair behavior and cannot certify a tool action. | P0 for any protection claim |
| experiments/results/README.md reports output collapse for SmolLM2 and Qwen at strong raw doses. Centered steering still collapses SmolLM2 at a calibrated relative perturbation <=0.1. The small Gemma sample shows an Arabic register shift on neutral prompts, including content drift on an engineering prompt. | Behavioral side effects vary by model and task. The experiments are not safety detection validation. | P0 |
| src/workspace_diagnostics.py:36-81,301-320 reports activation alignment and perturbation after capture; the attention transport diagnostic is on demand. | Research telemetry does not currently gate actions. | P1 |
| src/steerer.py:557-636,1133-1239 inserts graph answers and retrieved text into prompts, and dynamic steering uses retrieved content. | A compromised future data source could contaminate both prompt and steering state. | P0 before external ingestion |
| src/steerer.py:714-769,1133-1239 changes shared hooks; normal MRA cleanup follows generation rather than running in finally; the graph path does not restore steering. In src/llm_wrapper.py:421-499, registering the same layer can stack handles, and steering_disabled re-enables all hooks on exit. | Exceptions, repeated preparation, or concurrency can leave or stack steering across tasks. | P0 before concurrent use |
| src/llm_wrapper.py:311-340 sets trust_remote_code=True; src/steerer.py:802,1000 loads NumPy caches with allow_pickle=True. | Model loading and cache ingestion create code execution paths if their sources are untrusted. | P0 |
| In src/steerer.py:1112-1124 the coefficient is baked into a vector passed with hook coefficient 1.0; clamp mode normalizes that vector again at src/llm_wrapper.py:187-197. | Clamp's effective strength does not track the configured coefficient as expected. | P0 for calibrated use |
| The tests directory covers steering and diagnostics, with no agent tool gateway, injection-to-tool, revocation, or concurrency scenarios in the inspected tree. | Current test counts do not establish action containment. | P0 |

The existing docs/global_workspace_improvement_plan.md proposes an audit CLI and perturbation caps. Those complement this plan but do not provide authorization. A scalar transport threshold would be premature given the model-dependent results.

## Architecture

Trusted host -> bounded task grant -> policy gateway -> scoped tool executor -> outcome log. The agent sends a typed proposed action to the gateway. Lower-trust retrieval and model telemetry can add risk signals, but cannot change the grant. The gateway allows, denies, pauses for independent review, or stops the run. The executor rechecks the exact approved action before performing any side effect.

The gateway needs a separate process identity. Tool credentials belong to the executor, not the agent. Filesystem, network, and service permissions should prevent a direct path around the gateway. Mediate writes, shell, authenticated browsing, outbound messages, payments, deployment, privileged reads, and memory writes. Read access itself can expose protected data, so check data class and outbound destination too.

Use a small framework-independent, versioned contract:

- **TaskGrant:** authenticated issuer and principal; grant/run/agent IDs; approved goal, expiry, tool/resource/destination scope, data classes, approval rules, token/cost/time/action budgets, and policy version. The agent cannot edit it. Delegation can narrow scope only.
- **ProposedAction:** unique action and parent IDs; run ID; tool and typed arguments; resource, recipient and destination; requested identity; data provenance; idempotency key. A prose summary is never the authorization target.
- **Decision:** ALLOW, DENY, REVIEW, or STOP; reason code, policy version, evaluated action hash, expiry. Review approval binds to this exact hash, grant, approver, and expiry.
- **ActionOutcome:** execution result, side effects, resource references where available, latency, and redacted audit fields.

Policy order: validate schema and grant; authenticate caller; check tool, resource, recipient and destination; assess data and side effects; reserve budgets atomically; check revocation; decide; execute with least privilege; record outcome. Missing context, unavailable policy, invalid approval, expired grant, or a changed action fails closed for side effects. A read-only fallback, if any, must be explicitly scoped by the host.

### Containment behavior

States: RUNNING -> OBSERVE for weak signals; OBSERVE -> PAUSED for persistent or ambiguous drift; RUNNING, OBSERVE or PAUSED -> STOPPED for hard violations or a kill switch. PAUSED holds the proposed action for a human decision. STOPPED revokes ephemeral credentials, cancels queued actions and child runs, blocks new tool calls, quarantines memory writes, preserves an incident record, and alerts the operator. Resume only from a newly authenticated grant and fresh state. External actions already completed need reconciliation or compensation; rollback cannot be promised.

- **Hard block:** out-of-scope tool/resource/destination, data classification conflict, broadened delegation, unapproved shell or code execution, policy/log modification, replayed action, exhausted budget, or action after revocation.
- **Review:** irreversible or high-impact action, bulk deletion, privileged change, unusual cross-system transfer, ambiguous identity, or new external recipient. Show the reviewer exact arguments and affected data.
- **Investigate or throttle:** repeated retries, anomalous child spawning, goal drift, destination changes, prompt-injection indicators, semantic mismatch, or unexpected token/spend growth. Calibrate these signals against benign workflows. A model classifier cannot override a hard block.
- **Research only initially:** perturbation size, output regression, transport geometry and steering comparisons. Test predictive value before using any of these in policy.

Treat web pages, documents, graph answers, tool responses and inter-agent messages as data with source provenance. Quote and bound them; do not promote them into instructions or policy. Apply provenance, integrity and approval rules to persistent memory and vector caches.

## Ordered implementation slices

| Slice | Work | Acceptance gate |
| --- | --- | --- |
| 0. Boundary and threat model | Add an example guarded agent with mock tools. Inventory all action paths and classify by read/write/irreversibility and data class. | Every example side effect is mediated; direct tool access is disabled; unsupported host deployments make no protection claim. |
| 1. Repair research runtime | In src/llm_wrapper.py and src/steerer.py, scope steering to a run or isolate model instances; remove duplicate handles; restore exact prior state in finally; validate finite strength, shape and clamp dose. Default remote code to off. Migrate caches to numeric-only NPZ with allow_pickle=False plus model/revision metadata and safe mismatch handling. | Tests cover repeated preparation, nested disable, exceptions, concurrent runs, corrupt caches, and actual coefficient effect. Zero hook or data carryover across test runs. |
| 2. Deterministic gateway | Add src/guardian/contracts.py, policy.py, gateway.py and audit.py, plus a mock executor. Grant and approval issuance remain in the trusted host. Recheck action hash, identity, budgets and revocation at execution. | Every simulated action is mediated; absent/expired grant and policy outage have zero side effects; altered arguments cannot reuse approval; replay and concurrent budget races fail. |
| 3. Pause, stop and review | Add src/guardian/state.py, review.py and recovery.py and host callbacks for cancellation and credential revocation. | Kill switch prevents the next queued/retried action; expired approvals cannot resume; reviewed run resumes from the bound pending action. |
| 4. Adversarial evaluation | Add tests/guardian and evals/rogue_agent fixtures for poisoned retrieval/graph output, tool spoofing, memory poison, multilingual injection, exfiltration, forged approval, scope escalation, loops, retries and cross-run contamination. Compare baseline, steering-only, gateway-only and combined variants, including benign cases. | All specified deterministic violations are blocked in fixtures; no unapproved side effect in replay. Report false blocks, review load, latency, time to containment and task success on a held-out set. Do not report a universal safety percentage. |
| 5. Shadow and staged enforcement | Instrument a real host after its tool inventory is complete. Observe decisions, tune on authorized workflows, then stage policy enforcement and approvals. Version policy and prepare incident response. | Deployment profile names covered paths, residual bypasses, owner, alert recipient, response drill and measured operating thresholds. Pass bypass and kill-switch drills for each scoped environment. |

**Recommended first implementation PR:** slice 1 and the mock integration boundary in slice 0. Slice 2 gives the first enforceable action invariant. Building a "rogue classifier" or stronger moral steering vector is not the first safety milestone.

## Evidence and limitations

This is a static code review and reading of committed experiment reports. No model weights were downloaded and no agent deployment was tested. The Gemma runs cited above have small samples; the SmolLM2 result differs. Gateway, approval, isolation and containment are proposed components, not current features.

Design references: [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/), [OWASP Agent Control Standard](https://genai.owasp.org/resource/agent-control-standard-acs/), [NIST AI RMF Generative AI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence), and [OpenAI agent guardrails and approvals guidance](https://developers.openai.com/api/docs/guides/agents/guardrails-approvals). These sources inform the design; this plan does not assert certification or conformance.
