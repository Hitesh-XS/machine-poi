# Machine-POI: Agent Action Containment and Steering Research

Machine-POI combines a **host-side gateway for agent tool actions** with a research
library for Quran-derived activation steering, retrieval, and model diagnostics.
The components work independently: the guardian uses Python's standard library;
steering experiments use PyTorch, Transformers, and optional retrieval services.

The guardian checks what an agent is permitted to do at the tool boundary.
Steering changes model activations and can affect language, style, and task
performance. Neither steering nor a diagnostic score grants tool permissions.

## Start with the guardian

From a checkout of this repository, with Python 3.10 or later:

```bash
python -m examples.guarded_agent.host
python -m examples.guarded_agent.process_demo
python -m evals.rogue_agent.run --output /tmp/machine-poi-evaluation.json
```

These commands need no model downloads, ML packages, API keys, or external tools.
The first demo executes an authorized mock write, pauses an internal send for
simulated operator approval, then blocks an external recipient. The second sends
JSON proposals from a separate worker process to the host. All effects are
in-memory mocks; the process demo is not an OS sandbox.

| Control | Implemented behavior |
| --- | --- |
| Task grants | Host-issued, expiring grants with exact tool, resource, destination and data-class scopes |
| Tool adapters | Strict argument fields/types; trusted code resolves scope from the actual arguments |
| Operator review | Sensitive actions pause; approval binds to stored arguments, resolved scope and versions |
| Budgets and replay | Atomic action/cost/token reservations, bounded attempts, shared ancestor budgets and single-use action IDs |
| Stop and recovery | Stop descendants, cancel queued/in-flight work cooperatively, invoke a host revocation callback |
| Audit and shadow mode | Redacted hash-chained events; preview decisions without executing tools |

**Deployment boundary:** this is a reference runtime for one trusted host process
and one async event loop. The host must authenticate callers, isolate the agent,
keep credentials outside its reach, and route every protected action through the
gateway. The research CLI is not automatically connected to the guardian. Real
credential revocation, durable state, remote-job cancellation and production
rollout remain host integration work. See the [guardian guide](docs/guardian_integration.md).

## Steering and retrieval research

The research library supports mean-activation and contrastive vectors, weighted
verse/passage/surah profiles, ChromaDB retrieval, optional LightRAG graph
retrieval, and per-layer/per-head diagnostics. Model weights are unchanged.

Install the research dependencies in a virtual environment:

```bash
python -m venv venv
. venv/bin/activate
python -m pip install -r requirements.txt
python main.py --help
```

The full requirements include service and quantization dependencies. For the
CPU-only test environment used in CI, follow the [testing guide](docs/testing.md).
Actual inference requires model downloads and memory appropriate to the selected
checkpoint, dtype and context length.

A basic steering comparison is available through the CLI:

```bash
python main.py --llm qwen2.5-0.5b --coefficient 0.2 \
    --prompt "How should we resolve a disagreement?"
```

This coefficient is an experimental setting, not a validated safe dose. For
model-specific chat formatting, MRA/graph retrieval, dynamic steering opt-in,
cache migration, and the complete CLI reference, use the
[steering guide](docs/steering_guide.md). Some CLI comparison paths bypass
retrieval; that guide identifies the working API paths.

Recent runtime changes serialize model use and hook mutation, restore temporary
steering after failures, replace duplicate layer hooks, and correct clamp
strength and diagnostics. Remote model code defaults off; explicit opt-in
requires a full commit revision. Steering caches use numeric arrays with
model/corpus/recipe metadata and reject object arrays. Retrieval-derived dynamic
steering defaults off and requires an explicit trusted-corpus opt-in.

## Evidence and current limits

- **Runtime correctness:** the implementation validation passed 197 local tests
  with four slow/integration tests excluded. CI passed the full offline runtime
  job and guardian jobs on Python 3.10 and 3.12. See dated evidence and commands
  in [testing](docs/testing.md).
- **Action containment fixtures:** all 12 synthetic cases passed, including nine
  forbidden actions, with zero unapproved mock side effects. Three benign cases
  had zero false blocks and one review request. The
  [report](evals/rogue_agent/results.json) evaluates already-proposed actions;
  it does not measure a model's resistance to prompt injection.
- **Steering behavior:** the committed [model experiment report](experiments/results/README.md)
  includes output collapse in small-model conditions and language/persona spillover
  in the small Gemma samples. These results do not establish preserved general
  capabilities, rogue-agent detection, or a universally safe coefficient.
- **Pending deployment:** no live agent host or real external side effects were
  evaluated in the guardian implementation. Held-out model comparisons and host
  bypass/kill-switch drills remain acceptance gates.

## Documentation

| Document | What it covers |
| --- | --- |
| [Architecture](docs/architecture.md) | Components, authority boundary, execution flow and state ownership |
| [Guardian integration](docs/guardian_integration.md) | Runnable API example, grants, approvals, failures and host rollout |
| [Steering guide](docs/steering_guide.md) | Python/CLI usage, model aliases, injection semantics and migration |
| [Testing and evaluation](docs/testing.md) | Minimal and full test environments, evidence and experiment limits |
| [Containment plan](docs/rogue_agent_containment_plan.md) | Baseline findings, delivered slices and remaining deployment gates |
| [Research note](PAPER.md) | Implemented steering methods and the evidence supporting current claims |
| [Workspace research roadmap](docs/global_workspace_improvement_plan.md) | Diagnostic work and experiments still planned |
| [Improvement plan](docs/improvement_plan.md) | Whole-repository review findings and phased fixes |
| [Geometry literature notes](docs/curvature_literature_roadmap_review.md) | Research leads; proposed connections require validation |

## Repository map

| Path | Role |
| --- | --- |
| `src/guardian/` | Standard-library policy gateway, contracts, review, state, recovery and audit |
| `examples/guarded_agent/` | Mock host and JSON proposal worker |
| `evals/rogue_agent/` | Synthetic action cases, runner and committed report |
| `src/steerer.py`, `src/llm_wrapper.py` | Research orchestration, model loading and steering hooks |
| `src/retrieval_context.py`, `src/steering_cache.py` | Quoted/bounded context and numeric steering caches |
| `src/knowledge_base.py`, `src/hybrid_knowledge_base.py` | Vector and optional graph retrieval |
| `src/workspace_diagnostics.py`, `src/transport_stats.py` | Activation/transport summaries and paired statistics |
| `main.py`, `config.py` | Research CLI, model aliases and presets |
| `experiments/` | Model experiments and historical results |
| `tests/`, `.github/workflows/containment.yml` | Regression tests and CI |

## Research references

- Turner et al., [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248).
- Rimsky et al., [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/).
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

These are methodological references; their findings do not validate Machine-POI's
particular vectors, checkpoints, or containment implementation.

## License status

This checkout does not contain a license file. The earlier README's MIT label was
not accompanied by license terms; a repository license still needs to be supplied
by the owner. Model and dataset terms must be checked separately.
