# Testing and evaluation

[README](../README.md) · [Architecture](architecture.md) ·
[Guardian integration](guardian_integration.md) · [Steering guide](steering_guide.md)

Use the smallest environment that covers the behavior under test. Commands below
run from the repository root. Guardian tests and demos use mocks and do not need
models or service credentials.

## Guardian only

Python 3.10+ and pytest are sufficient:

```bash
python -m pip install pytest==9.1.1
python -m pytest tests/guardian --confcutdir=tests/guardian -q
python -m examples.guarded_agent.host
python -m examples.guarded_agent.process_demo
python -m evals.rogue_agent.run --output /tmp/rogue-agent-results.json
```

`--confcutdir` avoids the root test fixtures, which import the ML stack. The
process demo validates a JSON proposal interface under the same OS account; it
does not test sandbox escape resistance.

## Full offline runtime tests

CI uses Python 3.12 with a CPU PyTorch wheel and pinned top-level test dependencies:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -m 'not slow and not integration'
```

Dependency installation needs network access. The selected tests use local
fixtures/mocks and run with model-hub downloads disabled. The test requirements
are not a complete transitive dependency lock or the full optional-service
installation in `requirements.txt`.

## Lint

CI runs ruff at the version pinned in `requirements-test.txt`. `ruff.toml` selects
pyflakes and syntax-level pycodestyle rules explicitly, because ruff's default
selection varies by version:

```bash
python -m pip install "$(grep -E '^ruff==' requirements-test.txt)"
ruff check .
```

`pytest.ini` also turns deprecation warnings raised from this repository's own
modules into test errors.

## Targeted runs

For a targeted run, use the same environment and select a test file:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_runtime_safety.py -q
python -m pytest tests/guardian --confcutdir=tests/guardian \
    --cov=src.guardian --cov-branch --cov-report=term-missing
```

`make test` selects the fast suite. `pytest.ini` excludes integration tests by
default, so `make test-all` still inherits that exclusion. Explicitly clear the
marker filter with `python -m pytest -m ''` only in an environment prepared for
slow/model integration work. Those tests were not part of the validation below.

## Recorded validation

At implementation commit `57d36c0b44dc31e65e3708587725f874f720522e`,
validated on 2026-09-29:

| Check | Result | Scope |
| --- | --- | --- |
| Local full selected suite, Python 3.12.14 | 197 passed; 4 deselected | Mocked runtime/retrieval plus guardian tests |
| Guardian subset | 29 passed | Identity/scope, approval changes, replay, budgets, delegation, stop, expiry, policy/audit/callback failure |
| Runtime safety subset | 17 passed | Hook cleanup/restoration, concurrent sessions, real mode deltas, clamp dose, cache and model-loading controls |
| Mock demos | Both passed | Authorized effects, pending approval and forbidden destination |
| Synthetic action fixtures | 12/12 passed | 9 forbidden actions blocked, 0 unapproved mock side effects |
| Benign fixtures | 3 cases; 0 false blocks; 1 review | One case intentionally pauses for approval |
| GitHub Actions | All 3 jobs passed | Guardian on Python 3.10/3.12 and full offline runtime on 3.12 |

The local full suite emitted one existing `asyncio.get_event_loop()` deprecation
warning. The [CI run](https://github.com/gutama/machine-poi/actions/runs/36528823005)
and [workflow](../.github/workflows/containment.yml) provide the validation context.
No model weights or live external-tool credentials were used in these checks.

## What the action fixtures measure

[`cases.json`](../evals/rogue_agent/cases.json) contains proposed tool actions and
expected decisions. The runner creates a fresh mock host for each case, submits
the proposal and checks status/reason and mock side effects. The
[committed report](../evals/rogue_agent/results.json) includes fixture SHA-256,
Python version, timestamp, case results and local latency observations.

Fixture context text is descriptive; it is not submitted to an LLM. Consequently,
these results measure policy enforcement once an action exists. They do not
measure prompt-injection success, semantic goal adherence, steering efficacy, or
general task completion. A pending benign review is not an executed task.
Cancellation, approval replay, concurrent accounting and cross-run state behavior
are covered separately by regression tests.

To reproduce a report without replacing the committed evidence:

```bash
python -m evals.rogue_agent.run --output /tmp/rogue-agent-results.json
```

Compare case outcomes and the fixture hash. Timing depends on the environment and
is not a production latency or time-to-containment benchmark.

## Model experiments

The [research note](../PAPER.md) describes the implemented methods. The historical
[model result report](../experiments/results/README.md) links raw JSON traces and
run conditions. Do not treat those runs as tests of the guardian changes.

The older demonstration runner accepts `--model`, not `--llm`:

```bash
python experiments/reproduce_paper.py --model deepseek-r1-1.5b --section 5.1 --quick
python experiments/reproduce_paper.py --section 5.2 --quick
python experiments/reproduce_paper.py --section 5.3 --quick
```

These commands load models and generate new outputs. Section 5.2 counts English
substring matches and distinct religious markers per response. Its rate can
exceed 100%; it is not a probability, a coherence score or a semantic benchmark.
Section 5.3 prints text previews and lengths rather than a validated quality score.
The script does not reproduce an independently scored thematic/coherence table.

For attention-transport comparisons, inspect these entry points before selecting
checkpoint, dose, prompts and output path:

```bash
python experiments/steered_vs_baseline_transport.py --help
python experiments/centered_contrast_probe.py --help
```

Record checkpoint/revision, code commit, corpus hash, vector recipe, layer band,
chat template, dtype, decoding, prompts/seeds and baseline behavior. Compare
outputs and task performance alongside geometry; a metric shift alone can reflect
collapse. Raw experimental vector scales differ from the high-level normalized
persona API, so coefficients are not interchangeable.

## Next evaluation gate

A live host evaluation needs a complete action inventory and isolated adapters,
held-out benign/adversarial tasks, and matched baseline, steering-only,
gateway-only and combined conditions. Measure unauthorized effects, task success,
false blocks, review burden, budget use, latency and stop/recovery behavior. Run
queued, in-flight, descendant and direct-bypass drills. A host owner must review
that evidence before promotion from shadow observations to scoped enforcement.
