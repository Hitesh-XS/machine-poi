# Machine-POI improvement plan

[README](../README.md) · [Architecture](architecture.md) · [Testing](testing.md) ·
[Containment plan](rogue_agent_containment_plan.md) ·
[Workspace roadmap](global_workspace_improvement_plan.md)

Status: Phases 0–3 delivered 2026-09-29; later phases proposed. Reviewed at commit
`8be027d`. Paths and line numbers in the findings refer to that commit; Phase 1
moved `src/` to `machine_poi/`.

This plan covers the whole repository: the guardian gateway, the steering and
retrieval library, the CLI, the experiments and the engineering setup. It
complements the [containment plan](rogue_agent_containment_plan.md), which still
tracks live-host deployment, and the [workspace roadmap](global_workspace_improvement_plan.md).

## How the review was done

- Read every module under `src/`, plus `main.py`, `config.py`, the examples, the
  evals, the experiment scripts and the docs.
- Ran the offline suite in the CI configuration (CPU torch 2.8.0,
  `requirements-test.txt`, Python 3.11). Result: **197 passed, 4 deselected**.
- Ran `ruff check` with default rules. About 100 findings, mostly unused imports
  and f-strings without placeholders. No linter runs in CI.
- Reproduced each bug marked **confirmed** below with a small script or mock.
  No model weights were downloaded.

## Summary

The guardian is carefully designed and well tested. The docs are unusually
candid about limits, and the transport statistics use sound paired tests. The
main problems are:

1. **The library's default steering recipe is the one the project's own results
   rule out.** Centering and dose calibration exist only in `experiments/`
   (M1, M2).
2. **The CLI has bugs that quietly change the experiment being run.**
   Retrieval is dropped, `--coefficient 0` is replaced, presets are ignored, and
   comparisons are confounded (R3–R5).
3. **Retrieval integrity.** One crash path, substring keyword matching that
   produces inappropriate bridges, and verse indices that drift from canonical
   numbering and carry no citations (R1, R2, R8).
4. **The guardian stops a benign run on a race.** An allowed action submitted
   concurrently with a review-required one stops the whole run (G1).
5. **Engineering.** No packaging, no lint in CI, dependency floors that don't
   match the code, and mocks that hide missing attributes (E1–E3).

## Findings

Severity: **P0** means wrong results, a crash, or a safety-relevant defect.
**P1** means it undermines the research claims or usability. **P2** means
maintainability.

### Guardian (`src/guardian/`)

| ID | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| G1 | P0 | **Confirmed.** A benign agent submits an allowed `write_note` and a review-required `send_note` concurrently. The run moves to PAUSED, and the write's pre-execution recheck sees `run_paused` and raises `AuthorizationChanged`. `_finish` then **stops** the run: `on_stop` (credential revocation) fires, the pending review is wiped, and the agent is still told its send is `pending`. | `gateway.py:468-484`, `gateway.py:512-516`; repro via `asyncio.gather(submit(write), submit(send))` on `examples/guarded_agent/host.py` |
| G2 | P1 | **Confirmed.** Gateway state is unbounded, and callers who don't hold a grant can write audit events. There is no way to retire runs, so after `max_runs` issues the gateway can never issue again. Each `preview`/`submit` from an unknown caller appends an audit event (1,000 calls added 1,000 records). `AuditLog` keeps every record in memory, and `records` copies all of them. | `gateway.py:55-56,110-115,290-304,314-318`; `audit.py:28,38-40,66` |
| G3 | P2 | `ToolSpec` checks only top-level argument types, so the contents of `list`/`dict` arguments reach adapters unchecked. This is documented, but there is no helper for nested validation. | `policy.py:29-47` |
| G4 | P2 | The eval fixtures are single, independent actions. Sequences, concurrency, delegation and approval flows appear only in unit tests, not in the reported evaluation. | `evals/rogue_agent/run.py:25-46` |

### Research runtime correctness

| ID | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| R1 | P0 | **Confirmed.** The embedding-fallback bridge tier calls `embedder.create_embedding()`, which does not exist. It raises `AttributeError` whenever MRA mode sees a prompt without one of about 35 keywords, for example "What is truth?". The tests pass only because their unspecced `MagicMock` invents the attribute. | `steerer.py:493`; `quran_embeddings.py:238` |
| R2 | P0 | **Confirmed.** Bridge keywords match as substrings. "terror attacks" matches `error` and yields *forgiveness, learning from mistakes*. "latest contest" matches `test`, and "steam … sometimes" matches `team`/`time`. The same pattern appears in the graph term mapper and in auto-mode routing. | `steerer.py:548-550`; `graph_bridge.py:112-113`; `hybrid_knowledge_base.py:195-196` |
| R3 | P0 | `--mra` is silently ignored on the `--prompt` and default-demo paths. `compare()` goes to `SteeredLLM.generate`, which drops `mra_mode`. In interactive mode, the steered answer gets retrieval context but the baseline doesn't, so the comparison mixes up retrieval and steering. The README documents this instead of fixing it. | `main.py:297-310,494,511`; `llm_wrapper.py:841-842` |
| R4 | P0 | CLI dose precedence is wrong. `--coefficient 0` is falsy and gets replaced by the preset. For every known model, `get_recommended_config` overwrites the preset coefficient, so `--preset gentle`/`strong` never changes the dose. `recommended_layers` and the preset's `chunk_by` are never used. `--theme` first computes mean vectors and then discards them. | `main.py:454-459,463-469`; `config.py:477-478` |
| R5 | P1 | Steered-vs-baseline comparisons sample twice with no seed (default `do_sample=True`, `temperature=0.7`), so sampling noise mixes into every comparison. | `llm_wrapper.py:868-887`; no `manual_seed` in `src/` |
| R6 | P1 | `--init-db --build-graph` calls `asyncio.run` twice on LightRAG objects created in the first loop. The `*_sync` wrappers use deprecated `get_event_loop().run_until_complete`, which fails inside a running loop and is the source of the deprecation warning in `testing.md`. | `main.py:440-441`; `graph_bridge.py:232`; `hybrid_knowledge_base.py:264`; `lightrag_adapter.py:259,270`; `llm_adapters.py:179,227` |
| R7 | P1 | The local LightRAG adapter extracts entities with the steered model while steering is still active, so the knowledge graph gets built from steered output. | `llm_adapters.py:228-235` |
| R8 | P1 | Verse identity is lost. `min_chunk_length=10` drops 45 short verses in verse mode, such as the muqaṭṭaʿāt (e.g. "الم" at 2:1) and 20:30. (An earlier count of 19 measured bytes, not characters.) Every later index then shifts away from canonical numbering. Chroma metadata stores only a positional `index`, so retrieved verses carry no surah:āyah citation. The 19-verse passages also cross surah boundaries. | `quran_embeddings.py:203,205-212`; `knowledge_base.py:125` |
| R9 | P1 | A stale index is reused silently. An existing Chroma collection is skipped regardless of embedding model or corpus hash, so switching `--embedding` reuses vectors of the wrong model or dimension. The KB also loads a second copy of the embedding model. | `knowledge_base.py:59,103-106` |

### Steering methodology

| ID | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| M1 | P1 | The default recipe is the one the results discredit. `prepare_quran_steering` and the persona path use the **uncentered** mean. The results report shows this vector is about 93% generic (cos 0.999 with the neutral mean) and collapses small models. Centered CAA vectors and calibration exist only in `experiments/`. | `steerer.py:915-921,1099-1105`; `experiments/steered_vs_baseline_transport.py:181-202`; `experiments/centered_contrast_probe.py:301-312`; `experiments/results/README.md` |
| M2 | P1 | The dose scale is nearly inert and differs between models. High-level vectors are unit-norm, and the coefficient is capped at 2.0. Committed mean-activation norms are 59–96 (Gemma 4) and about 2,375 (SmolLM2), which bounds the relative perturbation at about 0.02–0.03 and 0.001. The calibrated Gemma runs that changed behavior used about 0.08. So "gentle/moderate/strong" are not comparable across models, and negative (steer-away) doses are rejected. | `steerer.py:187-188`; `config.py` presets; `geometry/*/quran_norm` in `experiments/results/*.json` |
| M3 | P1 | Language confound. `al-quran.txt` is Arabic (6,236 lines). The default negatives are 10 English sentences repeated to 50, so the contrast vector mostly encodes Arabic vs. English. That is consistent with the reported language shift on neutral prompts. | `steerer.py:1533-1547`; `experiments/results/README.md` |
| M4 | P1 | Mean pooling includes BOS/special-token positions. Those carry the "massive activation" component that the results identify as the dominant generic direction. | `steerer.py:728,912,1096,1441,1457` |
| M5 | P2 | The pooling loop is copied five times. It is unbatched (one forward pass per text), captures **all** layers, and clones every one. The hook also clones full hidden states on every forward pass, including each decode step, so `last_run_diagnostics` describes only the final token. | `steerer.py:719-731,900-913,1089-1097,1430-1458`; `llm_wrapper.py:189,890-932` |
| M6 | P2 | Dead or misleading code. `SteeringVectorExtractor` random/Xavier projections of sentence embeddings into the residual stream have no semantic grounding and no high-level caller. `ContrastiveSteeringExtractor` is computed redundantly, and on unequal lists `zip` truncates it. Graph "traversal" never produces bridges. `graph_entities`/`relationships` are always empty. `fusion_strategy` and `get_weighted_embedding` do nothing. | `steering_vectors.py:53-75,259-264`; `steerer.py:1483-1487`; `graph_bridge.py:173-186`; `hybrid_knowledge_base.py:172,203-247`; `knowledge_base.py:237-243` |
| M7 | P1 | There is no behavioral evaluation harness. Section 5.2 of `reproduce_paper.py` counts English substrings, and its rate can exceed 100%. The planned "workspace audit mode" does not exist yet. | `experiments/reproduce_paper.py`; `global_workspace_improvement_plan.md` §4 |

### Engineering

| ID | Sev | Finding | Evidence |
| --- | --- | --- | --- |
| E1 | P1 | The project isn't packaged. There is no `pyproject.toml`, and library modules `sys.path.insert` the repo root to import the top-level `config.py`. | `steerer.py:32-38`; `knowledge_base.py:19`; `quran_embeddings.py:19` |
| E2 | P1 | Dependencies don't match the code. The `dtype=` load keyword (transformers ≥ 4.56) conflicts with `transformers>=4.40`. The `load_in_8bit`/`load_in_4bit` kwargs are deprecated in favor of `BitsAndBytesConfig`. `google-generativeai` is deprecated in favor of `google-genai`. Optional services (LightRAG, Chroma, OpenAI, Gemini, bitsandbytes) are all mandatory. | `llm_wrapper.py:353-361`; `requirements.txt` |
| E3 | P1 | CI runs no lint or coverage. The tests use unspecced `MagicMock`s, which is how R1 survived. There are no CLI tests, which is how R3/R4 survived. | `.github/workflows/containment.yml`; `tests/test_steerer.py` |
| E4 | P2 | Three model registries (`config.LLM_MODELS`, `SteeredLLM.SUPPORTED_MODELS`/`REASONING_CONFIGS`, `MODEL_CONFIGS`) duplicate paths and hardcode dimensions. `MODEL_CONFIGS` has seven identical entries and an unused `residual_stream` key. The Gemma 4 checkpoints used in the experiments are missing from the CLI. | `config.py:102-177`; `llm_wrapper.py:32-75,257-290` |
| E5 | P2 | Instruct models get raw, untemplated prompts unless reasoning mode is on. The experiments use `chat_prompt`, but the library doesn't. | `llm_wrapper.py:837`; `experiments/centered_contrast_probe.py:65` |
| E6 | P1 | There is no license file, and the corpus source, edition and terms are undocumented (the README already flags the license gap). | repo root |
| E7 | P2 | Docs hardcode drifting numbers ("197 tests") and describe bugs as caveats ("some CLI comparison paths bypass retrieval"). | `README.md`, `PAPER.md`, `testing.md` |

## Phased plan

Two tracks can run in parallel: the guardian (G) and research (R/M/E). Each item
ships with a regression test that fails before the fix.

### Phase 0: correctness fixes (delivered)

Delivered as one commit per item, each with regression tests that fail on the
previous code. Two choices went beyond the table below. Without `--preset` or
`--layer-distribution`, a registered model now steers its recommended layer band
rather than the bell-selected band. The sync wrappers reuse one private event
loop per thread rather than calling `asyncio.run` each time, so LightRAG
resources stay on the loop they were created on.

| Item | Change | Acceptance |
| --- | --- | --- |
| G1 | In the pre-execution recheck, treat a lineage that is only PAUSED as "dispatch already authorized". PAUSED holds *new* dispatch, and the action was authorized and its budget reserved before the pause. Keep STOPPED, expiry, tool/scope/binding changes as hard failures. | Concurrent allow+review test: the write executes, the run stays PAUSED with the pending review intact, and `on_stop` isn't called. The existing guardian suite and fixtures still pass. |
| R1 | Use `create_embeddings([query])[0]`. Share one cached theme index between `QuranSteerer` and `GraphBridgeGenerator`, which currently re-embeds 40 themes on every call. | A test with `create_autospec(QuranEmbeddings)` covers tier 3. |
| R2 | Tokenize with `\b`-bounded regex (and a small stem list) in all three matchers. | "terror", "latest", "steam", "sometimes" produce no bridges; the existing keyword tests still pass. |
| R3 | Move comparison into `QuranSteerer.compare()`. Build the final prompt once (MRA/graph context), then generate steered and unsteered from the **same** prompt and seed. Make `SteeredLLM.generate` reject unknown kwargs instead of filtering them. | CLI test: `--prompt --mra` retrieves once, and both arms get identical context. |
| R4 | Precedence becomes CLI > preset > model default, compared with `is not None`. Wire `recommended_layers` into `config.target_layers`, drop the `--theme` pre-pass, and fix the interactive help text (range 0–2). | Parametrized `main` tests with a mocked steerer cover `--coefficient 0`, `--preset gentle` and `--preset strong`. |
| R6 | Use one `asyncio.run(async_main())` for DB/graph build. `*_sync` wrappers use `asyncio.run` and raise a clear error inside a running loop. Use `get_running_loop()` in adapters. | No `get_event_loop` in `src/`; the suite runs with `-W error::DeprecationWarning`. |
| R7 | Wrap local-adapter generation in `steering_disabled()`. | A test asserts that hooks are disabled during adapter calls. |
| E3a | Add `ruff check` (default rules) to CI and fix the roughly 100 existing findings. Switch steerer/KB mocks to `spec=`/`create_autospec`. | CI lint job is green. |

### Phase 1: foundations (delivered)

Delivered as one commit per item with regression tests. Choices beyond the table:
- The package is `machine_poi` with a flat layout, so the guardian demos and
  `python main.py` still run from a checkout without installing. The `src`
  import path was dropped without an alias (decision 3). The base package has
  no dependencies, so there is no separate `guardian` extra, and the vector
  store ships in `research` because the steerer imports it unconditionally.
- Experiment scripts keep inserting the repository root into `sys.path` so they
  run from a checkout; library code no longer does.
- The Gemma 4 aliases carry no recommended dose, because the calibrated
  coefficients in `experiments/results` applied to unnormalized vectors.
- The two transport experiments and the local LightRAG adapter pass
  `chat_template=False`, so their recorded or hand-built prompt formats are
  unchanged.

| Item | Change | Acceptance |
| --- | --- | --- |
| E1 | Add `pyproject.toml` with a real package name. Move `config.py` into the package, remove the `sys.path` hacks, and add a `machine-poi` console script. Extras: `guardian` (stdlib only), `research`, `rag`, `graph`, `providers`, `test`. | `pip install -e .[test]` works; `python -m pytest` runs without path hacks; the guardian installs with zero dependencies. |
| E2 | Raise the transformers floor or support both `dtype`/`torch_dtype`. Use `BitsAndBytesConfig`, migrate to `google-genai`, and add a constraints file for CI. | CI installs from the extras plus the constraints file; the 4-bit/8-bit paths are covered by a mocked unit test. |
| E4, E5 | Merge the model registries into one table (path, chat template use, reasoning config), and read dimensions from the loaded config. Add the Gemma 4 aliases. Move `chat_prompt` into `SteeredLLM` and turn it on by default for instruct checkpoints. | One registry is referenced from CLI, wrapper and experiments; a test checks templating is applied. |
| R5 | Add `seed` and `do_sample` to generation and comparison. Comparisons default to greedy or a fixed seed, and the settings are recorded in outputs. | Repeated comparisons are byte-identical under a fixed seed on CPU. |

### Phase 2: corpus and retrieval integrity (delivered)

Delivered as one commit per item with regression tests. Notes beyond the table:
- 45 verses were being dropped, not 19. Any corpus that is not exactly 6,236
  lines raises `CorpusError`, and an explicit corpus path that does not exist is
  an error rather than a silent fallback to the checkout's copy.
- Index identity lives in each Chroma collection's metadata. Indexes built before
  this change need one `--init-db --rebuild`, and steering caches moved to format 2.
- Graph bridges use real graph neighbors (the neighbor-based option); the step that
  stored an LLM answer snippet as an "entity" was removed.
- Also fixed: `compare_models.py --compare-embeddings` always raised `TypeError`.

| Item | Change | Acceptance |
| --- | --- | --- |
| R8 | Parse the corpus once into `(surah, ayah, text)` records, asserting 114 surahs and 6,236 verses against `SURAH_VERSE_COUNTS`. Keep short verses, store `surah`/`ayah` in Chroma metadata, and add `[S:A]` citations to prompt context. Passages stay within a surah. | Unit tests pin 2:1 = "الم" and the final verse 114:6. Retrieved items include citations. |
| R9 | Store the embedding model, dimension, corpus SHA-256 and schema version in collection metadata. On mismatch, rebuild when `--rebuild` is given, otherwise fail clearly. Share one embedder instance between the steerer and the KB. | Switching `--embedding` without `--rebuild` fails clearly. |
| M6 (graph) | Either implement neighbor-based bridge extraction and fill `graph_entities`/`relationships`, or document that graph bridging currently falls back to embeddings. Remove the unused `fusion_strategy`. | Docs match behavior; a test covers whichever path is kept. |

### Phase 3: steering methodology (port the evidence into the library) (delivered)

Delivered as one commit per item with tests. Notes beyond the table:
- M5 misses its speed target. On CPU, batched pooling of 50 texts was 0.65× the
  unbatched speed until batches were grouped by token length; with that it is
  1.4–1.7×, not 3×. GPU was not measured. Results match one-text-at-a-time
  pooling to floating-point tolerance.
- M2 measures the layer scale as the **median** per-token norm, not the mean: the
  first-position attention-sink token can inflate a mean by an order of
  magnitude. Diagnostics add `median_activation_norm` and the achieved
  `dose_ratio`; `relative_perturbation` keeps its mean-based definition. The
  library default is ratio 0.05. The `strong` and `workspace` presets moved from
  clamp to add mode, since a ratio has no clamp meaning; clamp, blend and replace
  take a raw `--coefficient`. Per-model recommended coefficients were removed.
- M3 uses 120 original Modern Standard Arabic sentences written for the project,
  pending review by a native speaker. The optional English-translation contrast
  was not added.
- M1 bumps steering caches to format 3; rejected caches log which field differs.
  The CLI gains `--recipe`. Retrieval-driven (dynamic and thematic) vectors are
  still uncentered means; Phase 4 should measure whether to center them too.
- M6 deleted `steering_vectors.py` and its tests.

| Item | Change | Acceptance |
| --- | --- | --- |
| M5 | One batched `pooled_layer_means(texts, layers, batch_size, exclude_special=True)` with attention masks that captures only the requested layers. Every recipe uses it. Hook capture becomes opt-in, and diagnostics accumulate running statistics across decode steps. | The five pooling copies are gone; CPU vector build for 50 verses is at least 3× faster at the same numerical result (tolerance test against the unbatched path). |
| M4 | Exclude BOS, special and padding positions from pooling by default, with a documented flag to restore the old behavior. | Test with a synthetic high-norm BOS activation. |
| M1 | Make centered CAA (`mean(pos) - mean(neg)`) the default for `prepare_quran_steering` and the persona. Keep the uncentered mean behind `recipe="raw_mean"` with a warning, and record the recipe in the cache metadata. | Cache/format bump; existing caches are rejected with a clear message. |
| M2 | Express dose as a **target relative perturbation** per layer, calibrated from activation norms on a small calibration set (port from `centered_contrast_probe.py`). Presets become target ratios (for example 0.02 / 0.05 / 0.1), and signed doses are allowed for steer-away ablations. `last_run_diagnostics` reports the achieved ratio. | For a mocked model with known norms, the achieved ratio matches the target within 5%. Docs explain that ratios, not raw coefficients, transfer between models. |
| M3 | Add language-matched controls. The negative set is Arabic non-Quranic prose (e.g. MSA news or encyclopedia text, subject to license; see decisions). Optionally add a parallel English-translation contrast. Deduplicate negatives and hash both sets into the metadata. | The eval (Phase 4) reports the language-ID shift on neutral prompts for each control. |
| M6 (vectors) | Remove `SteeringVectorExtractor` projection paths, or move them to `experiments/` with a warning. Delete the redundant `ContrastiveSteeringExtractor` computation in the steerer. | No high-level path imports projection code. |

### Phase 4: evaluation harness (implements roadmap §4 "workspace audit mode")

Add `experiments/steering_eval.py`, driven by a YAML/JSON spec:

- **Conditions:** baseline, raw mean, centered, centered + calibrated at 2–3
  ratios, RAG-only, RAG + steering.
- **Prompts:** held-out sets in English and Arabic (neutral, value-laden,
  technical), with train/test separation from any prompts used for tuning.
- **Metrics:**
  - language-ID rate
  - repetition / distinct-n (collapse detector)
  - perplexity of steered text under the unsteered model
  - accuracy on a small capability set (e.g. a subset of ARC/GSM8K) for retention
  - a documented rubric or judge score for thematic relevance, with inter-rater checks on a sample
  - transport ρ/holonomy via `transport_stats.paired_test`
- **Provenance:** code commit, model revision, corpus hash, recipe, layer band,
  dose ratio, chat template, decoding, seeds.

Acceptance: one committed run on a small ungated model, with a README table that
has confidence intervals. `reproduce_paper.py` §5.2 is retired or relabeled.
PAPER.md claims are updated to cite only this harness.

### Phase 5: guardian hardening and cleanup

| Item | Change | Acceptance |
| --- | --- | --- |
| G2 | Add `retire(operator, run_id)` for STOPPED or expired runs with no in-flight tasks. It frees capacity and keeps a compact tombstone set so run IDs can't be reused. `AuditLog` gets a write-through sink mode that keeps only a bounded tail in memory. Events from unknown callers are counted and sampled rather than appended one by one. | Capacity is reusable after retire; 10k unknown-caller previews add at most a bounded number of records; chain verification still passes. |
| G3 | Add optional nested validators on `ToolSpec` (callable or JSON-schema subset). | Tests cover rejected nested fields. |
| G4 | Extend fixtures with multi-step sequences (draft → send → approve), concurrent submissions (a G1 regression), delegation narrowing and stop-during-flight. Report these in `results.json`. | The fixture count grows; zero unapproved effects. |
| E7 | Generate the evidence numbers in docs from CI artifacts, or drop the exact counts. Remove caveats for bugs fixed in Phase 0. | No hardcoded test counts in the README. |

The live-host gates (held-out model traces, bypass and kill-switch drills,
staged enforcement) stay in the [containment plan](rogue_agent_containment_plan.md).

## Decisions needed from the owner

1. **License and corpus provenance (E6).** Choose a repository license, and record
   the source, edition and terms of `al-quran.txt` (plus a checksum) in a NOTICE.
2. **Neutral Arabic control corpus (M3).** Decided: original sentences written
   for the project (no third-party license); a native-speaker review is pending.
3. **Package rename (E1).** Decided: `machine_poi`, clean break, no `src` alias.
4. **G1 semantics.** Phase 0 implemented the recommendation: already-dispatched
   actions finish while a review is pending. The stricter alternative holds them
   until the review resolves and refunds the reservation, which adds latency and
   code. Revisit if a host needs pause to freeze in-flight work.
5. **Projection-based vectors (M6).** Decided: deleted.

## Not recommended

- Using any transport or perturbation metric as an authorization or safety gate.
  The existing docs already rule this out, and nothing here changes that.
- Raising the high-level coefficient cap before M2. The fix is a calibrated,
  model-relative dose, not a larger absolute number.
