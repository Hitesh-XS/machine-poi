# Research directions and open problems

[README](../README.md) · [Steering evaluation](evaluation.md) ·
[Research note](../PAPER.md) · [Literature map data](literature_map/README.md)

Machine-POI has four research strands:

- **A. Steering.** Quran-derived activation steering, with a calibrated dose and
  an evaluation harness.
- **B. Language and culture controls.** An Arabic neutral control set and
  measurements of script shift and register.
- **C. Attention geometry.** Diagnostics of curvature and transport in trained
  models.
- **D. Agent containment.** A host-side gateway that checks agent tool actions
  against task grants.

This page maps the recent literature around those strands, recommends where to
take the research next, and lists open problems that collaborators can pick up.
Contributions of any size are welcome; see [how to join](#how-to-join).

## Summary

- **Primary direction: language-matched, capability-audited steering for Arabic
  and Islamic contexts.** Cultural steering is an active niche. Steering toward
  Arabic or Islamic content is almost empty: 7 of 7,322 harvested abstracts
  combine steering with Arabic, Islamic or Quranic terms, and none steers toward
  Islamic content. This repository already has the pieces the question needs.
- **Secondary direction: representation signals as inputs to the gateway.** Agent
  containment is the fastest-growing and most crowded area. A probe or steering
  signal that feeds the gateway's `observe()` call, which never adds authority,
  is a narrower contribution than another gateway.
- **Deprioritize attention geometry as a headline.** It is the slowest-growing
  theme, and its active work designs curved-space architectures rather than
  diagnosing trained models. It stays useful as supporting analysis.

## How the map was built

Twelve themed arXiv queries (listed in
[`literature_map/queries.json`](literature_map/queries.json)), date-clamped to
2023-01-01 through 2026-09-30, returned 7,322 unique papers. No query reached the
harvest cap. Each paper was scored by TF-IDF similarity to this repository's own
documents: the research note, the evaluation guide and results, the geometry
review and the guardian docs. The top decile (732 papers) is the focus corpus.
It was clustered with k-means (k = 12) and its citations were fetched from
Semantic Scholar on 2026-09-30.

Read these caveats before the numbers:

- **2026 is annualized.** January–September counts are divided by 0.75. arXiv as
  a whole also grows, so compare themes with each other, not in absolute terms.
- **Citation counts are immature.** 72% of the focus corpus is from 2026, and
  Semantic Scholar returned records for 622 of 732 papers. For recent papers,
  citations per month measure velocity, not impact.
- **Small bases.** Quran and Islamic NLP grows from 3 papers in 2024, so its
  growth ratio is noisy.
- **No 2D map.** The first two components of the corpus explain 2.3% of its
  variance, so a scatter plot would mislead. The per-strand similarity rankings
  below are used instead.
- **"Cited" means mentioned anywhere in this repository's docs**, matched by
  arXiv ID and title shingles.

## The structural shift

![Growth by theme, 2024 to 2026 annualized](literature_map/theme_growth.png)

| Theme (strand) | 2024 | 2026, annualized | Growth |
| --- | --- | --- | --- |
| Agent policy enforcement (D) | 38 | 889 | 23.4× |
| Agent prompt injection (D) | 30 | 545 | 18.2× |
| Quran / Islamic NLP (B) | 3 | 40 | 13.3× |
| Residual stream, sinks, patching (A) | 86 | 861 | 10.0× |
| Activation steering core (A) | 76 | 621 | 8.2× |
| Sparse autoencoders (A) | 82 | 635 | 7.7× |
| Steering evaluation (A) | 139 | 973 | 7.0× |
| Value / persona steering (A) | 119 | 607 | 5.1× |
| Multilingual steering (B) | 29 | 124 | 4.3× |
| Arabic LLMs (B) | 66 | 229 | 3.5× |
| Cultural / religious alignment (B) | 154 | 376 | 2.4× |
| Attention geometry (C) | 142 | 265 | 1.9× |

- **Agent containment (D) grew fastest and is now crowded.** In the focus corpus,
  the cluster on authorization and capability architectures for agents had no
  papers in 2024 and 1 in 2025, then 54 in January–September 2026.
- **Steering (A) is moving from "does it work" to measurement.** The
  fastest-growing steering cluster covers mechanics: layers, dosing, personas,
  what steering actually changes, and pooling. It went from 1 paper in 2024 to
  68 in 2026.
- **Cultural alignment (B) has plateaued** at 361 papers in 2025 and 376 per year
  in 2026. Arabic LLM work grows modestly. Quran and Islamic NLP is small and
  rising, but it consists of benchmarks and question-answering systems, not
  steering.
- **Attention geometry (C) is the slowest theme.**

Focus-corpus clusters (k = 12, labelled by hand from top terms and titles):

| Cluster | Papers | 2024 | 2025 | 2026 (Jan–Sep) |
| --- | --- | --- | --- | --- |
| Inference-time steering frameworks | 91 | 4 | 20 | 66 |
| Indirect prompt injection: attacks and defenses | 87 | 2 | 12 | 73 |
| Activation-steering mechanics: layers, dosing, personas | 78 | 1 | 8 | 68 |
| Agent runtime policy enforcement | 70 | 1 | 7 | 62 |
| Steering vectors: methods and safety pitfalls | 69 | 12 | 24 | 30 |
| SAE feature steering | 62 | 6 | 21 | 35 |
| Quran and Islamic QA, RAG and datasets | 61 | 8 | 22 | 28 |
| Authorization and capability architectures for agents | 55 | 0 | 1 | 54 |
| Agentic-AI security and privacy surveys | 51 | 0 | 14 | 37 |
| Arabic and cultural alignment | 44 | 4 | 9 | 30 |
| Curvature and Riemannian geometry of attention | 34 | 2 | 10 | 17 |
| MCP / tool-protocol gateways | 30 | 0 | 6 | 24 |

## What the project does not yet cite

The repository's docs cite **9 of the 732 focus papers (1.2%)**. The
highest-ranked uncited work, by relevance and citation velocity:

- **Steering evaluation:**
  - [AxBench](https://arxiv.org/abs/2501.17148) (228 citations): simple baselines
    beat SAE-based steering.
  - [The Linear Representation Hypothesis](https://arxiv.org/abs/2311.03658) (738).
  - [Improving instruction-following through activation steering](https://arxiv.org/abs/2410.12877).
  - [Personalized steering via bi-directional preference optimization](https://arxiv.org/abs/2406.00045).
  - [Improving steering vectors by targeting SAE features](https://arxiv.org/abs/2411.02193).
- **Language and culture:**
  - [Do Multilingual LLMs Think In English?](https://arxiv.org/abs/2502.15603) (97).
  - [Investigating Cultural Alignment of LLMs](https://arxiv.org/abs/2402.13231).
  - [Hofstede-based cultural alignment](https://arxiv.org/abs/2309.12342).
  - [Fanar](https://arxiv.org/abs/2501.13944).
- **Agent security:**
  - Benchmarks: [InjecAgent](https://arxiv.org/abs/2403.02691) (628) and
    [AgentDojo](https://arxiv.org/abs/2406.13352) (151).
  - Privilege and policy systems: [Progent](https://arxiv.org/abs/2504.11703),
    [Cordon](https://arxiv.org/abs/2606.17573),
    [IPIGuard](https://arxiv.org/abs/2508.15310) and
    [DRIFT](https://arxiv.org/abs/2506.12104).
  - The [MCP safety audit](https://arxiv.org/abs/2504.03767).
- **Canonical work outside the focus corpus.** Its vocabulary barely overlaps
  this project's, but reviewers will expect it:
  - [Representation Engineering](https://arxiv.org/abs/2310.01405);
  - [Refusal is mediated by a single direction](https://arxiv.org/abs/2406.11717);
  - CaMeL (only its follow-ups were harvested).

The full ranked list is in the [reading list](literature_map/reading_list.md).

## Nearest work to each strand

- **A. Steering method and evaluation.** Direct comparisons for the dose ratio
  and pooling work:
  - [Persona Dosing: Calibrated Activation Steering for Graded Trait Control](https://arxiv.org/abs/2609.36388);
  - [A Geometric Account of Activation Steering through Angle-Norm Decomposition](https://arxiv.org/abs/2606.06735);
  - [When Does Activation Steering Change What a Model Computes From?](https://arxiv.org/abs/2606.29522);
  - [PoolBench](https://arxiv.org/abs/2608.05162);
  - [Mechanistic Indicators of Steering Effectiveness](https://arxiv.org/abs/2602.01716).
- **B. Language and culture controls.** The nearest items are Arabic resources,
  not methods. The closest method work:
  - [Steering Multilingual Models Towards Cultural Knowledge (SemEval-2026 Task 7)](https://arxiv.org/abs/2605.23069);
  - [LangFIR: language-specific SAE features for language steering](https://arxiv.org/abs/2604.03532);
  - [A Universal Vibe? Language-Agnostic Informal Register](https://arxiv.org/abs/2603.26236),
    which bears on the register confound found by this project's evaluation.
- **C. Attention geometry.** All ten nearest papers are already cited, and none
  diagnoses transport in trained models. The strand is uncontested but not
  growing.
- **D. Agent containment.** This strand is crowded.
  - The closest paper to the whole project is
    [Out-of-Band Policy Enforcement at a Trusted Tool Boundary](https://arxiv.org/abs/2608.27646).
    It ranks 4th of 7,322 and is in the top 40 for strands A, B and D.
  - Close neighbours: [Agent libOS](https://arxiv.org/abs/2606.03895),
    [LeaseGuard](https://arxiv.org/abs/2609.24077),
    [Cordon](https://arxiv.org/abs/2606.17573),
    [LLM Agent Capabilities Should Follow Task Intent and Context Source](https://arxiv.org/abs/2609.14631)
    and [ContainmentBench](https://arxiv.org/abs/2607.23999), which evaluates
    post-exposure containment.

Papers that bridge steering and agents:

- [Activation Steering Transfer to Agents](https://arxiv.org/abs/2607.09156);
- [ASA: representation engineering for tool-calling agents](https://arxiv.org/abs/2602.04935);
- [Same Bytes, Different Authority: reserved-token representations in prompt injection](https://arxiv.org/abs/2609.35932).

## The primary direction

The research question: *once language is controlled, does a Quranic direction
change Arabic and Islamic knowledge or value judgments, or only register, and at
what capability cost?*

- **The space is uncrowded.** The neighbouring clusters are small, growing, and
  made of benchmarks and QA systems that do not steer.
- **The repository combines assets no harvested paper combines:**
  - a verse-cited Quran corpus;
  - an Arabic neutral control set;
  - calibrated dose ratios;
  - a harness that measures script shift, degeneration, likelihood under the
    unsteered model and capability together.
- **Benchmarks now exist.** [QuranicMMLU](https://arxiv.org/abs/2609.22038),
  [IslamicMMLU](https://arxiv.org/abs/2603.23750) and
  [PalmX](https://arxiv.org/abs/2509.02550) can join ARC-Easy, so steering can be
  scored on Islamic knowledge and cultural appropriateness, not only on an
  embedding proxy.
- **It runs on modest hardware.** The committed
  [Qwen2.5-0.5B run](../experiments/results/README.md) ran on CPU. It found that
  centered steering at dose 0.05 left ARC-Easy accuracy unchanged within its
  interval, 0.1 raised a religious register but cost 15 points of accuracy, and
  0.2 degenerated 81% of outputs.

## The secondary direction

The project has both a steering stack and a capability gateway, which is
unusual. The gateway space itself is crowded, with 54 capability-architecture
papers in 2026 alone. Instead of competing there, the project can feed
probe- or steering-derived injection indicators into `Gateway.observe()`. That
call records the signal and moves a run into the `observe` state; it never adds
authority, and today it does not remove any either. Part of the work is deciding
what a calibrated signal should trigger, such as holding the next action for
review. The evaluation would use AgentDojo and InjecAgent. This needs
tool-capable models, so it costs more than the primary direction.

## Open problems

Each item says what is needed and where to start. Items marked **no code** need
domain knowledge rather than programming. Each links to its issue.

### Steering and evaluation

1. **Add Arabic and Islamic benchmarks to the harness** ([#25](https://github.com/gutama/machine-poi/issues/25)). Add QuranicMMLU,
   IslamicMMLU and PalmX as capability metrics next to ARC-Easy in
   [`experiments/steering_eval.py`](../experiments/steering_eval.py) (the
   `metrics.capability` block of a spec). Skills: Python, LLM evaluation.
2. **Rate the blinded outputs (no code)** ([#26](https://github.com/gutama/machine-poi/issues/26)). The committed
   [rating sheet](../experiments/results/qwen2.5-0.5b_phase4_rating_sheet.csv)
   needs two independent raters. The rubric is in
   [the evaluation guide](evaluation.md#human-ratings-of-thematic-relevance).
   Skills: reading English and Arabic answers, familiarity with Quranic themes.
3. **Review the Arabic prompts and controls (no code)** ([#27](https://github.com/gutama/machine-poi/issues/27)). The Modern Standard
   Arabic prompts in [`eval_prompts.json`](../experiments/eval_prompts.json) and
   the neutral control sentences in
   [`neutral_arabic.txt`](../machine_poi/data/neutral_arabic.txt) await a
   native-speaker review. Skills: MSA.
4. **Write an Arabic retrieval template** ([#28](https://github.com/gutama/machine-poi/issues/28)). The multi-resolution retrieval prompt
   (`_mra_context` in [`steerer.py`](../machine_poi/steerer.py)) is English-only,
   so the model answers Arabic prompts in English when retrieval is on. Skills:
   Arabic, prompt design.
5. **Sweep doses between 0.05 and 0.1, and add a 1–2B model** ([#29](https://github.com/gutama/machine-poi/issues/29)). The committed
   run brackets the trade-off but does not locate it. Copy
   [the spec](../experiments/specs/qwen2.5-0.5b.json), tune on the `dev` split
   and report `test`. Skills: running experiments; CPU time is enough.
6. **Compare against the strongest baselines** ([#30](https://github.com/gutama/machine-poi/issues/30)). Run AxBench-style prompting and
   SAE baselines against the centered recipe, and compare the dose calibration
   with Persona Dosing and the angle-norm account. Skills: interpretability.

### Agent containment

7. **Run the gateway on AgentDojo and InjecAgent** ([#31](https://github.com/gutama/machine-poi/issues/31)). Route the benchmarks' tool
   calls through [`Gateway`](../machine_poi/guardian/gateway.py) and report
   attack success and benign utility. The committed
   [containment report](../evals/rogue_agent/results.json) covers
   already-proposed actions only. Skills: agent security, Python asyncio.
8. **Feed representation signals into `observe()`** ([#32](https://github.com/gutama/machine-poi/issues/32)). Train a probe for injected
   authority (see [Same Bytes, Different Authority](https://arxiv.org/abs/2609.35932))
   and pass its calibrated output to the gateway as an `injection_indicator`.
   Then decide, with evidence, what the signal should trigger. Skills:
   interpretability and security.

### Writing

9. **Cite and position the related work** ([#33](https://github.com/gutama/machine-poi/issues/33)). Add the gap papers above to
   [PAPER.md](../PAPER.md) and the docs, and position the gateway against
   Out-of-Band Policy Enforcement, Progent, CaMeL, Cordon and ContainmentBench.
   Skills: reading papers carefully.

## How to join

1. **Claim a problem on its issue.** Each open problem above has an issue
   labelled [help wanted](https://github.com/gutama/machine-poi/labels/help%20wanted).
   Comment there with what you plan to do, so work is not duplicated. For a new
   idea, [open an issue](https://github.com/gutama/machine-poi/issues/new).
2. **Set up a checkout.** The guardian needs only the standard library; steering
   work needs PyTorch and Transformers. [Testing](testing.md) lists the commands
   for each environment, and CI runs the same checks on every pull request.
3. **Report results with the harness.** Claims about steering should cite a run
   of [`experiments/steering_eval.py`](evaluation.md), with its provenance block.

The repository does not have a license yet; the owner is choosing one (see the
[README](../README.md#license-status)). Until it does, please discuss larger
contributions in an issue first.

## Data and reproduction

The [literature map folder](literature_map/README.md) holds the scored focus
corpus, the growth tables, the reading list and the query set. Watch these
papers for follow-ups:

- [2608.27646](https://arxiv.org/abs/2608.27646), the closest paper overall;
- [2609.36388](https://arxiv.org/abs/2609.36388), Persona Dosing;
- [2605.23069](https://arxiv.org/abs/2605.23069), cultural steering at SemEval;
- [2609.35932](https://arxiv.org/abs/2609.35932), representations of injected
  authority.
