# Machine-POI: Activation Steering and a Host-Side Action Boundary

Research and implementation note, updated 2026-09-29.

[Architecture](docs/architecture.md) · [Steering guide](docs/steering_guide.md) ·
[Validation](docs/testing.md) · [Committed model results](experiments/results/README.md)

## Abstract

Machine-POI is a Python research implementation of Quran-derived activation
steering and hierarchical retrieval. It constructs mean-activation, weighted
persona and contrastive directions, injects them through model hooks, and reports
activation and attention-transport diagnostics. A separate standard-library
reference gateway evaluates structured agent tool proposals against host-issued
grants, with bound operator review, budgets, replay prevention and revocation.

These components address different questions: steering experiments examine
changes in generated behavior; the gateway constrains authorized effects at a
trusted host boundary. Committed model runs include output collapse and
language/persona spillover, so this note does not claim preserved general
capabilities or reliable moral alignment. Gateway validation consists of mocked
regression tests and synthetic proposed-action fixtures. Live-host protection and
held-out agent/model comparisons remain unmeasured.

## 1. Scope

Changing a model's style or thematic emphasis does not establish authority to use
a tool. Machine-POI therefore keeps behavioral interventions separate from the
host's task grant. Model output, retrieved instructions and diagnostic scores
cannot increase that grant.

The research implementation supports controlled comparisons without changing
model weights. Its verse/passage/surah organization is text chunking, termed
multi-resolution analysis (MRA) in the code. It does not imply a wavelet transform
or a proven hierarchy of semantic concepts.

## 2. Methodological context

### 2.1 Activation engineering

Activation Addition modifies intermediate model activations using directions
constructed from contrasting prompts [1]. Machine-POI uses forward hooks and
several vector recipes; its uncentered corpus-mean path is not the same protocol
as the original contrasting-prompt experiments.

### 2.2 Contrastive activation addition

CAA constructs steering directions from positive/negative activation differences
[2]. `ContrastiveQuranSteerer` pools examples at each layer, subtracts the two
means, and normalizes the difference. Quran-versus-neutral contrasts can mix
language, register, topic and behavior. Attributing an effect to a particular
value requires controls that separate those factors.

### 2.3 Retrieval

Retrieval-augmented generation supplies external context [3]. Machine-POI uses
ChromaDB and optional LightRAG to retrieve text and graph context. This changes
the model input. Dynamic steering additionally constructs a temporary activation
intervention from retrieval; that path is off by default and requires explicit
trusted-corpus opt-in.

## 3. Implemented steering methods

### 3.1 Vector construction

For layer `l`, let `h_l(T_i, t)` denote an unsteered decoder-layer output at token
`t` of sample `T_i`. The mean-activation recipe first pools within each text,
then averages across sampled texts:

$$
\mu_l = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{|T_i|}
\sum_{t=1}^{|T_i|} h_l(T_i,t), \qquad v_l = \operatorname{normalize}(\mu_l).
$$

Each text receives equal weight after token pooling. The high-level persona path
computes separate normalized means for verses, paragraph chunks of up to 19 verses
within a surah, and surahs, combines them with default weights 0.50/0.35/0.15, and normalizes the
combined vector. The contrastive path normalizes `mean(positive) - mean(negative)`.
Zero norms are handled by the underlying normalization routines; a zero vector
has no semantic direction.

The raw-vector experiments in `experiments/` can use different normalization and
dose conventions. Their coefficients must not be substituted directly into the
high-level normalized persona API.

### 3.2 Intervention

Hooks act on decoder-layer **outputs**, not directly on the
`post_attention_layernorm` submodule. The hook supports tensor or tuple outputs
and applies its intervention at all positions present in each forward call.
Let `a` be its coefficient, `v` its supplied vector, and
`u = v / (norm(v) + 1e-8)`:

| Mode | Hidden-state update |
| --- | --- |
| Add | `h' = h + a*v` |
| Blend | `h' = (1-a)*h + a*v` |
| Replace | `h' = v` |
| Clamp | `h' = h - dot(h,u)*u + a*u` |

The high-level API scales the coefficient by layer distribution; for replacement
it scales the vector before registration. Clamp controls a projection rather
than an additive dose. Zero clamp removes that projection and is not a baseline.
No mode has been shown here to preserve fluency at arbitrary strength.

### 3.3 Retrieval and domain bridges

MRA adds verse, passage and surah context to the prompt. Domain bridging first
tries static keyword/theme mappings, then optional graph traversal, then an
embedding-similarity fallback. Graph generation combines graph answers, selected
verses and bridge terms. Retrieved content is bounded and quoted as reference
data. An explicit trust flag permits dynamic retrieval steering for experiments;
it does not verify the corpus or detect malicious instructions.

## 4. Runtime and diagnostics

`SteeredLLM` serializes inference and hook mutations. High-level generation scopes
temporary steering to a session, restores prior vectors/modes/enabled flags on
success or failure. Registration replaces a layer's previous handle. Activation
pooling runs in right-padded batches with steering disabled and removes its
hooks in `finally`. Async graph retrieval finishes before entering
the synchronous session.

Pointwise diagnostics report activation/vector norms, cosine alignment,
projection magnitude and relative perturbation computed from the actual update
for each injection mode, averaged over every steered token of a generation.
High-level generation retains these scalar summaries in `last_run_diagnostics`. Attention-transport experiments summarize a constructed
connection using variation/commutator terms and holonomy. Those quantities are
research diagnostics, with no validated threshold for authorization or safety.

Remote model code defaults off and requires a pinned commit for explicit opt-in.
Steering caches store numeric arrays and identity metadata rather than pickled
objects. Metadata detects accidental reuse, not malicious artifact forgery.

## 5. Experimental evidence

The evidence source is the committed JSON and interpretation in
[`experiments/results/`](experiments/results/README.md). These historical runs were
not repeated during the containment implementation. Earlier narrative output
examples and thematic/coherence scores are omitted here because this repository's
reproduction script does not substantiate that scored table.

### 5.1 Qualitative behavior

The small-model raw-mean runs report degenerate generation. Calibrated centered
Gemma runs report fluent Arabic responses and small pooled transport changes in
some conditions. The four-prompt results on each of two Gemma variants also show
language shifts on neutral prompts and content drift in an engineering answer.
A persona effect is therefore not evidence of selective improvement or preserved
capability. The small samples do not establish model-wide generality.

### 5.2 Thematic measurement

`experiments/reproduce_paper.py --section 5.2` counts English keyword substrings
and distinct religious markers per response. Its reported religious rate can
exceed 100%, and it has no independent coherence score. It does not measure moral
reasoning, multilingual semantic consistency, or authorization. A publishable
behavioral evaluation needs a documented rubric, held-out tasks, language controls
and a comparison protocol.

### 5.3 Dose and geometry

The result report includes 16-prompt reruns for raw-mean SmolLM2/Qwen conditions
and centered SmolLM2. Centered SmolLM2 still shows collapsed generation at a
calibrated perturbation target of at most 0.1. The small Gemma conditions differ.
This motivates model/task-specific evaluation and rules out treating a small
perturbation ratio alone as a safety guarantee.

Transport changes must be read alongside text and task performance. A large shift
can accompany collapse; little pooled change can coexist with language or content
drift. Historical baseline-formatting and measurement limitations are recorded
with the results. The demonstration runner's section 5.3 prints text previews and
lengths for different coefficients, not validated coherence/thematic scores.

## 6. Host-side action containment

The guardian evaluates immutable JSON proposals using a trusted adapter registry.
An authenticated host creates exact-scope grants; adapters resolve real resources,
destinations, classifications and cost bounds. Policy checks scope, versions,
expiry, state and budgets. Sensitive actions pause for review bound to the stored
action and resolved scope. The gateway reserves ancestor-shared budgets, prevents
in-process replay, and rechecks authorization before executor entry.

Stopping a run stops descendants and cancels queued/in-flight tasks cooperatively.
The host supplies credential revocation and incident callbacks. Redacted audit
events form a hash chain; policy/audit failures close the execution path. Current
state is in memory and belongs to one host process/event loop. Authentication,
OS isolation, durable multi-host coordination and remote-effect reconciliation
are integration responsibilities.

At implementation commit `57d36c0`, validation passed 197 selected local tests
(four slow/integration cases excluded) and all CI jobs. Twelve synthetic action
fixtures passed: nine forbidden actions were blocked, with no unapproved mock
effects, while three benign cases had no false blocks and one required review.
These fixtures start from proposed actions and do not run an LLM. They demonstrate
the tested gateway behavior, not prompt-injection resistance or live-agent safety.
See [testing](docs/testing.md) for provenance and reproduction commands.

## 7. Limitations and next experiments

A model with direct tool credentials can bypass an unenforced gateway. A trusted
adapter can also be incorrect, and completed remote effects cannot be undone by
local cancellation. Scope/goal semantics depend on host classification and task
design. These limits require a concrete deployment inventory and response drills.

Next evaluations should compare baseline, steering-only, gateway-only and combined
conditions on held-out benign/adversarial tasks, with matched prompts and model
settings. Report task success, unauthorized effects, false blocks, review burden,
latency and containment time. Separately test language/register confounding,
centering, normalization, layer bands and decoding before making steering claims.
The [containment plan](docs/rogue_agent_containment_plan.md) and
[workspace roadmap](docs/global_workspace_improvement_plan.md) track these gates.

## Appendix: running the research demonstration

After installing the research dependencies, run from the repository root:

```bash
python experiments/reproduce_paper.py --model deepseek-r1-1.5b --section 5.1 --quick
python experiments/reproduce_paper.py --section 5.2 --quick
python experiments/reproduce_paper.py --section 5.3 --quick
```

The section identifiers remain those used by the existing script. These commands
load models and produce new samples; they do not reproduce an independently
validated thematic/coherence table. The [testing guide](docs/testing.md) describes
the offline regression suite and separate attention-transport experiments.

## References

1. Turner, A. M. et al. (2024 revision). *Steering Language Models With Activation Engineering.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248).
2. Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., and Turner, A. (2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL, pp. 15504–15522. [ACL Anthology](https://aclanthology.org/2024.acl-long.828/).
3. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
