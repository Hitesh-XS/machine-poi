# Steering and retrieval guide

[README](../README.md) · [Architecture](architecture.md) ·
[Guardian integration](guardian_integration.md) · [Testing](testing.md)

This guide covers the research API. Run examples from the repository root after
installing research dependencies. Inference examples load model weights; graph
indexing/querying can invoke the configured provider. They do not automatically
pass through the guardian.

## Generate with mean-activation steering

```python
from src import QuranSteerer

steerer = QuranSteerer(
    llm_model="qwen2.5-0.5b",
    embedding_model="paraphrase-minilm",
)
steerer.load_models()
steerer.config.coefficient = 0.2
steerer.prepare_quran_steering(
    chunk_by="verse", sample_size=8, cache_path="vectors/example_mean.npz"
)

# Use the selected instruct checkpoint's chat template consistently.
prompt = steerer.llm.tokenizer.apply_chat_template(
    [{"role": "user", "content": "How should we resolve a disagreement?"}],
    tokenize=False,
    add_generation_prompt=True,
)
print(steerer.generate(prompt, max_new_tokens=100))
print(steerer.last_run_diagnostics)
print(steerer.generate_unsteered(prompt, max_new_tokens=100))
```

This is a small API demonstration, not a calibrated behavioral evaluation.
`prepare_quran_steering` pools unsteered activations and normalizes the mean at
each layer. `prepare_quran_persona` instead combines normalized verse, paragraph
and surah means with default weights 0.50, 0.35 and 0.15, then normalizes the result.
Use one preparation method for the experiment being measured.

`ContrastiveQuranSteerer.prepare_contrastive_steering(positive_texts,
negative_texts)` constructs normalized differences between activation means.
`prepare_quran_contrastive()` supplies Quran/neutral-text examples. The contrast
can mix language, style and content effects; it does not isolate moral behavior
without appropriate controls.

The wrapper only applies certain chat templates automatically in reasoning mode.
For other instruct-model comparisons, provide consistent formatting explicitly.
A fluent baseline is a prerequisite for interpreting a steering comparison.

## Retrieval and dynamic steering

Continue with the `steerer` instance above to build a vector index and use MRA:

```python
steerer.initialize_knowledge_base()
steerer.knowledge_base.build_index("al-quran.txt")
answer = steerer.generate(
    "How should I handle team conflict?",
    mra_mode=True,
    use_dynamic_steering=False,
)
print(answer)
```

MRA retrieves verse, passage and surah context and adds it to the prompt. This
path assembles its own prompt; check checkpoint formatting when designing an
experiment. Retrieval-derived activation steering is a separate, explicit opt-in:

```python
answer = steerer.generate(
    "How should I handle team conflict?",
    mra_mode=True,
    use_dynamic_steering=True,
    trusted_retrieval=True,
    dynamic_blend_ratio=0.3,
)
```

`trusted_retrieval=True` is an assertion by the integrator, not a corpus integrity
check. Use it only for an intentionally trusted research corpus. Text returned by
MRA and graph retrieval is quoted as reference data with a 12,000-character bound
per MRA resolution, or for the combined graph context. Oversized context raises
an error instead of silently truncating. Quoting does not detect prompt injection.

## Graph retrieval

Use the async API for graph-enhanced generation. This standalone example expects
`GRAPH_MODEL` to name a model accessible to the configured OpenAI account and
`OPENAI_API_KEY` to be available to its client. Indexing can make many provider
calls. Ollama and Gemini adapter factories are also available in
`src/llm_adapters.py`; configure their model, endpoint and credentials for your host.

```python
import asyncio
import os

from src import QuranSteerer
from src.llm_adapters import create_openai_adapter

async def main():
    steerer = QuranSteerer(
        llm_model="qwen2.5-0.5b",
        use_graph_kb=True,
        llm_func=create_openai_adapter(model_name=os.environ["GRAPH_MODEL"]),
    )
    steerer.load_models()
    steerer.prepare_quran_steering(sample_size=8)
    await steerer.initialize_hybrid_knowledge_base()
    # Build once, then reuse the index on later runs.
    await steerer.hybrid_kb.build_index("al-quran.txt", build_graph=True)
    print(await steerer.generate_with_graph(
        "How should I handle team conflict?",
        query_mode="hybrid",
        use_dynamic_steering=False,
    ))

asyncio.run(main())
```

`query_mode` accepts `vector`, `graph`, `hybrid` or `auto`. Async retrieval completes
before the synchronous steering session starts; do not hold that session across
an `await`. Provider calls and index storage need their own authorization boundary
when incorporated into an agent host.

## Injection semantics

Let `h` be a token's hidden state, `v` the supplied vector, and `a` the hook
coefficient. Clamp uses `u = v / (norm(v) + 1e-8)`.

| Mode | Hook operation | Interpretation |
| --- | --- | --- |
| `add` | `h + a * v` | Vector addition |
| `blend` | `(1 - a) * h + a * v` | Interpolation; coefficient must be in [0, 1] |
| `replace` | `v` at every position | Erases the original hidden state; the low-level hook ignores its coefficient |
| `clamp` | `h - dot(h, u) * u + a * u` | Sets a projection along the normalized direction, up to numerical epsilon |

The high-level API applies a layer-distribution scale to the configured
coefficient. For `replace`, it scales the vector before registering the hook.
`SteeringConfig` accepts finite coefficients in [0, 2], with [0, 1] for blend;
the low-level hook accepts finite coefficients, with the same blend constraint.
These are configuration bounds, not validated safety thresholds. Low-level
experiment coefficients and normalized high-level vectors are not interchangeable.

**Clamp coefficient zero still removes the existing projection.** Use
`steering_disabled()` or `generate_unsteered()` for an unsteered baseline. No
injection mode is established as universally more fluent or more stable. Read the
[committed results](../experiments/results/README.md) before interpreting a dose.

## Diagnostics and state lifetime

After `QuranSteerer.generate` or `generate_with_graph`, read
`steerer.last_run_diagnostics`. It contains scalar summaries captured before the
session restores the previous hooks and drops temporary activation tensors. A
new high-level generation resets this field; it is not a per-request history.

At the low level, `SteeredLLM.get_steering_diagnostics()` summarizes currently
captured, enabled hooks after a forward pass. It uses the actual add/blend/replace/
clamp delta. `get_attention_transport_diagnostics(prompt)` makes a separate
forward pass; wrap it in `steering_disabled()` for its baseline. Geometry and
perturbation metrics are research measurements, not action authorization signals.

`QuranSteerer.compare` delegates directly to `SteeredLLM.compare_outputs`: it
compares activation steering without assembling MRA or graph context, and it does
not update `last_run_diagnostics`. For retrieval experiments, explicitly design
and run the desired context/steering conditions through the high-level APIs.

## Model loading and cache migration

| Area | Current behavior | Migration action |
| --- | --- | --- |
| Remote code | Off by default for LLMs and embedders | Review code before opt-in; supply a full 40-character commit revision |
| LLM revision | `QuranSteerer(llm_revision=...)` or `SteeredLLM(revision=...)` | Pin the checkpoint for reproducible runs |
| Embedding revision | `QuranEmbeddings(revision=..., trust_remote_code=...)` | Configure separately; the high-level LLM revision does not pin the embedder |
| Steering caches | Numeric NPZ arrays and JSON model/revision/corpus/recipe metadata | Recompute old/mismatched caches; do not convert them by loading pickle |
| Corrupt artifacts | Invalid arrays/metadata are rejected; supported cache errors trigger recomputation | Other corruption can raise; investigate and rebuild from a trusted source |
| Dynamic retrieval steering | Off by default | Explicitly pass both opt-in flags for trusted-corpus experiments |
| Temporary hooks | Restored after high-level generation, including failure | Use scalar diagnostics instead of relying on retained activation tensors |

Metadata detects accidental cache reuse; it is not a signature. Protect model and
cache storage from agent writes. An unresolved local revision is recorded as
`unresolved`. Repeated registration replaces a layer's previous hook. Use public
wrapper APIs for serialized inference; direct model/hook mutation bypasses them.

## CLI reference

```bash
python main.py --help
python main.py --llm qwen2.5-0.5b --coefficient 0.2 --prompt "What is justice?"
python main.py --quran-persona --interactive
python main.py --preset workspace --layer-distribution workspace --interactive
python main.py --init-db
python main.py --mra --interactive
python compare_models.py --list-models
```

The interactive path calls `QuranSteerer.generate`, so `--mra --interactive` uses
MRA context after the vector index exists. Single-prompt/default/comparison paths
call `compare`; `--mra --prompt` does **not** add MRA context there. `--graph-kb`
configures the graph provider and enables graph index building with
`--init-db --build-graph`; current CLI generation does not call
`generate_with_graph`. Use the async API above for graph generation.

| Flag | Behavior |
| --- | --- |
| `--llm`, `--llm-path` | Registered alias, or `--llm custom --llm-path MODEL_PATH`; default `deepseek-r1-1.5b` |
| `--embedding` | Registered embedding alias; default `paraphrase-minilm` |
| `--revision`, `--trust-remote-code` | LLM revision and reviewed-code opt-in; opt-in requires a full commit hash |
| `--preset` | `gentle`, `moderate`, `strong`, `focused`, `workspace` |
| `--coefficient` | Override strength; API validation permits [0, 2], blend [0, 1]; CLI help still displays [0, 1] |
| `--injection-mode` | `add`, `blend`, `replace`, `clamp` |
| `--layer-distribution` | `uniform`, `bell`, `focused`, `workspace` |
| `--chunk-by`, `--quran-persona`, `--theme` | Select text resolution, weighted persona, or thematic preparation |
| `--quran-path`, `--cache-dir` | Corpus and steering-cache paths |
| `--device`, `--quantize` | Device (`cpu`, `cuda`, `mps`) and optional `4bit`/`8bit` loading |
| `--max-tokens`, `--temperature` | Generation options; not all CLI paths forward temperature, so use the Python API for controlled experiments |
| `--interactive`, `--compare`, `--prompt` | Interactive generation, predefined comparisons, or one comparison prompt |
| `--reasoning` | Model-specific prompt/decoding behavior; inspect it when matching experimental conditions |
| `--init-db`, `--mra` | Build vector index; enable MRA on the interactive generation path |
| `--graph-kb`, `--build-graph` | Configure graph provider; build graph with `--init-db` |
| `--llm-provider`, `--llm-api-model` | Provider (`openai`, `gemini`, `ollama`) and its model name |

The CLI uses a truthiness check when applying `--coefficient`: a zero value falls
back to the preset. Use `set_steering_strength(0.0)` in Python if testing a zero
coefficient, and remember that zero clamp is not an unsteered baseline.

## Registered model aliases

These are the repository's convenience mappings, not a current compatibility or
quality certification for every checkpoint/dependency combination. Custom paths
also require a supported model layout and sufficient memory.

| LLM alias | Checkpoint |
| --- | --- |
| `deepseek-r1-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `phi4-mini` | `microsoft/Phi-4-mini-reasoning` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` |
| `smollm3` | `HuggingFaceTB/SmolLM3-3B` |
| `gemma-270m` | `google/gemma-3-270m-it` |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` |

| Embedding alias | Checkpoint |
| --- | --- |
| `paraphrase-minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `paraphrase-mpnet` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `bge-m3` | `BAAI/bge-m3` |
| `multilingual-e5` | `intfloat/multilingual-e5-large-instruct` |
| `multilingual-e5-large` | `intfloat/multilingual-e5-large` |
| `qwen-embedding` | `Alibaba-NLP/gte-Qwen2-7B-instruct` |
