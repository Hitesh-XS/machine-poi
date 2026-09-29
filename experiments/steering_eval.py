"""
Steering evaluation harness (improvement plan Phase 4).

Runs a JSON or YAML spec of conditions over held-out English and Arabic
prompts: an unsteered baseline, raw-mean and centered vectors at calibrated
dose ratios, and retrieval (RAG) with and without steering. It writes every
output with its metrics, per-condition means with 95% bootstrap intervals,
paired differences from the baseline, and the provenance of the run.

    python experiments/steering_eval.py --spec experiments/specs/qwen2.5-0.5b.json
    python experiments/steering_eval.py --score-ratings KEY.json RATER_A.csv RATER_B.csv

Metrics (see machine_poi/evaluation.py and docs/evaluation.md):
- script: share of outputs written mostly in Arabic script, by prompt
  language (a script count, not full language identification)
- fluency: distinct-2 and a degeneration flag
- nll: mean token NLL of each output under the unsteered model, given the
  same final prompt
- capability: zero-shot ARC-Easy accuracy under the condition's hooks
- thematic proxy: embedding contrast between each output and the Quran
  versus neutral-control centroids; a proxy, not a judgment of relevance.
  Human ratings use the blinded rating sheet and rubric.
- transport: attention non-abelian ratio rho and holonomy on the English
  prompts, paired with the baseline
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from machine_poi import evaluation as ev  # noqa: E402
from machine_poi.config import TEST_PROMPTS, THEMATIC_TEST_PROMPTS  # noqa: E402
from machine_poi.controls import calibration_texts, neutral_texts, texts_sha256  # noqa: E402
from machine_poi.steerer import (  # noqa: E402
    QuranSteerer,
    SteeringConfig,
    layer_distribution_scale,
    select_target_layers,
)

DEFAULT_SPEC = {
    "model": "qwen2.5-0.5b",
    "revision": None,
    "embedding_model": "paraphrase-minilm",
    "device": "cpu",
    "seed": 42,
    "prompts": {"file": "experiments/eval_prompts.json", "split": "test"},
    "decoding": {"max_new_tokens": 80},
    "chat_template": None,
    "steering": {
        "layer_distribution": "bell",
        "target_layers": None,
        "sample_size": 50,
        "chunk_by": "verse",
    },
    "metrics": {
        "capability": {"dataset": "arc_easy", "n_items": 100},
        "transport": {"n_prompts": 8, "lang": "en", "eta": 1.0, "max_loop_positions": 8},
        "thematic_proxy": {"n_verses": 200},
    },
    "rag": {"use_domain_bridges": True},
    "rating_sheet": {"per_condition": 6, "category": "value"},
    "bootstrap": {"n_boot": 2000, "n_perm": 5000},
}
NEAR_DUPLICATE_JACCARD = 0.6
RUBRIC_VERSION = 1


# ---------------------------------------------------------------------------
# Spec, prompts and held-out checks
# ---------------------------------------------------------------------------

def merged(defaults: dict, overrides: dict) -> dict:
    result = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merged(result[key], value)
        else:
            result[key] = value
    return result


def load_spec(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # PyYAML ships with transformers

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    spec = merged(DEFAULT_SPEC, raw)
    validate_conditions(spec["conditions"])
    if not spec.get("name") or not re.fullmatch(r"[\w.-]+", spec["name"]):
        raise ValueError("The spec needs a file-name-safe 'name'")
    return spec


def steering_signature(condition: dict):
    """Hooks a condition installs: (recipe, control, dose ratio), or None."""
    if condition.get("recipe") is None:
        return None
    return (condition["recipe"], condition.get("control", "ar"), condition["dose_ratio"])


def validate_conditions(conditions: list) -> None:
    names = [c["name"] for c in conditions]
    if len(set(names)) != len(names):
        raise ValueError("Condition names must be unique")
    if "baseline" not in names:
        raise ValueError("A condition named 'baseline' is required")
    for condition in conditions:
        recipe = condition.get("recipe")
        if recipe not in (None, "centered", "raw_mean"):
            raise ValueError(f"{condition['name']}: unknown recipe {recipe!r}")
        if recipe is not None and not isinstance(condition.get("dose_ratio"), (int, float)):
            raise ValueError(f"{condition['name']}: steering needs a numeric dose_ratio")
    baseline = next(c for c in conditions if c["name"] == "baseline")
    if baseline.get("recipe") is not None or baseline.get("rag"):
        raise ValueError("The baseline must have neither steering nor retrieval")


def load_prompts(path: Path, split: str) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = [p for p in data["prompts"] if p["split"] == split]
    if not prompts:
        raise ValueError(f"No prompts in split {split!r}")
    return prompts


def tuning_texts() -> list:
    """Texts used to build, center or calibrate vectors, or in earlier experiments."""
    from steered_vs_baseline_transport import DEFAULT_PROMPTS, NEUTRAL_SENTENCES

    texts = list(calibration_texts()) + neutral_texts("ar") + neutral_texts("en")
    texts += TEST_PROMPTS + [p for group in THEMATIC_TEST_PROMPTS.values() for p in group]
    texts += DEFAULT_PROMPTS + NEUTRAL_SENTENCES
    texts += [
        "What is the meaning of life?",
        "How should I deal with a bug in my code?",
        "What is patience?",
    ]  # reproduce_paper.py
    return texts


def check_held_out(prompts: list, tuning: list) -> None:
    """Reject evaluation prompts that repeat or nearly repeat a tuning text."""
    tuning_sets = [(text, set(ev.words(text))) for text in tuning]
    clashes = []
    for prompt in prompts:
        words = set(ev.words(prompt["text"]))
        for text, other in tuning_sets:
            union = words | other
            if union and len(words & other) / len(union) >= NEAR_DUPLICATE_JACCARD:
                clashes.append(f"{prompt['id']} ~ {text!r}")
    if clashes:
        raise ValueError("Evaluation prompts overlap tuning texts: " + "; ".join(clashes))


# ---------------------------------------------------------------------------
# Provenance and external data
# ---------------------------------------------------------------------------

def git_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def provenance(steerer: QuranSteerer, spec: dict, spec_text: str) -> dict:
    import transformers

    llm = steerer.llm
    model_config = getattr(getattr(llm, "model", None), "config", None)
    template = getattr(llm.tokenizer, "chat_template", None)
    return {
        "code": git_state(),
        "spec_sha256": hashlib.sha256(spec_text.encode("utf-8")).hexdigest(),
        "model": spec["model"],
        "model_path": getattr(llm, "model_path", None),
        "model_revision": getattr(model_config, "_commit_hash", None) or spec.get("revision"),
        "chat_template_sha256": (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
        ),
        "corpus_sha256": hashlib.sha256(steerer.quran_path.read_bytes()).hexdigest(),
        "control_sha256": {lang: texts_sha256(neutral_texts(lang)) for lang in ("ar", "en")},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


ARC_ROWS = "https://datasets-server.huggingface.co/rows"


def fetch_json(url: str, attempts: int = 5) -> dict:
    """GET JSON, retrying server errors and dropped connections with backoff."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** (attempt + 1))


def fetch_arc_easy(n_items: int, seed: int, cache_dir: Path) -> tuple:
    """A seeded sample of ARC-Easy test questions, cached locally (not committed)."""
    cache = cache_dir / "arc_easy_test.json"
    if cache.exists():
        rows = json.loads(cache.read_text(encoding="utf-8"))
    else:
        rows = []
        while True:
            query = urllib.parse.urlencode({
                "dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
                "offset": len(rows), "length": 100,
            })
            page = fetch_json(f"{ARC_ROWS}?{query}")
            rows += [item["row"] for item in page["rows"]]
            if len(rows) >= page["num_rows_total"] or not page["rows"]:
                break
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows), encoding="utf-8")
    dataset_sha = fetch_json("https://huggingface.co/api/datasets/allenai/ai2_arc").get("sha")
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_items, len(rows)))
    items = [
        {
            "id": row["id"],
            "question": row["question"],
            "choices": row["choices"]["text"],
            "answer": row["choices"]["label"].index(row["answerKey"]),
        }
        for row in sample
    ]
    info = {
        "dataset": "allenai/ai2_arc", "config": "ARC-Easy", "split": "test",
        "dataset_sha": dataset_sha, "n_total": len(rows), "seed": seed,
        "scoring": "zero-shot, log-likelihood per character (acc_norm)",
    }
    return items, info


# ---------------------------------------------------------------------------
# Running conditions
# ---------------------------------------------------------------------------

def prepare_vectors(steerer: QuranSteerer, spec: dict) -> dict:
    """Steering vectors for each (recipe, control) the conditions use."""
    vectors = {}
    for condition in spec["conditions"]:
        signature = steering_signature(condition)
        if signature is None or signature[:2] in vectors:
            continue
        recipe, control, _ = signature
        prepared = steerer.prepare_quran_steering(
            chunk_by=spec["steering"]["chunk_by"],
            sample_size=spec["steering"]["sample_size"],
            recipe=recipe,
            control=control,
        )
        vectors[(recipe, control)] = {k: v.detach().clone() for k, v in prepared.items()}
    steerer.llm.clear_steering()
    return vectors


def configure(steerer: QuranSteerer, condition: dict, vectors: dict, spec: dict) -> None:
    """Install exactly the hooks a condition calls for."""
    steerer.llm.clear_steering()
    signature = steering_signature(condition)
    if signature is None:
        steerer.steering_vectors = None
        return
    recipe, control, ratio = signature
    steerer.steering_vectors = {k: v.clone() for k, v in vectors[(recipe, control)].items()}
    steerer.config = SteeringConfig(
        dose_ratio=float(ratio),
        target_layers=spec["steering"]["target_layers"],
        layer_distribution=spec["steering"]["layer_distribution"],
    )
    steerer.config.validate()
    steerer._apply_steering()


def dose_attainment(steerer: QuranSteerer) -> tuple:
    """Peak achieved dose ratio and mean achieved/target ratio across steered layers."""
    diagnostics = steerer.last_run_diagnostics
    if not diagnostics or steerer.config.dose_ratio is None:
        return None, None
    num_layers = steerer.llm.num_layers
    distribution = steerer.config.layer_distribution
    achieved, attainment = [], []
    for layer, summary in diagnostics.items():
        target = abs(steerer.config.dose_ratio) * layer_distribution_scale(
            layer, num_layers, distribution
        )
        achieved.append(summary.dose_ratio)
        if target > 0:
            attainment.append(summary.dose_ratio / target)
    return max(achieved), (sum(attainment) / len(attainment) if attainment else None)


def generate_outputs(steerer, condition, prompts, final_prompts, spec) -> list:
    records = []
    llm = steerer.llm
    for prompt in prompts:
        text = final_prompts[prompt["id"]] if condition.get("rag") else prompt["text"]
        output = steerer.generate(
            text,
            max_new_tokens=spec["decoding"]["max_new_tokens"],
            do_sample=False,
            seed=spec["seed"],
            chat_template=spec["chat_template"],
        )
        peak, attainment = dose_attainment(steerer)
        context, templated = llm.format_prompt(text, chat_template=spec["chat_template"])
        with llm.steering_disabled():
            nll = ev.mean_token_nll(llm, context, output, add_special_tokens=not templated)
        tokens = ev.words(output)
        records.append({
            "condition": condition["name"],
            "prompt_id": prompt["id"],
            "pair": prompt.get("pair"),
            "lang": prompt["lang"],
            "category": prompt["category"],
            "final_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "output": output,
            "n_words": len(tokens),
            "arabic_share": ev.arabic_share(output),
            "script": ev.dominant_script(output),
            "distinct_2": ev.distinct_n(tokens, 2),
            "degenerate": ev.degenerate(output),
            "nll_unsteered": nll,
            "dose_ratio_peak": peak,
            "dose_attainment": attainment,
        })
    return records


def score_capability(steerer, items) -> dict:
    llm = steerer.llm
    return {
        item["id"]: ev.multiple_choice_correct(llm, item["question"], item["choices"], item["answer"])
        for item in items
    }


def transport_values(steerer, prompts, settings, layers) -> dict:
    """Per-prompt mean rho and holonomy across ``layers``, under the current hooks."""
    from steered_vs_baseline_transport import transport_summary

    model = steerer.llm.model
    previous = getattr(model.config, "_attn_implementation", None)
    set_attention(model, "eager")  # only eager attention returns weights
    try:
        values = {}
        for prompt in prompts:
            diagnostics = steerer.llm.get_attention_transport_diagnostics(
                prompt["text"], layers=layers, eta=settings["eta"],
                max_loop_positions=settings["max_loop_positions"],
            )
            layers = transport_summary(diagnostics)
            rhos = [v["rho"] for v in layers.values()]
            hols = [v["holonomy"] for v in layers.values()]
            values[prompt["id"]] = {
                "rho": sum(rhos) / len(rhos) if rhos else float("nan"),
                "holonomy": sum(hols) / len(hols) if hols else float("nan"),
            }
        return values
    finally:
        if previous:
            set_attention(model, previous)


def set_attention(model, implementation: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(implementation)
    else:
        model.config._attn_implementation = implementation


def round_robin(prompts: list, n: int) -> list:
    """Up to ``n`` prompts, taking one category at a time in file order."""
    by_category = {}
    for prompt in prompts:
        by_category.setdefault(prompt["category"], []).append(prompt)
    queues = list(by_category.values())
    chosen = []
    while len(chosen) < n and any(queues):
        for queue in queues:
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def checkpoint_key(spec: dict, prompts: list, commit: str) -> dict:
    """What a checkpoint must match to be resumed: the spec, prompts and code."""
    return {
        "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        "prompt_ids": [p["id"] for p in prompts],
        "commit": commit,
    }


def load_checkpoint(path, key) -> dict:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["conditions"] if data.get("key") == key else {}


def save_checkpoint(path, key, done) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps({"key": key, "conditions": done}, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def run_evaluation(
    steerer: QuranSteerer,
    spec: dict,
    prompts: list,
    arc_items=None,
    embed=None,
    final_prompts=None,
    log=print,
    checkpoint=None,
    resume_key=None,
) -> dict:
    """Run every condition; returns records, summaries and paired differences.

    With a ``checkpoint`` path, each finished condition is saved there, and a
    later call with the same ``resume_key`` reuses them instead of rerunning.
    """
    conditions = spec["conditions"]
    metrics = spec["metrics"]
    if any(c.get("rag") for c in conditions) and final_prompts is None:
        raise ValueError("RAG conditions need final_prompts")

    vectors = prepare_vectors(steerer, spec)
    calibration = None
    if vectors:
        steerer.calibrate_dose()
        calibration = {
            "texts_sha256": steerer.dose_calibration["texts_sha256"],
            "num_texts": steerer.dose_calibration["num_texts"],
            "layer_norms": {str(k): v for k, v in steerer.dose_calibration["layer_norms"].items()},
        }

    transport_prompts, transport_layers = [], None
    if metrics.get("transport"):
        settings = metrics["transport"]
        transport_prompts = round_robin(
            [p for p in prompts if p["lang"] == settings["lang"]], settings["n_prompts"]
        )
        # The steered band, measured the same way in every condition.
        transport_layers = spec["steering"]["target_layers"] or select_target_layers(
            steerer.llm.num_layers, spec["steering"]["layer_distribution"]
        )

    records, capability, transport, timings = [], {}, {}, {}
    done = load_checkpoint(checkpoint, resume_key)
    resumed = []
    for condition in conditions:
        name = condition["name"]
        signature = steering_signature(condition)
        if name in done:
            entry = done[name]
            records += entry["records"]
            timings[name] = entry["timing_s"]
            if entry["capability"] is not None:
                capability.setdefault(signature, entry["capability"])
            if entry["transport"] is not None:
                transport.setdefault(signature, entry["transport"])
            resumed.append(name)
            log(f"[{name}] resumed from checkpoint")
            continue
        started = time.time()
        configure(steerer, condition, vectors, spec)
        new_records = generate_outputs(steerer, condition, prompts, final_prompts or {}, spec)
        records += new_records
        entry = {"records": new_records, "capability": None, "transport": None}
        # Capability and transport see only the hooks, so RAG conditions share them.
        if arc_items and signature not in capability:
            capability[signature] = entry["capability"] = score_capability(steerer, arc_items)
        if transport_prompts and signature not in transport:
            transport[signature] = entry["transport"] = transport_values(
                steerer, transport_prompts, metrics["transport"], transport_layers
            )
        timings[name] = entry["timing_s"] = round(time.time() - started, 1)
        if checkpoint is not None:
            done[name] = entry
            save_checkpoint(checkpoint, resume_key, done)
        log(f"[{name}] {timings[name]}s")
    steerer.llm.clear_steering()

    if embed is not None:
        add_thematic_proxy(steerer, records, embed, metrics["thematic_proxy"], spec["seed"])

    return {
        "calibration": calibration,
        "records": records,
        "capability": {
            c["name"]: capability.get(steering_signature(c)) for c in conditions
        } if arc_items else None,
        "transport": {
            c["name"]: transport.get(steering_signature(c)) for c in conditions
        } if transport_prompts else None,
        "transport_layers": transport_layers,
        "timings_s": timings,
        "resumed_conditions": resumed,
    }


def add_thematic_proxy(steerer, records, embed, settings, seed) -> None:
    verses = steerer.embedder.load_quran_text(steerer.quran_path, chunk_by="verse")
    rng = random.Random(seed)
    sample = rng.sample(verses, min(settings["n_verses"], len(verses)))
    positive = np.asarray(embed(sample)).mean(axis=0)
    negative = np.asarray(embed(neutral_texts("ar") + neutral_texts("en"))).mean(axis=0)
    scores = ev.centroid_contrast(np.asarray(embed([r["output"] for r in records])), positive, negative)
    for record, score in zip(records, scores):
        record["thematic_proxy"] = float(score)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def by_prompt(records, condition, key, **filters) -> dict:
    return {
        r["prompt_id"]: r[key] for r in records
        if r["condition"] == condition and all(r[k] == v for k, v in filters.items())
    }


def summarize_results(spec: dict, results: dict) -> dict:
    boot = spec["bootstrap"]
    n_boot, n_perm = boot["n_boot"], boot["n_perm"]
    records = results["records"]
    summary = {}

    def paired(values, base):
        ids = sorted(base)
        return ev.paired_difference(
            [values.get(i, float("nan")) for i in ids], [base[i] for i in ids],
            n_boot=n_boot, n_perm=n_perm,
        )

    def as_float(values):
        return {k: float(v) if v is not None else float("nan") for k, v in values.items()}

    for condition in spec["conditions"]:
        name = condition["name"]
        entry = {"condition": condition}
        for lang in ("en", "ar"):
            arabic = {
                k: float(v == "arabic")
                for k, v in by_prompt(records, name, "script", lang=lang).items()
            }
            entry[f"arabic_output_rate_{lang}"] = ev.summarize(arabic.values(), n_boot)
            neutral = {
                k: float(v == "arabic")
                for k, v in by_prompt(records, name, "script", lang=lang, category="neutral").items()
            }
            entry[f"arabic_output_rate_{lang}_neutral"] = ev.summarize(neutral.values(), n_boot)
        for key in ("distinct_2", "nll_unsteered", "thematic_proxy", "dose_ratio_peak", "dose_attainment"):
            values = as_float(by_prompt(records, name, key)) if records and key in records[0] else {}
            entry[key] = ev.summarize(values.values(), n_boot)
            if name != "baseline" and key in ("distinct_2", "nll_unsteered", "thematic_proxy"):
                base = as_float(by_prompt(records, "baseline", key)) if values else {}
                entry[f"{key}_vs_baseline"] = paired(values, base) if values else None
        degenerate = {k: float(v) for k, v in by_prompt(records, name, "degenerate").items()}
        entry["degenerate_rate"] = ev.summarize(degenerate.values(), n_boot)

        if results.get("capability"):
            correct = {k: float(v) for k, v in results["capability"][name].items()}
            entry["arc_easy_accuracy"] = ev.summarize(correct.values(), n_boot)
            if name != "baseline":
                base = {k: float(v) for k, v in results["capability"]["baseline"].items()}
                entry["arc_easy_accuracy_vs_baseline"] = paired(correct, base)
        if results.get("transport"):
            for metric in ("rho", "holonomy"):
                values = {k: v[metric] for k, v in results["transport"][name].items()}
                entry[metric] = ev.summarize(values.values(), n_boot)
                if name != "baseline":
                    base = {k: v[metric] for k, v in results["transport"]["baseline"].items()}
                    entry[f"{metric}_vs_baseline"] = paired(values, base)
        summary[name] = entry
    return summary


def interval(stat: dict, digits: int = 2, key: str = "mean") -> str:
    if not stat or stat.get("n", 0) == 0 or math.isnan(stat.get(key, float("nan"))):
        return "n/a"
    return f"{stat[key]:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def signed(stat: dict, digits: int = 3) -> str:
    if not stat or stat.get("n", 0) == 0:
        return "n/a"
    return (
        f"{stat['mean_diff']:+.{digits}f} [{stat['ci_low']:+.{digits}f}, "
        f"{stat['ci_high']:+.{digits}f}]"
    )


def render_markdown(result: dict) -> str:
    spec, summary, prov = result["spec"], result["summary"], result["provenance"]
    n_prompts = len({r["prompt_id"] for r in result["records"]})
    lines = [
        f"# Steering evaluation: {spec['name']}",
        "",
        f"Model `{spec['model']}` (revision `{prov['model_revision']}`), code "
        f"`{(prov['code']['commit'] or 'unknown')[:12]}`{' (dirty)' if prov['code']['dirty'] else ''}, "
        f"{n_prompts} held-out prompts ({spec['prompts']['split']} split), greedy decoding of "
        f"{spec['decoding']['max_new_tokens']} tokens, layer distribution "
        f"`{spec['steering']['layer_distribution']}`. Intervals are 95% bootstrap intervals "
        "over prompts (or ARC items); differences are paired with the baseline.",
        "",
        "## Language and fluency",
        "",
        "| Condition | Dose ratio (target / achieved peak) | Arabic-script outputs, English prompts "
        "| …English neutral prompts | Arabic-script outputs, Arabic prompts | Degenerate outputs "
        "| Distinct-2 | ΔNLL under unsteered model |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in summary.items():
        condition = entry["condition"]
        target = condition.get("dose_ratio")
        dose = "–" if target is None else f"{target:g} / {interval(entry['dose_ratio_peak'], 3)}"
        if condition.get("rag"):
            name = f"{name} (RAG)"
        lines.append(
            f"| {name} | {dose} | {interval(entry['arabic_output_rate_en'])} "
            f"| {interval(entry['arabic_output_rate_en_neutral'])} "
            f"| {interval(entry['arabic_output_rate_ar'])} | {interval(entry['degenerate_rate'])} "
            f"| {interval(entry['distinct_2'])} "
            f"| {signed(entry.get('nll_unsteered_vs_baseline'), 2)} |"
        )
    lines += [
        "",
        "## Capability, thematic proxy and transport",
        "",
        "| Condition | ARC-Easy accuracy | ΔARC vs baseline | Δ thematic proxy | Δρ | Δholonomy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in summary.items():
        label = f"{name} (RAG)" if entry["condition"].get("rag") else name
        lines.append(
            f"| {label} | {interval(entry.get('arc_easy_accuracy'))} "
            f"| {signed(entry.get('arc_easy_accuracy_vs_baseline'))} "
            f"| {signed(entry.get('thematic_proxy_vs_baseline'))} "
            f"| {signed(entry.get('rho_vs_baseline'), 4)} "
            f"| {signed(entry.get('holonomy_vs_baseline'), 4)} |"
        )
    lines += [
        "",
        "Arabic-script outputs: share of outputs whose letters are mostly Arabic script. "
        "The thematic proxy is an embedding contrast, not a rating; see the rating sheet. "
        "Capability and transport depend only on the hooks, so RAG conditions repeat "
        "the matching non-RAG values.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Blinded human ratings
# ---------------------------------------------------------------------------

def write_rating_sheet(result: dict, out_dir: Path) -> tuple:
    spec = result["spec"]
    settings = spec["rating_sheet"]
    rng = random.Random(spec["seed"])
    items = []
    for condition in spec["conditions"]:
        pool = [
            r for r in result["records"]
            if r["condition"] == condition["name"] and r["category"] == settings["category"]
        ]
        items += rng.sample(pool, min(settings["per_condition"], len(pool)))
    rng.shuffle(items)
    prompts = {p["id"]: p["text"] for p in result["prompts"]}
    sheet = out_dir / f"{spec['name']}_rating_sheet.csv"
    key_path = out_dir / f"{spec['name']}_rating_key.json"
    key = {"rubric_version": RUBRIC_VERSION, "items": {}}
    with sheet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "prompt", "output", "rating", "note"])
        for index, record in enumerate(items, 1):
            item_id = f"R{index:03d}"
            writer.writerow([item_id, prompts[record["prompt_id"]], record["output"], "", ""])
            key["items"][item_id] = {"condition": record["condition"], "prompt_id": record["prompt_id"]}
    key_path.write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    return sheet, key_path


def read_ratings(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["item_id"]: int(row["rating"])
            for row in csv.DictReader(handle)
            if row.get("rating", "").strip()
        }


def score_ratings(key_path: Path, rater_a: Path, rater_b: Path, n_boot: int = 2000) -> str:
    key = json.loads(key_path.read_text(encoding="utf-8"))["items"]
    a, b = read_ratings(rater_a), read_ratings(rater_b)
    shared = sorted(set(a) & set(b) & set(key))
    if not shared:
        raise ValueError("The two rating files share no rated items")
    kappa = ev.cohens_kappa([a[i] for i in shared], [b[i] for i in shared], weights="quadratic")
    lines = [
        f"Items rated by both: {len(shared)}; quadratic-weighted Cohen's kappa {kappa:.2f}",
        "",
        "| Condition | Mean rating (0–2), both raters |",
        "| --- | --- |",
    ]
    for condition in sorted({key[i]["condition"] for i in shared}):
        values = [(a[i] + b[i]) / 2 for i in shared if key[i]["condition"] == condition]
        lines.append(f"| {condition} | {interval(ev.summarize(values, n_boot))} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_rag_prompts(steerer, prompts, spec, work_dir: Path) -> dict:
    steerer.initialize_knowledge_base(persist_dir=str(work_dir / "quran_db"))
    steerer.knowledge_base.build_index()
    final = {}
    for prompt in prompts:
        final[prompt["id"]], _ = steerer._mra_context(
            prompt["text"], spec["rag"]["use_domain_bridges"]
        )
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output-dir", type=Path, default=REPO / "experiments" / "results")
    parser.add_argument("--work-dir", type=Path, default=REPO / ".eval_work",
                        help="Vector index and dataset cache (not committed)")
    parser.add_argument("--limit", type=int, help="Use only the first N prompts (smoke runs)")
    parser.add_argument("--score-ratings", nargs=3, type=Path, metavar=("KEY", "RATER_A", "RATER_B"))
    args = parser.parse_args(argv)

    if args.score_ratings:
        print(score_ratings(*args.score_ratings))
        return
    if args.spec is None:
        parser.error("--spec is required")

    spec_text = args.spec.read_text(encoding="utf-8")
    spec = load_spec(args.spec)
    prompts = load_prompts(REPO / spec["prompts"]["file"], spec["prompts"]["split"])
    check_held_out(prompts, tuning_texts())
    if args.limit:
        prompts = prompts[: args.limit]

    torch.manual_seed(spec["seed"])
    steerer = QuranSteerer(
        llm_model=spec["model"],
        embedding_model=spec["embedding_model"],
        device=spec["device"],
        llm_revision=spec.get("revision"),
    )
    steerer.load_models()
    prov = provenance(steerer, spec, spec_text)

    arc_items, arc_info = None, None
    if spec["metrics"].get("capability"):
        arc_items, arc_info = fetch_arc_easy(
            spec["metrics"]["capability"]["n_items"], spec["seed"], args.work_dir
        )
    final_prompts = None
    if any(c.get("rag") for c in spec["conditions"]):
        final_prompts = build_rag_prompts(steerer, prompts, spec, args.work_dir)

    def embed(texts):
        return steerer.embedder.create_embeddings(list(texts), show_progress=False)

    # Resume after an interruption only from committed, unchanged code.
    clean = prov["code"]["commit"] and not prov["code"]["dirty"]
    results = run_evaluation(
        steerer, spec, prompts, arc_items=arc_items,
        embed=embed if spec["metrics"].get("thematic_proxy") else None,
        final_prompts=final_prompts,
        checkpoint=args.work_dir / f"{spec['name']}.checkpoint.json" if clean else None,
        resume_key=checkpoint_key(spec, prompts, prov["code"]["commit"]),
    )
    prov["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = {
        "spec": spec,
        "provenance": prov,
        "prompts": prompts,
        "capability_items": (
            {**arc_info, "ids": [item["id"] for item in arc_items]} if arc_items else None
        ),
        **results,
    }
    result["summary"] = summarize_results(spec, result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"{spec['name']}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    table = render_markdown(result)
    (args.output_dir / f"{spec['name']}.md").write_text(table, encoding="utf-8")
    sheet, key = write_rating_sheet(result, args.output_dir)
    print(table)
    print(f"Wrote {out}, {sheet.name} and {key.name}")


if __name__ == "__main__":
    main()
