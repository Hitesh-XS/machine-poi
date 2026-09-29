"""The evaluation harness end to end on the tiny model, without downloads."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from machine_poi.quran_embeddings import QuranEmbeddings
from machine_poi.steerer import QuranSteerer

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("steering_eval", EXPERIMENTS / "steering_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = [
    {"id": "en-neutral-01", "pair": "neutral-01", "lang": "en", "category": "neutral",
     "split": "test", "text": "the water light"},
    {"id": "en-value-01", "pair": "value-01", "lang": "en", "category": "value",
     "split": "test", "text": "mercy patience justice"},
    {"id": "ar-value-01", "pair": "value-01", "lang": "ar", "category": "value",
     "split": "test", "text": "الصبر والرحمة"},
]
ARC = [
    {"id": "q1", "question": "the quran", "choices": ["mercy", "sound time"], "answer": 0},
    {"id": "q2", "question": "water", "choices": ["light", "path"], "answer": 1},
]


def embed(texts):
    """Deterministic stand-in for a sentence embedder."""
    rows = []
    for text in texts:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        rows.append(np.random.default_rng(seed).normal(size=8))
    return np.array(rows)


def make_spec(harness, **overrides):
    raw = {
        "name": "tiny",
        "decoding": {"max_new_tokens": 4},
        "steering": {"sample_size": 4, "layer_distribution": "uniform"},
        "metrics": {
            "capability": {"n_items": 2},
            "transport": {"n_prompts": 1, "lang": "en", "eta": 1.0, "max_loop_positions": 4},
            "thematic_proxy": {"n_verses": 5},
        },
        "rating_sheet": {"per_condition": 1, "category": "value"},
        "bootstrap": {"n_boot": 200, "n_perm": 200},
        "conditions": [
            {"name": "baseline"},
            {"name": "raw_mean_r0.05", "recipe": "raw_mean", "dose_ratio": 0.05},
            {"name": "centered_r0.1", "recipe": "centered", "dose_ratio": 0.1},
            {"name": "centered_en_r0.1", "recipe": "centered", "control": "en", "dose_ratio": 0.1},
            {"name": "rag_centered_r0.1", "recipe": "centered", "dose_ratio": 0.1, "rag": True},
        ],
        **overrides,
    }
    spec = harness.merged(harness.DEFAULT_SPEC, raw)
    harness.validate_conditions(spec["conditions"])
    return spec


@pytest.fixture
def steerer(tiny_llm):
    steerer = QuranSteerer(device="cpu")
    steerer.llm = tiny_llm
    steerer.embedder = Mock(spec=QuranEmbeddings)
    steerer.embedder.load_quran_text.return_value = [
        "the quran mercy", "patience water", "light sound time", "justice path", "the path",
    ]
    yield steerer
    tiny_llm.clear_steering()


def test_harness_runs_every_condition_and_summarizes(harness, steerer, tmp_path):
    spec = make_spec(harness)
    final_prompts = {p["id"]: f"the quran {p['text']}" for p in PROMPTS}
    with pytest.warns(UserWarning, match="raw_mean"):
        results = harness.run_evaluation(
            steerer, spec, PROMPTS, arc_items=ARC, embed=embed,
            final_prompts=final_prompts, log=lambda *_: None,
        )
    records = results["records"]
    assert len(records) == len(spec["conditions"]) * len(PROMPTS)
    assert {r["condition"] for r in records} == {c["name"] for c in spec["conditions"]}
    assert all("thematic_proxy" in r and not np.isnan(r["nll_unsteered"]) for r in records)
    baseline = [r for r in records if r["condition"] == "baseline"]
    assert all(r["dose_ratio_peak"] is None for r in baseline)
    steered = [r for r in records if r["condition"] == "centered_r0.1"]
    assert all(r["dose_attainment"] > 0 for r in steered)
    # RAG conditions generate from the retrieval prompt but share the hooks' scores.
    rag = [r for r in records if r["condition"] == "rag_centered_r0.1"]
    assert rag[0]["final_prompt_sha256"] != steered[0]["final_prompt_sha256"]
    assert results["capability"]["rag_centered_r0.1"] == results["capability"]["centered_r0.1"]
    assert set(results["transport"]["baseline"]) == {"en-neutral-01"}
    assert results["calibration"]["num_texts"] == 20

    result = {
        "spec": spec, "prompts": PROMPTS, **results,
        "provenance": {"code": harness.git_state(), "model_revision": None},
    }
    summary = harness.summarize_results(spec, result)
    assert summary["baseline"]["arabic_output_rate_en"]["n"] == 2
    assert summary["centered_r0.1"]["arc_easy_accuracy"]["n"] == 2
    assert summary["centered_r0.1"]["nll_unsteered_vs_baseline"]["n"] == 3
    assert "rho_vs_baseline" in summary["centered_en_r0.1"]
    result["summary"] = summary
    table = harness.render_markdown(result)
    assert "| centered_r0.1 | 0.1 / " in table and "rag_centered_r0.1 (RAG)" in table

    sheet, key = harness.write_rating_sheet(result, tmp_path)
    with sheet.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(spec["conditions"])  # one value-prompt output each
    assert "condition" not in rows[0]  # blinded


def test_transport_prompts_rotate_through_categories(harness):
    prompts = [
        {"id": f"{category}-{i}", "category": category}
        for category in ("neutral", "value", "technical") for i in range(3)
    ]
    chosen = [p["id"] for p in harness.round_robin(prompts, 5)]
    assert chosen == ["neutral-0", "value-0", "technical-0", "neutral-1", "value-1"]
    assert len(harness.round_robin(prompts, 20)) == 9


def test_downloads_retry_server_errors_only(harness, monkeypatch):
    import io
    import urllib.error

    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 502, "Bad Gateway", {}, None)
        return io.StringIO('{"ok": true}')

    monkeypatch.setattr(harness.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(harness.time, "sleep", lambda seconds: None)
    assert harness.fetch_json("https://example.test") == {"ok": True}
    assert len(calls) == 3

    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(harness.urllib.request, "urlopen", not_found)
    with pytest.raises(urllib.error.HTTPError):
        harness.fetch_json("https://example.test")


def test_ratings_are_scored_with_agreement(harness, tmp_path):
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"items": {
        "R001": {"condition": "baseline"}, "R002": {"condition": "centered"},
        "R003": {"condition": "centered"}, "R004": {"condition": "baseline"},
    }}))
    for name, ratings in (("a.csv", [0, 2, 1, 0]), ("b.csv", [0, 2, 2, 1])):
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["item_id", "prompt", "output", "rating", "note"])
            for index, rating in enumerate(ratings, 1):
                writer.writerow([f"R{index:03d}", "p", "o", rating, ""])
    report = harness.score_ratings(key, tmp_path / "a.csv", tmp_path / "b.csv", n_boot=200)
    assert "Items rated by both: 4" in report and "kappa" in report
    assert "| centered | 1.75 [" in report


def test_specs_are_validated(harness, tmp_path):
    with pytest.raises(ValueError, match="baseline"):
        harness.validate_conditions([{"name": "centered", "recipe": "centered", "dose_ratio": 0.1}])
    with pytest.raises(ValueError, match="dose_ratio"):
        harness.validate_conditions([{"name": "baseline"}, {"name": "x", "recipe": "centered"}])
    with pytest.raises(ValueError, match="neither"):
        harness.validate_conditions([{"name": "baseline", "rag": True}])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"name": "bad name!", "conditions": [{"name": "baseline"}]}))
    with pytest.raises(ValueError, match="name"):
        harness.load_spec(spec)
    committed = sorted((EXPERIMENTS / "specs").glob("*.json"))
    assert committed and all(harness.load_spec(path)["name"] for path in committed)
