"""CLI settings must match what the user asked for."""

import sys
from unittest.mock import Mock

import pytest

from machine_poi import cli
from machine_poi.config import LLM_MODELS, STEERING_PRESETS, get_recommended_config


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    return cli.parse_args()


def resolve(monkeypatch, *argv):
    args = parse(monkeypatch, *argv)
    config = get_recommended_config(args.llm, args.embedding, intensity=args.preset)
    return cli.resolve_steering(args, config)


MODEL = "qwen2.5-0.5b"


def test_model_recommendations_apply_without_a_preset(monkeypatch):
    steering, chunk_by = resolve(monkeypatch, "--llm", MODEL)
    assert steering.coefficient == LLM_MODELS[MODEL]["recommended_coefficient"]
    assert steering.target_layers == LLM_MODELS[MODEL]["recommended_layers"]
    assert steering.injection_mode == STEERING_PRESETS["moderate"].injection_mode
    assert chunk_by == "verse"


@pytest.mark.parametrize("preset", ["gentle", "strong"])
def test_explicit_preset_overrides_model_recommendations(monkeypatch, preset):
    steering, chunk_by = resolve(monkeypatch, "--llm", MODEL, "--preset", preset)
    expected = STEERING_PRESETS[preset]
    assert steering.coefficient == expected.coefficient
    assert steering.injection_mode == expected.injection_mode
    assert steering.layer_distribution == expected.layer_distribution
    assert steering.target_layers is None
    assert chunk_by == expected.chunk_by


def test_zero_coefficient_and_explicit_flags_win(monkeypatch):
    steering, chunk_by = resolve(
        monkeypatch,
        "--llm", MODEL,
        "--preset", "strong",
        "--coefficient", "0",
        "--layer-distribution", "workspace",
        "--chunk-by", "surah",
    )
    assert steering.coefficient == 0.0
    assert steering.layer_distribution == "workspace"
    assert steering.target_layers is None
    assert chunk_by == "surah"


def run_main(monkeypatch, *argv):
    steerer = Mock()
    steerer.compare.return_value = ("steered", "baseline")
    factory = Mock(return_value=steerer)
    monkeypatch.setattr(cli, "QuranSteerer", factory)
    monkeypatch.setattr(sys, "argv", ["main.py", *argv])
    cli.main()
    return factory, steerer


def test_main_applies_zero_coefficient_before_preparing_vectors(monkeypatch):
    _, steerer = run_main(monkeypatch, "--llm", MODEL, "--coefficient", "0", "--prompt", "q")
    assert steerer.config.coefficient == 0.0
    steerer.prepare_quran_steering.assert_called_once()
    assert steerer.prepare_quran_steering.call_args.kwargs["chunk_by"] == "verse"
    steerer.compare.assert_called_once()


def test_theme_skips_discarded_mean_vector_pass(monkeypatch):
    _, steerer = run_main(monkeypatch, "--llm", MODEL, "--theme", "mercy", "--prompt", "q")
    steerer.prepare_thematic_steering.assert_called_once_with("mercy")
    steerer.prepare_quran_steering.assert_not_called()


def test_invalid_coefficient_fails_before_loading_models(monkeypatch):
    with pytest.raises(SystemExit, match="Invalid steering configuration"):
        run_main(monkeypatch, "--llm", MODEL, "--coefficient", "5", "--prompt", "q")
    assert cli.QuranSteerer.call_count == 0
