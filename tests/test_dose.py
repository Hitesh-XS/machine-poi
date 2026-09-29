"""Doses as target relative perturbations, calibrated from activation norms."""

from contextlib import contextmanager

import pytest
import torch

from machine_poi.controls import calibration_texts
from machine_poi.steerer import (
    InvalidConfigError,
    QuranSteerer,
    SteeringConfig,
    layer_distribution_scale,
)

TEXTS = ["the quran", "mercy patience water light", "sound", "time justice path the quran"]
KNOWN_NORMS = [2.0, 20.0, 200.0]


@contextmanager
def known_norms(llm, norms=KNOWN_NORMS):
    """Rescale every token at each layer's output to a fixed norm.

    Registered before any steering hook, so steering sees (and calibration
    measures) exactly these norms.
    """
    def make(norm):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = torch.nn.functional.normalize(hidden, dim=-1) * norm
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden
        return hook

    layers = llm.model.model.layers
    handles = [layer.register_forward_hook(make(n)) for layer, n in zip(layers, norms)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        llm.clear_steering()


def steerer_for(llm, **config):
    steerer = QuranSteerer(device="cpu")
    steerer.llm = llm
    torch.manual_seed(1)
    # Vectors of norm 3, so the coefficient must account for the vector norm.
    steerer.steering_vectors = {
        layer: 3 * torch.nn.functional.normalize(torch.randn(16), dim=0) for layer in range(3)
    }
    steerer.config = SteeringConfig(target_layers=[0, 1, 2], **config)
    return steerer


def test_token_norms_are_the_median_over_content_tokens(tiny_llm):
    norms = tiny_llm.layer_token_norms(TEXTS, batch_size=3)
    for layer in range(3):
        per_token = torch.cat([
            tiny_llm.extract_layer_activations(text)[layer][0][1:].norm(dim=-1)
            for text in TEXTS
        ])
        assert norms[layer] == pytest.approx(float(per_token.median()), rel=1e-4)


def test_a_massive_sink_token_barely_moves_the_median(tiny_llm):
    def spike(module, inputs, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        hidden[:, 0, :] = 1000.0
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    plain = tiny_llm.layer_token_norms(TEXTS, layers=[0], exclude_special=False)[0]
    handle = tiny_llm.model.model.layers[0].register_forward_hook(spike)
    try:
        spiked = tiny_llm.layer_token_norms(TEXTS, layers=[0], exclude_special=False)[0]
    finally:
        handle.remove()
    assert spiked < 2 * plain  # a mean would be above 1000 / 3


@pytest.mark.parametrize("ratio", [0.05, -0.05, 0.2])
def test_achieved_ratio_matches_the_target_on_a_model_with_known_norms(tiny_llm, ratio):
    steerer = steerer_for(tiny_llm, dose_ratio=ratio, layer_distribution="uniform")
    with known_norms(tiny_llm):
        norms = steerer.calibrate_dose(TEXTS)
        assert [norms[layer] for layer in range(3)] == pytest.approx(KNOWN_NORMS, rel=1e-4)
        steerer._apply_steering()
        steerer.generate("the quran", max_new_tokens=4, do_sample=False, chat_template=False)
    for layer, norm in enumerate(KNOWN_NORMS):
        diagnostics = steerer.last_run_diagnostics[layer]
        assert diagnostics.median_activation_norm == pytest.approx(norm, rel=1e-4)
        assert diagnostics.dose_ratio == pytest.approx(abs(ratio), rel=0.05)
        coefficient = steerer.last_run_settings["layer_coefficients"][layer]
        assert coefficient == pytest.approx(ratio * norm / 3, rel=1e-4)  # signed
    assert steerer.last_run_settings["dose_calibration"]["num_texts"] == len(TEXTS)


def test_layer_distribution_shapes_the_ratio(tiny_llm):
    steerer = steerer_for(tiny_llm, dose_ratio=0.1, layer_distribution="bell")
    with known_norms(tiny_llm):
        steerer.calibrate_dose(TEXTS)
        steerer._apply_steering()
        steerer.generate("the quran", max_new_tokens=2, do_sample=False, chat_template=False)
    for layer in range(3):
        expected = 0.1 * layer_distribution_scale(layer, 3, "bell")
        assert steerer.last_run_diagnostics[layer].dose_ratio == pytest.approx(expected, rel=0.05)


def test_first_application_calibrates_on_the_default_set(tiny_llm, monkeypatch):
    steerer = steerer_for(tiny_llm)  # default dose: ratio 0.05
    seen = []
    measure = tiny_llm.layer_token_norms
    monkeypatch.setattr(
        tiny_llm, "layer_token_norms", lambda texts, **kw: seen.append(texts) or measure(texts, **kw)
    )
    try:
        steerer._apply_steering()
        steerer._apply_steering()
    finally:
        tiny_llm.clear_steering()
    assert seen == [calibration_texts()]  # once, then reused
    assert len(calibration_texts()) == 20


def test_raw_coefficients_remain_available_and_switching_is_explicit(tiny_llm):
    steerer = steerer_for(tiny_llm, layer_distribution="uniform")
    steerer.dose_calibration = {"layer_norms": {0: 2.0, 1: 20.0, 2: 200.0}}
    try:
        steerer.set_steering_strength(0.3)
        assert steerer.config.dose_ratio is None
        assert tiny_llm.hooks[2].coefficient == pytest.approx(0.3)

        steerer.set_dose_ratio(-0.1)
        assert tiny_llm.hooks[2].coefficient == pytest.approx(-0.1 * 200.0 / 3)

        with pytest.raises(InvalidConfigError):
            steerer.set_dose_ratio(1.5)
        assert steerer.config.dose_ratio == -0.1
        with pytest.raises(InvalidConfigError, match="add mode"):
            SteeringConfig(injection_mode="clamp").validate()
        SteeringConfig(injection_mode="clamp", dose_ratio=None).validate()
    finally:
        tiny_llm.clear_steering()


def test_blended_dynamic_vectors_get_the_same_update_size(tiny_llm):
    steerer = steerer_for(tiny_llm, dose_ratio=0.05, layer_distribution="uniform")
    dynamic = {layer: torch.eye(16)[layer] for layer in range(3)}
    with known_norms(tiny_llm), tiny_llm.steering_session():
        steerer.calibrate_dose(TEXTS)
        steerer.apply_dynamic_steering(dynamic, blend_ratio=0.5)  # blend norm below 3
        for layer, norm in enumerate(KNOWN_NORMS):
            hook = tiny_llm.hooks[layer]
            update = hook.coefficient * float(hook.steering_vector.norm())
            assert update == pytest.approx(0.05 * norm, rel=1e-4)
