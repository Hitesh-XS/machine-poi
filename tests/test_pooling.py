"""Batched activation pooling and running steering statistics on a real tiny model."""

import pytest
import torch

HIDDEN = 16  # the tiny_llm fixture in conftest.py


TEXTS = [
    "the quran",
    "mercy patience water light",
    "sound",
    "time justice path the quran mercy",
    "light",
]


def test_batched_pooling_matches_one_text_at_a_time(tiny_llm):
    tiny_llm.tokenizer.padding_side = "left"
    every_token = tiny_llm.pooled_layer_means(TEXTS, batch_size=3, exclude_special=False)
    content = tiny_llm.pooled_layer_means(TEXTS, batch_size=3)
    assert tiny_llm.tokenizer.padding_side == "left"  # restored after right-padding
    for index, text in enumerate(TEXTS):
        single = tiny_llm.extract_layer_activations(text)
        for layer in range(3):
            hidden = single[layer][0]
            torch.testing.assert_close(
                every_token[layer][index], hidden.mean(dim=0), atol=1e-5, rtol=1e-4
            )
            # Position 0 is [BOS]; the default mean covers content tokens only.
            torch.testing.assert_close(
                content[layer][index], hidden[1:].mean(dim=0), atol=1e-5, rtol=1e-4
            )


def test_high_norm_bos_no_longer_dominates_the_mean(tiny_llm):
    def spike(module, inputs, output):
        hidden = (output[0] if isinstance(output, tuple) else output).clone()
        hidden[:, 0, :] = 1000.0  # an attention-sink-sized activation at [BOS]
        return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

    handle = tiny_llm.model.model.layers[0].register_forward_hook(spike)
    try:
        content = tiny_llm.pooled_layer_means(TEXTS, layers=[0])[0]
        every_token = tiny_llm.pooled_layer_means(TEXTS, layers=[0], exclude_special=False)[0]
    finally:
        handle.remove()
    assert content.abs().max() < 50
    assert every_token.mean() > 100


def test_pooling_is_unsteered_and_leaves_no_hooks(tiny_llm):
    tiny_llm.register_steering_hook(1, torch.ones(HIDDEN), coefficient=5.0)
    try:
        steered_free = tiny_llm.pooled_layer_means(TEXTS[:2], layers=[1, 2])
        assert set(steered_free) == {1, 2}
        layers = tiny_llm.model.model.layers
        assert [len(layer._forward_hooks) for layer in layers] == [0, 1, 0]
        tiny_llm.clear_steering()
        torch.testing.assert_close(
            steered_free[2], tiny_llm.pooled_layer_means(TEXTS[:2], layers=[2])[2]
        )
    finally:
        tiny_llm.clear_steering()


def test_diagnostics_cover_every_generated_token_of_the_latest_call(tiny_llm):
    tiny_llm.register_steering_hook(1, torch.ones(HIDDEN), coefficient=0.5)
    try:
        for _ in range(2):  # a second call starts fresh statistics
            tiny_llm.generate("the quran", max_new_tokens=4, do_sample=False, chat_template=False)
            stats = tiny_llm.hooks[1].stats
            assert stats.tokens == 3 + 3  # 3-token prefill, then 3 single-token decode steps
        diagnostics = tiny_llm.get_steering_diagnostics()[1]
        assert tiny_llm.hooks[1].captured_activation is None
        # add mode: every token moves by 0.5 * |ones(16)| = 2.0
        assert diagnostics.relative_perturbation == pytest.approx(
            2.0 / diagnostics.activation_norm, rel=1e-5
        )
    finally:
        tiny_llm.clear_steering()
