"""Batched activation pooling and running steering statistics on a real tiny model."""

import pytest
import torch

from machine_poi.llm_wrapper import DECODER_LAYOUT, SteeredLLM

WORDS = "the quran mercy patience water light sound time justice path".split()
HIDDEN = 16


@pytest.fixture(scope="module")
def tiny_llm():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, **{w: i + 3 for i, w in enumerate(WORDS)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="[BOS]", pad_token="[PAD]", unk_token="[UNK]",
        model_input_names=["input_ids", "attention_mask"],  # like causal-LM tokenizers
    )
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(vocab), hidden_size=HIDDEN, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=None,
    )
    llm = SteeredLLM("tiny-llama", device="cpu")
    llm.model = LlamaForCausalLM(config).eval()
    llm.tokenizer = tokenizer
    llm.config = dict(DECODER_LAYOUT)
    return llm


TEXTS = [
    "the quran",
    "mercy patience water light",
    "sound",
    "time justice path the quran mercy",
    "light",
]


def test_batched_pooling_matches_one_text_at_a_time(tiny_llm):
    tiny_llm.tokenizer.padding_side = "left"
    batched = tiny_llm.pooled_layer_means(TEXTS, batch_size=3)
    assert tiny_llm.tokenizer.padding_side == "left"  # restored after right-padding
    for index, text in enumerate(TEXTS):
        single = tiny_llm.extract_layer_activations(text)
        for layer in range(3):
            torch.testing.assert_close(
                batched[layer][index], single[layer][0].mean(dim=0), atol=1e-5, rtol=1e-4
            )


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
