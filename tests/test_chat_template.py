"""Instruct checkpoints get their chat template; raw prompts stay available."""

import pytest
import torch

from machine_poi.llm_wrapper import SteeredLLM


class TemplateTokenizer:
    pad_token_id = 0

    def __init__(self, template="{{ messages }}", suffix="<|assistant|>\n"):
        self.chat_template = template
        self.suffix = suffix
        self.template_calls = []
        self.encoded = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **options):
        self.template_calls.append(options)
        return f"<bos><|user|>{messages[0]['content']}{self.suffix}"

    def __call__(self, text, return_tensors, add_special_tokens=True):
        self.encoded.append((text, add_special_tokens))
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens, skip_special_tokens):
        return "reply"


class EchoModel:
    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[3]])], dim=1)


def llm_with(tokenizer, alias="smollm2-135m"):
    llm = SteeredLLM(alias, device="cpu")
    llm.model = EchoModel()
    llm.tokenizer = tokenizer
    return llm


def test_template_applied_by_default_without_duplicate_special_tokens():
    tokenizer = TemplateTokenizer()
    assert llm_with(tokenizer).generate("What is mercy?") == "reply"
    assert tokenizer.encoded == [("<bos><|user|>What is mercy?<|assistant|>\n", False)]


def test_raw_prompt_when_disabled_or_no_template():
    tokenizer = TemplateTokenizer()
    llm_with(tokenizer).generate("raw", chat_template=False)
    assert tokenizer.encoded == [("raw", True)] and not tokenizer.template_calls

    untemplated = TemplateTokenizer(template=None)
    llm_with(untemplated).generate("raw")
    assert untemplated.encoded == [("raw", True)]
    with pytest.raises(ValueError, match="no chat template"):
        llm_with(untemplated).generate("raw", chat_template=True)


@pytest.mark.parametrize("reasoning_mode", [False, True])
def test_qwen3_thinking_follows_reasoning_mode(reasoning_mode):
    tokenizer = TemplateTokenizer()
    llm_with(tokenizer, "qwen3-0.6b").generate("q", reasoning_mode=reasoning_mode)
    assert tokenizer.template_calls == [{"enable_thinking": reasoning_mode}]


@pytest.mark.parametrize("suffix", ["<|assistant|>\n", "<|assistant|><think>\n"])
def test_deepseek_reasoning_opens_the_assistant_turn_with_think_once(suffix):
    tokenizer = TemplateTokenizer(suffix=suffix)
    llm_with(tokenizer, "deepseek-r1-1.5b").generate("q", reasoning_mode=True)
    text = tokenizer.encoded[0][0]
    assert text.startswith("<bos><|user|>q") and text.endswith("<think>\n")
    assert text.count("<think>") == 1
