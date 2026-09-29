"""Steered-vs-baseline comparisons must differ only in steering."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from machine_poi import cli
from machine_poi.knowledge_base import QuranKnowledgeBase
from machine_poi.llm_wrapper import SteeredLLM
from machine_poi.steerer import QuranSteerer


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors, add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens, skip_special_tokens):
        return " ".join(str(token) for token in tokens.tolist())


class SamplingModel:
    """Returns random tokens, so only seeding makes two calls agree."""

    device = "cpu"

    def generate(self, input_ids, **kwargs):
        return torch.cat([input_ids, torch.randint(0, 10_000, (1, 6))], dim=1)


def fake_llm():
    llm = SteeredLLM(device="cpu")
    llm.model = SamplingModel()
    llm.tokenizer = FakeTokenizer()
    return llm


def recording_steerer():
    steerer = QuranSteerer(device="cpu")
    state = {"disabled": False}
    calls = []

    @contextmanager
    def disabled():
        state["disabled"] = True
        try:
            yield
        finally:
            state["disabled"] = False

    def generate(prompt, **options):
        calls.append({"prompt": prompt, "disabled": state["disabled"], **options})
        return "baseline" if state["disabled"] else "steered"

    steerer.llm = Mock(spec=SteeredLLM)
    steerer.llm.steering_session.side_effect = nullcontext
    steerer.llm.steering_disabled.side_effect = disabled
    steerer.llm.generate.side_effect = generate
    steerer.llm.get_steering_diagnostics.side_effect = lambda: {
        "captured_while_disabled": state["disabled"]
    }
    steerer.knowledge_base = Mock(spec=QuranKnowledgeBase)
    steerer.knowledge_base.query_multiresolution.return_value = {
        "verse": [{"content": "retrieved verse"}],
        "passage": [],
        "surah": [],
    }
    return steerer, calls


def test_compare_retrieves_once_and_gives_both_arms_the_same_prompt_and_seed():
    steerer, calls = recording_steerer()
    steered, baseline = steerer.compare(
        "What is truth?", mra_mode=True, use_domain_bridges=False, temperature=0.3
    )
    assert (steered, baseline) == ("steered", "baseline")
    assert steerer.knowledge_base.query_multiresolution.call_count == 1
    assert [call["disabled"] for call in calls] == [False, True]
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert "retrieved verse" in calls[0]["prompt"]
    assert calls[0]["seed"] == calls[1]["seed"] is not None
    assert calls[0]["temperature"] == calls[1]["temperature"] == 0.3
    assert steerer.last_run_diagnostics == {"captured_while_disabled": False}


def test_wrapper_rejects_retrieval_options_instead_of_dropping_them():
    with pytest.raises(TypeError, match="mra_mode"):
        fake_llm().generate("prompt", mra_mode=True)


def test_seeded_comparison_arms_share_sampling_noise():
    steered, unsteered = fake_llm().compare_outputs("prompt", seed=7)
    assert steered == unsteered


@pytest.mark.parametrize("mode", ["prompt", "interactive"])
def test_cli_comparisons_pass_retrieval_and_sampling_options(mode, monkeypatch):
    args = SimpleNamespace(max_tokens=12, temperature=0.2, mra=True, reasoning=False)
    steerer = Mock()
    steerer.compare.return_value = ("s", "b")
    if mode == "prompt":
        cli.run_single_prompt(steerer, "q", args)
    else:
        inputs = iter(["q", "quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli.run_interactive(steerer, args)
    steerer.compare.assert_called_once_with(
        "q", max_new_tokens=12, temperature=0.2, mra_mode=True, reasoning_mode=False
    )
    steerer.generate_unsteered.assert_not_called()
