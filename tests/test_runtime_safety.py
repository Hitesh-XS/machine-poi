"""Regressions for leaked hooks, contaminated requests and unsafe artifacts."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from torch import nn

from machine_poi.llm_wrapper import DECODER_LAYOUT, ActivationHook, SteeredLLM
from machine_poi.knowledge_base import QuranKnowledgeBase
from machine_poi.steerer import InvalidConfigError, QuranSteerer, SteeringConfig
from machine_poi.steering_cache import load_vectors, save_vectors
from machine_poi.workspace_diagnostics import summarize_steering_hooks


def tiny_llm():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(layers=[nn.Identity()])
            self.config = SimpleNamespace(hidden_size=2, num_hidden_layers=1)
            self.device = "cpu"
            self.fail = False

        def forward(self, **kwargs):
            result = self.model.layers[0](torch.tensor([[[2.0, 3.0]]]))
            if self.fail:
                raise RuntimeError("simulated model failure")
            return result

    llm = SteeredLLM(device="cpu")
    llm.model = Tiny()
    llm.config = DECODER_LAYOUT
    llm.tokenizer = lambda text, return_tensors: {"input_ids": torch.tensor([[1]])}
    return llm


def test_duplicate_hooks_replaced_and_nested_disable_restored():
    llm = tiny_llm()
    for _ in range(3):
        llm.register_steering_hook(0, torch.tensor([1.0, 0.0]))
    assert len(llm.model.model.layers[0]._forward_hooks) == 1
    assert torch.equal(llm.model(), torch.tensor([[[3.0, 3.0]]]))
    llm.disable_steering()
    with llm.steering_disabled():
        with llm.steering_disabled():
            pass
        assert not llm.hooks[0].enabled
    assert not llm.hooks[0].enabled


def test_exception_restores_exact_session_and_clears_capture_hooks():
    llm = tiny_llm()
    llm.register_steering_hook(0, torch.tensor([1.0, 0.0]), coefficient=0.2).disable()
    with pytest.raises(RuntimeError):
        with llm.steering_session():
            llm.register_steering_hook(0, torch.tensor([0.0, 1.0]), coefficient=0.8)
            llm.model.fail = True
            llm.extract_layer_activations("sample")
    assert len(llm.model.model.layers[0]._forward_hooks) == 1
    assert llm.hooks[0].coefficient == 0.2
    assert not llm.hooks[0].enabled
    assert llm.hooks[0].captured_activation is None


def test_activation_extraction_is_unsteered():
    llm = tiny_llm()
    llm.register_steering_hook(0, torch.tensor([9.0, 9.0]))
    assert torch.equal(
        llm.extract_layer_activations("sample")[0], torch.tensor([[[2.0, 3.0]]])
    )
    assert llm.hooks[0].enabled


def test_concurrent_sessions_do_not_overlap():
    llm = tiny_llm()
    entered, release, attempted, second_entered = Event(), Event(), Event(), Event()

    def first():
        with llm.steering_session():
            llm.register_steering_hook(0, torch.tensor([9.0, 9.0]))
            entered.set()
            assert release.wait(3)

    def second():
        attempted.set()
        with llm.steering_session():
            second_entered.set()
            assert not llm.hooks

    with ThreadPoolExecutor(2) as pool:
        one = pool.submit(first)
        assert entered.wait(3)
        two = pool.submit(second)
        assert attempted.wait(3)
        assert not second_entered.wait(0.05)
        release.set()
        one.result(timeout=3)
        two.result(timeout=3)


@pytest.mark.parametrize("mode", ["add", "blend", "clamp", "replace"])
def test_diagnostics_measure_actual_mode_delta(mode):
    hidden = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    hook = ActivationHook(0, torch.tensor([1.0, 0.0]), 0.2, mode)
    output = hook(None, (), hidden)
    expected = (output - hidden).norm(dim=-1).mean() / hidden.norm(dim=-1).mean()
    assert summarize_steering_hooks({0: hook})[
        0
    ].relative_perturbation == pytest.approx(expected.item())


def test_high_level_clamp_dose_and_repeated_preparation():
    steerer = QuranSteerer(device="cpu")
    steerer.llm = tiny_llm()
    steerer.steering_vectors = {0: torch.tensor([1.0, 0.0])}
    steerer.config = SteeringConfig(
        target_layers=[0], layer_distribution="uniform", injection_mode="clamp"
    )
    steerer.set_steering_strength(0.2)
    first = steerer.llm.model()[0, 0, 0].item()
    steerer.set_steering_strength(0.8)
    second = steerer.llm.model()[0, 0, 0].item()
    assert first == pytest.approx(0.2)
    assert second == pytest.approx(0.8)
    steerer._apply_steering()
    assert len(steerer.llm.model.model.layers[0]._forward_hooks) == 1


@pytest.mark.parametrize("graph", [False, True])
def test_dynamic_generation_failure_restores_baseline(graph):
    import asyncio

    steerer = QuranSteerer(device="cpu")
    steerer.llm = tiny_llm()
    steerer.llm.register_steering_hook(0, torch.tensor([1.0, 0.0]), 0.4).disable()
    results = {"verse": [{"content": "reference"}], "passage": [], "surah": []}
    steerer.compute_dynamic_steering = Mock(return_value={0: torch.tensor([0.0, 1.0])})
    steerer.config.target_layers = [0]
    steerer.knowledge_base = Mock(spec=QuranKnowledgeBase)
    steerer.knowledge_base.query_multiresolution.return_value = results

    async def query(**kwargs):
        return SimpleNamespace(
            vector_results=results, graph_answer="context", bridges=[]
        )

    steerer.hybrid_kb = SimpleNamespace(query=query)
    with patch.object(steerer.llm, "generate", side_effect=RuntimeError("fail")):
        with pytest.raises(RuntimeError):
            if graph:
                asyncio.run(
                    steerer.generate_with_graph(
                        "q", use_dynamic_steering=True, trusted_retrieval=True
                    )
                )
            else:
                steerer.generate(
                    "q",
                    mra_mode=True,
                    use_domain_bridges=False,
                    use_dynamic_steering=True,
                    trusted_retrieval=True,
                )
    assert not steerer.llm.hooks[0].enabled
    assert steerer.llm.hooks[0].coefficient == 0.4


def test_dynamic_steering_requires_explicit_trust():
    steerer = QuranSteerer(device="cpu")
    with pytest.raises(InvalidConfigError):
        steerer.generate("q", mra_mode=True, use_dynamic_steering=True)


def metadata():
    return {
        "format": 1,
        "model": "toy",
        "revision": "revision-a",
        "hidden_size": 2,
        "num_layers": 2,
    }


def test_numeric_cache_roundtrip_and_identity(tmp_path):
    path = tmp_path / "cache.npz"
    save_vectors(path, {0: np.ones(2)}, metadata())
    assert np.array_equal(load_vectors(path, metadata())[0], np.ones(2))
    with pytest.raises(ValueError):
        load_vectors(path, {**metadata(), "revision": "revision-b"})
    with pytest.raises(ValueError):
        save_vectors(path, {0: np.array([float("nan"), 1])}, metadata())


def test_pickle_cache_rejected_without_executing(tmp_path):
    import json

    path = tmp_path / "cache.npz"
    np.savez(
        path,
        metadata=np.array(json.dumps(metadata())),
        layer_0=np.array([{}, {}], dtype=object),
    )
    with pytest.raises(ValueError, match="Object arrays"):
        load_vectors(path, metadata())


def test_unsafe_model_code_requires_pin_and_is_off_by_default():
    llm = tiny_llm()
    with patch(
        "machine_poi.llm_wrapper.AutoModelForCausalLM.from_pretrained", return_value=llm.model
    ) as model:
        with patch("machine_poi.llm_wrapper.AutoTokenizer.from_pretrained") as tokenizer:
            llm.load_model()
    assert model.call_args.kwargs["trust_remote_code"] is False
    assert tokenizer.call_args.kwargs["trust_remote_code"] is False
    with pytest.raises(ValueError):
        SteeredLLM(trust_remote_code=True, revision="main")


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf")])
def test_nonfinite_hook_strength_rejected(coefficient):
    with pytest.raises(ValueError):
        ActivationHook(0, torch.ones(2), coefficient)
