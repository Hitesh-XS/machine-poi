"""Model aliases have one source of truth: machine_poi.config."""

from machine_poi import cli
from machine_poi.config import EMBEDDING_MODELS, LLM_MODELS, get_recommended_config
from machine_poi.llm_wrapper import SteeredLLM
from machine_poi.quran_embeddings import QuranEmbeddings


def test_wrapper_aliases_and_reasoning_come_from_the_registry():
    assert SteeredLLM.SUPPORTED_MODELS == {
        alias: spec["hf_path"] for alias, spec in LLM_MODELS.items()
    }
    assert SteeredLLM.REASONING_CONFIGS == {
        alias: spec["reasoning"] for alias, spec in LLM_MODELS.items() if "reasoning" in spec
    }


def test_registry_does_not_hardcode_model_dimensions():
    for spec in LLM_MODELS.values():
        assert not {"hidden_size", "num_layers"} & spec.keys()


def test_embedding_aliases_come_from_the_registry():
    assert QuranEmbeddings.SUPPORTED_MODELS == {
        alias: spec["hf_path"] for alias, spec in EMBEDDING_MODELS.items()
    }


def test_gemma_4_aliases_resolve_and_use_the_preset_dose(monkeypatch):
    assert SteeredLLM("gemma-4-e2b").model_path == "google/gemma-4-E2B-it"
    assert SteeredLLM("gemma-4-e4b").model_path == "google/gemma-4-E4B-it"
    monkeypatch.setattr("sys.argv", ["machine-poi", "--llm", "gemma-4-e2b"])
    args = cli.parse_args()
    config = get_recommended_config(args.llm, args.embedding, intensity=args.preset)
    steering, _ = cli.resolve_steering(args, config)
    assert config.custom_coefficient is None and steering.target_layers is None
    assert steering.coefficient == config.get_preset().coefficient
