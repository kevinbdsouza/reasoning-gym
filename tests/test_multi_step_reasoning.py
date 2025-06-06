import importlib.util
import sys
import types
import pytest

# Set up minimal package structure with coaching module
from pathlib import Path

pkg = types.ModuleType("reasoning_gym")
pkg.__path__ = [str(Path("reasoning_gym").resolve())]
logic_pkg = types.ModuleType("reasoning_gym.logic")
logic_pkg.__path__ = []
sys.modules["reasoning_gym"] = pkg
sys.modules["reasoning_gym.logic"] = logic_pkg

coach_spec = importlib.util.spec_from_file_location(
    "reasoning_gym.coaching", "reasoning_gym/coaching/__init__.py"
)
coach_module = importlib.util.module_from_spec(coach_spec)
coach_spec.loader.exec_module(coach_module)
sys.modules["reasoning_gym.coaching"] = coach_module

spec = importlib.util.spec_from_file_location(
    "reasoning_gym.logic.multi_step_reasoning",
    "reasoning_gym/logic/multi_step_reasoning.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

MultiStepReasoningConfig = module.MultiStepReasoningConfig
MultiStepReasoningCurriculum = module.MultiStepReasoningCurriculum
MultiStepReasoningDataset = module.MultiStepReasoningDataset


def test_multi_step_config_validation():
    with pytest.raises(AssertionError):
        MultiStepReasoningConfig(min_steps=4).validate()
    with pytest.raises(AssertionError):
        MultiStepReasoningConfig(min_steps=8, max_steps=7).validate()


def test_multi_step_deterministic():
    cfg = MultiStepReasoningConfig(seed=42, size=5, min_steps=5, max_steps=5)
    d1 = MultiStepReasoningDataset(cfg)
    d2 = MultiStepReasoningDataset(cfg)
    for i in range(len(d1)):
        assert d1[i] == d2[i]


def test_multi_step_items():
    cfg = MultiStepReasoningConfig(seed=1, size=3, min_steps=5, max_steps=5)
    dataset = MultiStepReasoningDataset(cfg)
    for item in dataset:
        assert isinstance(item, dict)
        assert "question" in item
        assert "answer" in item
        assert "metadata" in item
        lines = item["question"].split("\n")
        assert len(lines) == cfg.max_steps
        for i, line in enumerate(lines, start=1):
            assert line.startswith(f"Step {i}:")
        assert dataset.score_answer(item["answer"], item) == 1.0
        assert item["metadata"]["source_dataset"] == "multi_step_reasoning"


def test_multi_step_curriculum():
    curriculum = MultiStepReasoningCurriculum()
    base = {"size": 10, "seed": 123}
    cfg = curriculum.generate_configuration(base)
    assert cfg.size == 10 and cfg.seed == 123
    assert cfg.max_steps == 5
    curriculum.increment_attr_level("num_steps")
    cfg_inc = curriculum.generate_configuration(base)
    assert cfg_inc.max_steps == 6
    curriculum.decrement_attr_level("num_steps")
    cfg_dec = curriculum.generate_configuration(base)
    assert cfg_dec.max_steps == 5
