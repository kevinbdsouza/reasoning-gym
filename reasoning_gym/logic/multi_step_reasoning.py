from dataclasses import dataclass
from random import Random
from typing import Any, Optional

import numpy as np

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..factory import ProceduralDataset, register_dataset

DATASET_NAME = "multi_step_reasoning"


@dataclass
class MultiStepReasoningConfig:
    """Configuration for the multi step logic dataset."""

    min_steps: int = 5
    max_steps: int = 10
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        assert 5 <= self.min_steps <= self.max_steps <= 10, "steps must be between 5 and 10"


class MultiStepReasoningDataset(ProceduralDataset):
    """Dataset generating multi-step puzzles mixing deduction, induction, abduction and transduction."""

    WORD_BANK = [
        "lion",
        "tiger",
        "bear",
        "wolf",
        "eagle",
        "shark",
        "horse",
        "whale",
        "otter",
        "camel",
    ]

    NAME_BANK = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Heidi",
        "Ivan",
        "Judy",
    ]

    def __init__(self, config: MultiStepReasoningConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = Random(self.seed + idx if self.seed is not None else None)
        return self._generate_item(rng, idx)

    def _deduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Make a deterministic reasoning step that requires deduction."""
        choice = rng.random()
        if choice < 0.33 or state.get("word") is None:
            # make the arithmetic harder with larger numbers and subtraction
            mult = rng.randint(10, 20)
            sub = rng.randint(5, 15)
            res = state["num"] * mult - sub
            line = f"Step {step_no}: Multiply {state['num']} by {mult} and then subtract {sub}. " "What do you get?"
            state["num"] = res
        elif choice < 0.66:
            short_thres = rng.randint(3, 4)
            long_thres = rng.randint(7, 9)
            if len(state["word"]) < short_thres:
                classification = "short"
            elif len(state["word"]) > long_thres:
                classification = "long"
            else:
                classification = "medium"
            line = (
                f"Step {step_no}: Words shorter than {short_thres} letters are 'short', "
                f"between {short_thres} and {long_thres} letters are 'medium', otherwise 'long'. "
                f"Is '{state['word']}' short, medium, or long?"
            )
            state["word"] = classification
        else:
            # introduce a simple family relationship puzzle
            a, b, c = rng.sample(self.NAME_BANK, 3)
            line = f"Step {step_no}: {a} is {b}'s parent and {b} is {c}'s parent. " f"Who is {c}'s grandparent?"
            state["person"] = a
        return line

    def _induction(self, step_no: int, rng: Random, state: dict) -> str:
        """Make an inductive reasoning step with slightly harder patterns."""
        choice = rng.random()
        if choice < 0.33:
            inc1 = rng.randint(2, 5)
            inc2 = rng.randint(1, inc1)
            n = rng.randint(3, 6)
            value = state["num"]
            for i in range(n):
                if i % 2 == 0:
                    value += inc1
                else:
                    value -= inc2
            line = (
                f"Step {step_no}: Starting at {state['num']}, alternate adding {inc1} and subtracting {inc2} "
                f"for {n} steps. What number results?"
            )
            state["num"] = value
        elif choice < 0.66:
            reversed_word = state["word"][::-1] + state["word"][0]
            line = (
                f"Step {step_no}: Reverse '{state['word']}' and append its first letter at the end. "
                "What word results?"
            )
            state["word"] = reversed_word
        else:
            start = state.get("person", rng.choice(self.NAME_BANK))
            step = rng.randint(1, 3)
            n = rng.randint(2, 4)
            idx = self.NAME_BANK.index(start)
            target = self.NAME_BANK[(idx + step * n) % len(self.NAME_BANK)]
            line = (
                f"Step {step_no}: Starting from {start}, move forward {step} names {n} times in {self.NAME_BANK}. "
                f"Which name do you reach?"
            )
            state["person"] = target
        return line

    def _abduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Generate an abductive reasoning step with trickier inference."""
        choice = rng.random()
        if choice < 0.33:
            # harder numeric abduction using a square
            secret = rng.randint(2, 9)
            add = state["num"] - secret**2
            wrong = max(2, secret + rng.randint(1, 3))
            options = [secret, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: The number {state['num']} was made by squaring a secret number "
                f"and adding {add}. Was that number {options[0]} or {options[1]}?"
            )
            state["num"] = secret
        elif choice < 0.66:
            shift = rng.choice([1, 2])
            orig = rng.choice(self.WORD_BANK)
            encoded = "".join(chr(((ord(c) - 97 + shift) % 26) + 97) for c in orig[::-1])
            wrong = rng.choice([w for w in self.WORD_BANK if w != orig])
            options = [orig, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: A secret word was reversed and each letter shifted forward by {shift} to become '{encoded}'. "
                f"Was the original word '{options[0]}' or '{options[1]}'?"
            )
            state["word"] = orig
        else:
            a, b, c = rng.sample(self.NAME_BANK, 3)
            line = f"Step {step_no}: {a} and {b} are siblings. {b} is {c}'s parent. " f"Who is {a} to {c}?"
            state["person"] = a
        return line

    def _transduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Perform a transduction step mixing representations."""
        choice = rng.random()
        if choice < 0.33:
            if rng.random() < 0.5:
                base3 = np.base_repr(state["num"], base=3)
                res = base3.count("2")
                line = f"Step {step_no}: Write {state['num']} in base 3. How many digits '2' appear?"
            else:
                bin_str = bin(state["num"])[2:]
                rotated = bin_str[1:] + bin_str[:1]
                res = int(rotated, 2)
                line = (
                    f"Step {step_no}: Write {state['num']} in binary and rotate the digits left by one. "
                    "What is the decimal value of the result?"
                )
            state["num"] = res
        elif choice < 0.66:
            index = (state["num"] ** 2) % len(self.WORD_BANK)
            res_word = self.WORD_BANK[index]
            line = (
                f"Step {step_no}: Square {state['num']} and use it as an index into {self.WORD_BANK}. "
                f"Which word do you get?"
            )
            state["word"] = res_word
        else:
            person = state.get("person", rng.choice(self.NAME_BANK))
            idx = self.NAME_BANK.index(person)
            new_name = self.NAME_BANK[(idx - 2 * state["num"]) % len(self.NAME_BANK)]
            line = (
                f"Step {step_no}: Starting from {person}, move backward {2 * state['num']} places in {self.NAME_BANK}. "
                f"Which name do you land on?"
            )
            state["person"] = new_name
        return line

    def _generate_item(self, rng: Random, idx: int) -> dict[str, Any]:
        steps = rng.randint(self.config.min_steps, self.config.max_steps)

        state = {
            "num": rng.randint(2, 9),
            "word": rng.choice(self.WORD_BANK),
            "person": rng.choice(self.NAME_BANK),
        }

        question_lines = []

        step_types = rng.sample(["deduction", "induction", "abduction", "transduction"], 4)

        for i in range(1, steps):
            if i <= 4:
                op = step_types[i - 1]
            else:
                op = rng.choice(["deduction", "induction", "abduction", "transduction"])

            if op == "deduction":
                line = self._deduction(i, rng, state)
            elif op == "induction":
                line = self._induction(i, rng, state)
            elif op == "abduction":
                line = self._abduction(i, rng, state)
            else:
                line = self._transduction(i, rng, state)
            question_lines.append(line)

        vowels = sum(1 for c in state["word"] if c in "aeiou")
        question_lines.append(
            f"Step {steps}: Multiply {state['num']} by the number of vowels in '{state['word']}'. What is the result?"
        )
        final_answer = state["num"] * vowels

        question_text = "\n".join(question_lines)

        return {
            "question": question_text,
            "answer": str(final_answer),
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "num_steps": steps,
            },
        }


class MultiStepReasoningCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(MultiStepReasoningCurriculum.__name__, MultiStepReasoningConfig)
        self._define_attributes(
            ScalarAttributeDefinition(
                name="num_steps",
                field_name="max_steps",
                levels=list(range(5, 11)),
                description="Maximum number of steps in the puzzle",
            )
        )


register_dataset(DATASET_NAME, MultiStepReasoningDataset, MultiStepReasoningConfig, MultiStepReasoningCurriculum)
