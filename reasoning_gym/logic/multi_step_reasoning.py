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
            mult1 = rng.randint(10, 20)
            add = rng.randint(5, 15)
            mult2 = rng.randint(2, 5)
            sub = rng.randint(2, 10)
            res = (state["num"] * mult1 + add - sub) * mult2
            line = (
                f"Step {step_no}: Multiply {state['num']} by {mult1}, add {add}, subtract {sub}, "
                f"then multiply the result by {mult2}. What number results?"
            )
            state["num"] = res
        elif choice < 0.66:
            short_thres = rng.randint(3, 4)
            long_thres = rng.randint(7, 9)
            vowel_count = sum(1 for c in state["word"] if c in "aeiou")
            if len(state["word"]) <= short_thres or vowel_count < 2:
                classification = "small"
            elif len(state["word"]) >= long_thres and vowel_count >= 3:
                classification = "large"
            else:
                classification = "medium"
            line = (
                f"Step {step_no}: Words with \u2264{short_thres} letters or fewer than 2 vowels are 'small'; "
                f"those with \u2265{long_thres} letters and at least 3 vowels are 'large'; otherwise 'medium'. "
                f"Is '{state['word']}' small, medium, or large?"
            )
            state["word"] = classification
        else:
            a, b, c, d = rng.sample(self.NAME_BANK, 4)
            line = (
                f"Step {step_no}: {a} and {b} are siblings. {b} is {c}'s parent and {c} and {d} are siblings. "
                f"Who is {d}'s aunt or uncle?"
            )
            state["person"] = a
        return line

    def _induction(self, step_no: int, rng: Random, state: dict) -> str:
        """Make an inductive reasoning step with slightly harder patterns."""
        choice = rng.random()
        if choice < 0.33:
            mult = rng.randint(2, 4)
            add = rng.randint(3, 7)
            n = rng.randint(3, 5)
            value = state["num"]
            for _ in range(n):
                value = value * mult + add
            line = (
                f"Step {step_no}: Starting at {state['num']}, repeatedly multiply by {mult} and add {add} "
                f"for {n} iterations. What number results?"
            )
            state["num"] = value
        elif choice < 0.66:
            new_word = state["word"][::2][::-1] + state["word"][-1]
            line = (
                f"Step {step_no}: Take every second letter of '{state['word']}', reverse those letters, "
                f"and append the last letter of the original word. What word results?"
            )
            state["word"] = new_word
        else:
            start = state.get("person", rng.choice(self.NAME_BANK))
            forward = rng.randint(1, 3)
            backward = rng.randint(1, 3)
            n = rng.randint(2, 4)
            idx = self.NAME_BANK.index(start)
            for _ in range(n):
                idx = (idx + forward) % len(self.NAME_BANK)
                idx = (idx - backward) % len(self.NAME_BANK)
            target = self.NAME_BANK[idx]
            line = (
                f"Step {step_no}: Starting from {start}, move forward {forward} names then backward {backward} names, "
                f"repeating this {n} times in {self.NAME_BANK}. Which name do you reach?"
            )
            state["person"] = target
        return line

    def _abduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Generate an abductive reasoning step with trickier inference."""
        choice = rng.random()
        if choice < 0.33:
            secret = rng.randint(2, 5)
            mult = rng.randint(2, 4)
            add = state["num"] - secret**3 * mult
            wrong = max(2, secret + rng.randint(1, 3))
            options = [secret, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: The number {state['num']} was made by cubing a secret number, multiplying by {mult}, "
                f"and adding {add}. Was that number {options[0]} or {options[1]}?"
            )
            state["num"] = secret
        elif choice < 0.66:
            shift = rng.randint(1, 3)
            orig = rng.choice(self.WORD_BANK)
            encoded = "".join(chr(((ord(c) - 97 + shift) % 26) + 97) for c in orig)
            encoded = encoded[::-1]
            wrong = rng.choice([w for w in self.WORD_BANK if w != orig])
            options = [orig, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: Each letter of a secret word was shifted forward by {shift} and then the result was reversed to get '{encoded}'. "
                f"Was the original word '{options[0]}' or '{options[1]}'?"
            )
            state["word"] = orig
        else:
            a, b, c, d = rng.sample(self.NAME_BANK, 4)
            line = (
                f"Step {step_no}: {a} is {b}'s parent. {b} and {c} are siblings. {c} is {d}'s parent. "
                f"Who is {a} to {d}?"
            )
            state["person"] = a
        return line

    def _transduction(self, step_no: int, rng: Random, state: dict) -> str:
        """Perform a transduction step mixing representations."""
        choice = rng.random()
        if choice < 0.33:
            if rng.random() < 0.5:
                base = 4
                base_repr = np.base_repr(state["num"], base=base)
                res = base_repr.count("3")
                line = f"Step {step_no}: Write {state['num']} in base {base}. How many digits '3' appear?"
            else:
                bin_str = bin(state["num"])[2:]
                res = int(bin_str[::-1], 2)
                line = (
                    f"Step {step_no}: Write {state['num']} in binary and reverse the digits. "
                    "What is the decimal value of the result?"
                )
            state["num"] = res
        elif choice < 0.66:
            index = (state["num"] ** 3 + state["num"]) % len(self.WORD_BANK)
            res_word = self.WORD_BANK[index]
            line = (
                f"Step {step_no}: Cube {state['num']} and add it to itself, then use this as an index into {self.WORD_BANK}. "
                f"Which word do you get?"
            )
            state["word"] = res_word
        else:
            person = state.get("person", rng.choice(self.NAME_BANK))
            idx = self.NAME_BANK.index(person)
            offset = state["num"] ** 2
            if rng.random() < 0.5:
                new_idx = (idx + offset) % len(self.NAME_BANK)
                line = (
                    f"Step {step_no}: Starting from {person}, move forward {offset} places in {self.NAME_BANK}. "
                    f"Which name do you land on?"
                )
            else:
                new_idx = (idx - offset) % len(self.NAME_BANK)
                line = (
                    f"Step {step_no}: Starting from {person}, move backward {offset} places in {self.NAME_BANK}. "
                    f"Which name do you land on?"
                )
            state["person"] = self.NAME_BANK[new_idx]
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
        name_len = len(state["person"])
        question_lines.append(
            f"Step {steps}: Multiply {state['num']} by the number of vowels in '{state['word']}' "
            f"and then by the number of letters in '{state['person']}'. What is the result?"
        )
        final_answer = state["num"] * vowels * name_len

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
