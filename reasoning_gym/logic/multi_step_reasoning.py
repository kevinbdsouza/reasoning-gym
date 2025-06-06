from dataclasses import dataclass
from random import Random
from typing import Any, Optional

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
        choice = rng.random()
        if choice < 0.33 or state.get("word") is None:
            mult = rng.randint(2, 5)
            add = rng.randint(1, 9)
            res = state["num"] * mult + add
            line = f"Step {step_no}: Multiply {state['num']} by {mult} and add {add}. What do you get?"
            state["num"] = res
        elif choice < 0.66:
            threshold = rng.randint(4, 7)
            classification = "long" if len(state["word"]) >= threshold else "short"
            line = (
                f"Step {step_no}: Words with at least {threshold} letters are called 'long'. "
                f"Is '{state['word']}' long or short?"
            )
            state["word"] = classification
        else:
            a, b, c = rng.sample(self.NAME_BANK, 3)
            line = f"Step {step_no}: {a} is older than {b} and {b} is older than {c}. " "Who is the oldest?"
            state["person"] = a
        return line

    def _induction(self, step_no: int, rng: Random, state: dict) -> str:
        choice = rng.random()
        if choice < 0.33:
            inc = rng.randint(2, 5)
            n = rng.randint(3, 5)
            res = state["num"] + inc * (n - 1)
            line = f"Step {step_no}: Start at {state['num']} and add {inc} each time. " f"What is the {n}th term?"
            state["num"] = res
        elif choice < 0.66:
            reps = rng.randint(1, 3)
            res_word = state["word"] + state["word"][-1] * reps
            line = f"Step {step_no}: Repeat the last letter of '{state['word']}' {reps} times. " "What word results?"
            state["word"] = res_word
        else:
            start = state.get("person", rng.choice(self.NAME_BANK))
            n = rng.randint(1, 3)
            idx = self.NAME_BANK.index(start)
            target = self.NAME_BANK[(idx + n) % len(self.NAME_BANK)]
            line = (
                f"Step {step_no}: Starting from {start} and moving {n} places forward alphabetically "
                f"in {self.NAME_BANK}, which name do you reach?"
            )
            state["person"] = target
        return line

    def _abduction(self, step_no: int, rng: Random, state: dict) -> str:
        choice = rng.random()
        if choice < 0.33:
            # numeric abduction
            for _ in range(10):
                secret = rng.randint(2, 8)
                mult = rng.randint(2, 5)
                add = state["num"] - secret * mult
                if 1 <= add <= 9:
                    break
            wrong = secret + rng.randint(1, 4)
            options = [secret, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: The number {state['num']} was made by multiplying a secret number by {mult} "
                f"and adding {add}. Was that number {options[0]} or {options[1]}?"
            )
            state["num"] = secret
        elif choice < 0.66:
            shift = rng.choice([1, 2])
            orig = "".join(chr(((ord(c) - 97 - shift) % 26) + 97) for c in state["word"])
            wrong = rng.choice([w for w in self.WORD_BANK if w != orig])
            options = [orig, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: The word '{state['word']}' was formed by shifting a secret word forward by {shift} letters. "
                f"Was the original word '{options[0]}' or '{options[1]}'?"
            )
            state["word"] = orig
        else:
            shift = rng.choice([1, 2])
            orig_name = rng.choice(self.NAME_BANK)
            encoded = "".join(chr(((ord(c.lower()) - 97 + shift) % 26) + 97) for c in orig_name)
            wrong = rng.choice([n for n in self.NAME_BANK if n != orig_name])
            options = [orig_name, wrong]
            rng.shuffle(options)
            line = (
                f"Step {step_no}: A secret name was shifted forward by {shift} letters to become '{encoded}'. "
                f"Was it '{options[0]}' or '{options[1]}'?"
            )
            state["person"] = orig_name
        return line

    def _transduction(self, step_no: int, rng: Random, state: dict) -> str:
        choice = rng.random()
        if choice < 0.33:
            if rng.random() < 0.5:
                res = bin(state["num"]).count("1")
                line = f"Step {step_no}: Write {state['num']} in binary. How many ones appear?"
            else:
                hex_repr = hex(state["num"])[2:]
                res = int(hex_repr[0], 16)
                line = (
                    f"Step {step_no}: Convert {state['num']} to hexadecimal. "
                    "What is the decimal value of the first digit?"
                )
            state["num"] = res
        elif choice < 0.66:
            index = state["num"] % len(self.WORD_BANK)
            res_word = self.WORD_BANK[index]
            line = (
                f"Step {step_no}: Use {state['num']} as an index to pick a word from {self.WORD_BANK}. "
                f"Which word do you get?"
            )
            state["word"] = res_word
        else:
            person = state.get("person", rng.choice(self.NAME_BANK))
            idx = self.NAME_BANK.index(person)
            new_name = self.NAME_BANK[(idx + state["num"]) % len(self.NAME_BANK)]
            line = (
                f"Step {step_no}: Starting from {person}, move {state['num']} places forward in {self.NAME_BANK}. "
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

        question_lines.append(
            f"Step {steps}: Add the number of letters in '{state['word']}' to {state['num']}. What is the result?"
        )
        final_answer = state["num"] + len(state["word"])

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
