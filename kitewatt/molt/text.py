"""Safe, bounded program synthesis for text transforms.

This is a transparent search over a small DSL, not arbitrary code generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
from typing import Callable
from urllib.parse import urlsplit

OPERATIONS = (
    "strip", "lower", "upper", "collapse_spaces", "spaces_to_dash",
    "email_domain", "last_url_segment", "remove_query", "reverse",
    "keep_digits", "remove_punctuation",
)
TASKS = ("slug", "email_domain", "url_segment")


def apply_op(op: str, text: str) -> str:
    if op == "strip":
        return text.strip()
    if op == "lower":
        return text.lower()
    if op == "upper":
        return text.upper()
    if op == "collapse_spaces":
        return re.sub(r"\s+", " ", text)
    if op == "spaces_to_dash":
        return re.sub(r"\s+", "-", text)
    if op == "email_domain":
        return text.rsplit("@", 1)[-1]
    if op == "last_url_segment":
        return urlsplit(text).path.rstrip("/").split("/")[-1]
    if op == "remove_query":
        return text.split("?", 1)[0]
    if op == "reverse":
        return text[::-1]
    if op == "keep_digits":
        return "".join(ch for ch in text if ch.isdecimal())
    if op == "remove_punctuation":
        return "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    raise ValueError("Unknown operation")


def supported(task: str, text: str) -> bool:
    if type(text) is not str or not 2 <= len(text) <= 180:
        return False
    if task == "slug":
        return bool(re.fullmatch(r"[\w\s-]+", text, flags=re.UNICODE))
    if task == "email_domain":
        return bool(re.fullmatch(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text))
    if task == "url_segment":
        parsed = urlsplit(text)
        return (parsed.scheme in ("https", "http") and bool(parsed.netloc)
                and bool(parsed.path.strip("/")) and not parsed.username
                and not parsed.password)
    return False


@dataclass(frozen=True)
class TextProgram:
    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.operations) <= 3 or any(op not in OPERATIONS for op in self.operations):
            raise ValueError("Only 1–3 built-in operations are allowed")

    def __call__(self, text: str) -> str:
        if type(text) is not str or len(text) > 180:
            raise ValueError("Input must be a short string")
        for op in self.operations:
            text = apply_op(op, text)
        return text

    def to_dict(self) -> dict:
        return {"operations": list(self.operations)}

    @classmethod
    def from_dict(cls, data: dict) -> "TextProgram":
        if not isinstance(data, dict) or set(data) != {"operations"} or not isinstance(data["operations"], list):
            raise ValueError("Invalid text program schema")
        if any(type(op) is not str for op in data["operations"]):
            raise ValueError("Invalid operation name")
        return cls(tuple(data["operations"]))


@dataclass(frozen=True)
class TextSkill:
    task: str
    program: TextProgram
    teacher_calls: int
    validation_count: int
    verified_accuracy: float
    validation_pairs: tuple[tuple[str, str], ...]

    def predict(self, text: str) -> str:
        if not supported(self.task, text):
            raise ValueError("Outside compiled skill domain")
        return self.program(text)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "task": self.task, "program": self.program.to_dict(),
                "teacher_calls": self.teacher_calls, "validation_count": self.validation_count,
                "verified_accuracy": self.verified_accuracy,
                "validation_pairs": [list(p) for p in self.validation_pairs]}

    @classmethod
    def from_dict(cls, data: dict) -> "TextSkill":
        required = {"schema_version", "task", "program", "teacher_calls", "validation_count", "verified_accuracy", "validation_pairs"}
        if not isinstance(data, dict) or set(data) != required or data["schema_version"] != 1:
            raise ValueError("Unknown skill schema")
        if data["task"] not in TASKS or type(data["teacher_calls"]) is not int or not 1 <= data["teacher_calls"] <= 64:
            raise ValueError("Invalid task or query count")
        if type(data["validation_count"]) is not int or not 8 <= data["validation_count"] <= 1000:
            raise ValueError("Invalid validation count")
        if type(data["verified_accuracy"]) is not float or data["verified_accuracy"] != 1.0:
            raise ValueError("Only fully validated skills can be loaded")
        pairs = data["validation_pairs"]
        if (not isinstance(pairs, list) or len(pairs) != data["validation_count"]
                or any(not isinstance(p, list) or len(p) != 2
                       or any(type(v) is not str for v in p) for p in pairs)):
            raise ValueError("Invalid validation examples")
        program = TextProgram.from_dict(data["program"])
        if any(not supported(data["task"], x) or program(x) != y for x, y in pairs):
            raise ValueError("Saved skill failed its validation examples")
        return cls(data["task"], program, data["teacher_calls"],
                   data["validation_count"], data["verified_accuracy"],
                   tuple((x, y) for x, y in pairs))


def programs():
    for count in (1, 2, 3):
        for ops in product(OPERATIONS, repeat=count):
            yield TextProgram(ops)


@dataclass(frozen=True)
class ActiveResult:
    skill: TextSkill | None
    status: str
    teacher_calls: int
    programs_examined: int
    survivors: int
    trace: tuple[dict, ...]


def induce(
    task: str, teacher: Callable[[str], str], discovery: list[str],
    validation: list[str], *, max_queries: int = 24,
    min_queries: int = 6,
    verifier: Callable[[str], str] | None = None,
) -> ActiveResult:
    """Choose informative teacher queries, then certify a selected DSL program.

    Validation is reserved until candidate selection. Test inputs must never be
    passed to this function. A perfect finite validation set is not a proof of
    correctness outside that set.
    """
    if task not in TASKS or not 1 <= min_queries <= max_queries <= 64:
        raise ValueError("Unsupported task or query budget")
    if not 8 <= len(validation) <= 1000 or len(discovery) != len(set(discovery)):
        raise ValueError("Invalid discovery or validation partition")
    if not discovery or set(discovery) & set(validation):
        raise ValueError("Discovery and validation must be disjoint")
    if not all(supported(task, x) for x in discovery + validation):
        raise ValueError("Inputs violate the declared task domain")
    viable = list(programs())
    checked = len(viable)
    remaining = list(discovery)
    trace = []
    while remaining and len(trace) < max_queries:
        # Query-by-committee: select an input where candidate programs disagree.
        # No teacher labels are read until the query has been selected.
        choice, diversity = remaining[0], -1
        for x in remaining[:48]:
            score = len({p(x) for p in viable})
            if score > diversity:
                choice, diversity = x, score
        remaining.remove(choice)
        answer = teacher(choice)
        if type(answer) is not str:
            raise ValueError("Teacher must return a string")
        viable = [p for p in viable if p(choice) == answer]
        trace.append({"input": choice, "teacher_output": answer,
                      "remaining_candidates": len(viable),
                      "competing_predictions": diversity})
        if not viable:
            return ActiveResult(None, "no_matching_program", len(trace), checked, 0, tuple(trace))
        if len(trace) >= min_queries and all(
            len({p(x) for p in viable}) == 1 for x in remaining[:48]
        ):
            break
    if len(trace) < min_queries:
        return ActiveResult(None, "insufficient_examples", len(trace), checked, len(viable), tuple(trace))
    # Prefer shortest representatives; never choose based on validation results.
    chosen = viable[0]
    # An independent verifier is preferable to agreement with the teacher.
    reference = verifier if verifier is not None else teacher
    pairs = tuple((x, reference(x)) for x in validation)
    passes = sum(chosen(x) == y for x, y in pairs)
    if passes != len(validation):
        return ActiveResult(None, "validation_failed", len(trace), checked, len(viable), tuple(trace))
    skill = TextSkill(task, chosen, len(trace), len(validation), 1.0, pairs)
    return ActiveResult(skill, "compiled", len(trace), checked, len(viable), tuple(trace))


class TextRouter:
    """Use validated skills in-domain, with optional independent spot checks."""

    def __init__(self, fallback: Callable[[str, str], str], *,
                 verifier: Callable[[str, str], str] | None = None,
                 verify_every: int = 0) -> None:
        if type(verify_every) is not int or verify_every < 0 or (verify_every and verifier is None):
            raise ValueError("Periodic verification requires a positive interval and verifier")
        self.fallback = fallback
        self.verifier = verifier
        self.verify_every = verify_every
        self.skills: dict[str, TextSkill] = {}
        self.counts = {"compiled": 0, "fallback": 0, "verification_failures": 0}
        self._compiled_calls = 0

    def install(self, skill: TextSkill) -> None:
        if skill.task not in TASKS or skill.verified_accuracy != 1.0:
            raise ValueError("Uncertified skill")
        if len(skill.validation_pairs) != skill.validation_count or any(
            not supported(skill.task, x) or skill.program(x) != y
            for x, y in skill.validation_pairs
        ):
            raise ValueError("Failed certificate revalidation")
        self.skills[skill.task] = skill

    def predict(self, task: str, text: str) -> str:
        skill = self.skills.get(task)
        if skill is not None and supported(task, text):
            result = skill.predict(text)
            self._compiled_calls += 1
            if self.verify_every and self._compiled_calls % self.verify_every == 0:
                assert self.verifier is not None
                checked = self.verifier(task, text)
                if checked != result:
                    # A failed check removes the skill from routing immediately.
                    self.skills.pop(task, None)
                    self.counts["verification_failures"] += 1
                    self.counts["fallback"] += 1
                    return checked
            self.counts["compiled"] += 1
            return result
        self.counts["fallback"] += 1
        return self.fallback(task, text)
