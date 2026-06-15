from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TextMetrics:
    cer: float
    accuracy: float
    distance: int
    reference_length: int


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if char_a == char_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def character_metrics(prediction: str, reference: str) -> TextMetrics:
    distance = levenshtein(prediction, reference)
    length = max(1, len(reference))
    cer = distance / length
    accuracy = max(0.0, 1.0 - cer)
    return TextMetrics(cer=cer, accuracy=accuracy, distance=distance, reference_length=len(reference))

