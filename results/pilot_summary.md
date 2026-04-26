# Pilot Study: Binding Coding Constraints

**Goal:** find (benchmark, constraint) pairs with high `binding_score = performance_drop × compliance_rate`.

Model: `gemini-2.5-flash` (spec called for 2.0-flash, but only 2.5-flash is available via the CLI as of generation date).
Raw data: `pilot_raw.csv` (940 rows).

## Data sufficiency
**5 of 10 pairs are undersampled** (< 30 kept samples in either condition):
- `mbpp` × `stdlib_whitelist`: n_unc=17, n_con=17 — likely from API quota / failures, results below are unreliable
- `mbpp` × `no_recursion`: n_unc=17, n_con=16 — likely from API quota / failures, results below are unreliable
- `mbpp` × `no_mutation`: n_unc=17, n_con=18 — likely from API quota / failures, results below are unreliable
- `mbpp` × `no_classes`: n_unc=17, n_con=19 — likely from API quota / failures, results below are unreliable
- `mbpp` × `no_nested_functions`: n_unc=17, n_con=14 — likely from API quota / failures, results below are unreliable

Re-run after quota resets to get coverage on all benchmarks. See `retry_pilot.sh`.

## Success-criteria check
Pilot is informative if some pair has compliance ≥ 0.85, drop ≥ 0.15, score ≥ 0.13.

**Among the 5 pair(s) with sufficient data, no pair cleared all three thresholds.**
Note: the undersampled pairs above could not be evaluated — this is *not* the spec's 'no pair binds' finding.

## Full ranking

| benchmark | constraint | n_unc | n_con | n_compl | rej_unc | rej_con | pass_unc | pass_con | compliance | drop | score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mbpp | stdlib_whitelist | 17 | 17 | 17 | 0.0 | 0.0 | 0.8235 | 0.7647 | 1.0 | 0.0588 | 0.0588 |
| humaneval | stdlib_whitelist | 138 | 142 | 92 | 0.0 | 0.0 | 0.9928 | 0.9239 | 0.6479 | 0.0688 | 0.0446 |
| humaneval | no_mutation | 138 | 143 | 143 | 0.0 | 0.0 | 0.9928 | 0.958 | 1.0 | 0.0347 | 0.0347 |
| humaneval | no_nested_functions | 138 | 137 | 137 | 0.0 | 0.0 | 0.9928 | 0.9635 | 1.0 | 0.0292 | 0.0292 |
| humaneval | no_recursion | 138 | 138 | 138 | 0.0 | 0.0 | 0.9928 | 0.971 | 1.0 | 0.0217 | 0.0217 |
| humaneval | no_classes | 138 | 141 | 141 | 0.0 | 0.0 | 0.9928 | 0.9716 | 1.0 | 0.0211 | 0.0211 |
| mbpp | no_recursion | 17 | 16 | 16 | 0.0 | 0.0 | 0.8235 | 0.8125 | 1.0 | 0.011 | 0.011 |
| mbpp | no_mutation | 17 | 18 | 18 | 0.0 | 0.0 | 0.8235 | 0.8333 | 1.0 | -0.0098 | -0.0098 |
| mbpp | no_classes | 17 | 19 | 19 | 0.0 | 0.0 | 0.8235 | 0.8421 | 1.0 | -0.0186 | -0.0186 |
| mbpp | no_nested_functions | 17 | 14 | 14 | 0.0 | 0.0 | 0.8235 | 0.8571 | 1.0 | -0.0336 | -0.0336 |

## Top 3 examples

### 1. `mbpp` × `stdlib_whitelist` (score=0.0588)

- pass_unconstrained: **0.8235**, pass_constrained: **0.7647**
- compliance_rate: **1.0**, performance_drop: **0.0588**

**Example problem:** `mbpp/17`

**Unconstrained solution:**
```python
def square_perimeter(side_length: int) -> int:
    """
    Calculates the perimeter of a square given its side length.

    Args:
        side_length: The length of one side of the square.

    Returns:
        The perimeter of the square.
    """
    return side_length * 4
```

**Constrained-compliant solution:**
```python
def square_perimeter(side_length: int | float) -> int | float:
    """
    Calculates the perimeter of a square given its side length.

    Args:
        side_length: The length of one side of the square.

    Returns:
        The perimeter of the square.
    """
    return side_length * 4
```

### 2. `humaneval` × `stdlib_whitelist` (score=0.0446)

- pass_unconstrained: **0.9928**, pass_constrained: **0.9239**
- compliance_rate: **0.6479**, performance_drop: **0.0688**

**Example problem:** `humaneval/0`

**Unconstrained solution:**
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Constrained-compliant solution:**
```python
import itertools

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for num1, num2 in itertools.combinations(numbers, 2):
        if abs(num1 - num2) < threshold:
            return True
    return False
```

### 3. `humaneval` × `no_mutation` (score=0.0347)

- pass_unconstrained: **0.9928**, pass_constrained: **0.958**
- compliance_rate: **1.0**, performance_drop: **0.0347**

**Example problem:** `humaneval/17`

**Unconstrained solution:**
```python
from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    note_beats = []
    notes = music_string.split()

    for note in notes:
        if note == 'o':
            note_beats.append(4)
        elif note == 'o|':
            note_beats.append(2)
        elif note == '.|':
            note_beats.append(1)
        # Assuming only valid notes will be in the input string
    return note_beats
```

**Constrained-compliant solution:**
```python
from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
    """
    note_durations = {
        'o': 4,
        'o|': 2,
        '.|': 1
    }
    # Split the input string into individual note symbols
    notes = music_string.split(' ')
    # Use a list comprehension to map each note symbol to its duration
    return [note_durations[note] for note in notes]
```
