# Pilot Study: Binding Coding Constraints

**Goal:** find (benchmark, constraint) pairs with high `binding_score = performance_drop × compliance_rate`.

Model: `gemini-2.5-flash` (spec called for 2.0-flash, but only 2.5-flash is available via the CLI as of generation date).
Raw data: `pilot_v4_raw.csv` (1026 rows).

## Data sufficiency
## Success-criteria check
Pilot is informative if some pair has `drop_overall ≥ 0.15` (unbiased binding pressure).

**2 pair(s) cleared the threshold:**
- `lcb_medium` × `stdlib_whitelist`: drop_overall=0.8866, compliance=0.0065
- `lcb_medium` × `no_loops_no_recursion`: drop_overall=0.3397, compliance=0.6467

## Full ranking

Sorted by `drop_overall` (unbiased — counts non-compliant constrained samples as failures).
`drop_among_compliant` is the spec metric, kept for reference but biased upward when compliance < 1.

| benchmark | constraint | n_unc | n_con | n_compl | rej_unc | rej_con | pass_unc | **pass_overall** | **drop_overall** | compliance | pass_compl | drop_compl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lcb_medium | stdlib_whitelist | 159 | 154 | 1 | 0.0702 | 0.0994 | 0.8931 | **0.0065** | **0.8866** | 0.0065 | 1.0 | -0.1069 |
| lcb_medium | no_loops_no_recursion | 159 | 150 | 97 | 0.0702 | 0.1228 | 0.8931 | **0.5533** | **0.3397** | 0.6467 | 0.8557 | 0.0374 |
| lcb_medium | no_loops | 159 | 151 | 148 | 0.0702 | 0.117 | 0.8931 | **0.755** | **0.1381** | 0.9801 | 0.7703 | 0.1228 |
| lcb_medium | no_helpers | 159 | 157 | 155 | 0.0702 | 0.0819 | 0.8931 | **0.8662** | **0.0268** | 0.9873 | 0.8774 | 0.0157 |
| lcb_medium | no_recursion | 159 | 160 | 156 | 0.0702 | 0.0643 | 0.8931 | **0.9062** | **-0.0132** | 0.975 | 0.9295 | -0.0364 |

## Top 3 examples

### 1. `lcb_medium` × `stdlib_whitelist` (drop_overall=0.8866)

- pass_unconstrained: **0.8931**, pass_overall: **0.0065**
- compliance_rate: **0.0065**, drop_among_compliant (biased): -0.1069

**Example problem:** `lcb/abc363_c`

**Unconstrained solution:**
```python
import sys
from itertools import permutations

def is_palindrome(s):
    """Checks if a string is a palindrome."""
    return s == s[::-1]

def solve():
    # Read N and K from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, k = map(int, line1)
        # Read the string S
        s = sys.stdin.readline().strip()
    except ValueError:
        return

    # Since N is very small (up to 10), we can generate all unique permutations.
    # The number of permutations of a string of length 10 is at most 10! = 3,628,800.
    # We can use a set to store unique permutations to handle duplicate characters.
    
    # Using itertools.permutations generates all permutations based on index.
    # We convert them to strings and use a set to keep only unique ones.
    all_perms = set(permutations(s))
    
    count = 0
    for p in all_perms:
        # Convert tuple of characters back to a string
        t = "".join(p)
        
        # Check if any substring of length K is a palindrome
        has_palindrome = False
        for i in range(n - k + 1):
            substring = t[i : i + k]
            if is_palindrome(substring):
                has_palindrome = True
                break
        
        if not has_palindrome:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()
```

**Constrained-compliant solution:**
```python
import itertools
import collections

def solve():
    # Read N and K from the first line
    try:
        line1 = input().split()
        if not line1:
            return
        n, k = map(int, line1)
        # Read the string S
        s = input().strip()
    except EOFError:
        return

    # To find all unique permutations of S, we can use itertools.permutations
    # and put them into a set, but since N is up to 10, 10! is 3,628,800.
    # This is small enough to iterate through in Python if we are careful,
    # but generating all permutations and checking each is the most straightforward way.
    
    # To handle duplicates efficiently, we can use a set of the permutations
    # or use a recursive approach to generate unique permutations.
    
    chars = sorted(list(s))
    used = [False] * n
    current_permutation = []
    count = 0

    def is_palindrome(sub):
        # Check if a substring is a palindrome
        return sub == sub[::-1]

    def contains_palindrome_of_length_k(p):
        # Check if the permutation contains any palindrome of length K
        for i in range(n - k + 1):
            # Extract substring of length k
            sub = p[i : i + k]
            if is_palindrome(sub):
                return True
        return False

    # We use a set to store unique permutations to avoid redundant checks
    # and counting the same string multiple times.
    # However, for N=10, the number of unique permutations is at most 10!
    # Generating them via itertools.permutations and using a set is memory intensive.
    # A better way is to use a recursive generator for unique permutations.
    
    unique_perms = set(itertools.permutations(chars))
    
    valid_count = 0
    for p in unique_perms:
        # p is a tuple of characters
        # Check if this permutation contains a palindrome of length k
        # Optimization: slice the tuple and check
        found_palindrome = False
        for i in range(n - k + 1):
            # Check if p[i...i+k-1] is a palindrome
            is_pal = True
            for j in range(k // 2):
                if p[i + j] != p[i + k - 1 - j]:
                    is_pal = False
                    break
            if is_pal:
                found_palindrome = True
                break
        
        if not found_palindrome:
            valid_count += 1
            
    print(valid_count)

if __name__ == "__main__":
    solve()
```

### 2. `lcb_medium` × `no_loops_no_recursion` (drop_overall=0.3397)

- pass_unconstrained: **0.8931**, pass_overall: **0.5533**
- compliance_rate: **0.6467**, drop_among_compliant (biased): 0.0374

**Example problem:** `lcb/abc364_c`

**Unconstrained solution:**
```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    X = int(input_data[1])
    Y = int(input_data[2])
    
    A = list(map(int, input_data[3:3+N]))
    B = list(map(int, input_data[3+N:3+2*N]))
    
    # To minimize the number of dishes eaten, we want to reach the 
    # stopping condition (Sum(A) > X OR Sum(B) > Y) as quickly as possible.
    # This means we want to maximize the "progress" towards the limit.
    # However, the stopping condition is an OR. To make the total 
    # number of dishes minimum, we want to pick dishes that are 
    # "most effective" at pushing either the sweetness over X or saltiness over Y.
    
    # Let's consider the strategy:
    # We want to find the smallest k such that there exists a subset of k dishes
    # where Sum(A_subset) > X OR Sum(B_subset) > Y.
    
    # This is equivalent to:
    # Find the smallest k such that (max sum of k A_i > X) OR (max sum of k B_i > Y).
    
    # 1. Sort A in descending order to find the minimum k such that Sum(A_i) > X.
    # 2. Sort B in descending order to find the minimum k such that Sum(B_i) > Y.
    # 3. The answer is the minimum of these two values.
    # 4. If neither condition can be met even with all N dishes, the answer is N.
    
    # Sort sweetness and saltiness descending
    sorted_A = sorted(A, reverse=True)
    sorted_B = sorted(B, reverse=True)
    
    ans_A = N + 1
    current_sum_A = 0
    for i in range(N):
        current_sum_A += sorted_A[i]
        if current_sum_A > X:
            ans_A = i + 1
            break
            
    ans_B = N + 1
    current_sum_B = 0
    for i in range(N):
        current_sum_B += sorted_B[i]
        if current_sum_B > Y:
            ans_B = i + 1
            break
            
    # The minimum number of dishes to trigger the stop condition
    result = min(ans_A, ans_B)
    
    # If result is still N+1, it means he eats all N dishes without exceeding limits
    if result > N:
        print(N)
    else:
        print(result)

if __name__ == "__main__":
    solve()
```

**Constrained-compliant solution:**
```python
import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # Parse N, X, Y
    N = int(input_data[0])
    X = int(input_data[1])
    Y = int(input_data[2])
    
    # Parse A and B lists
    A = list(map(int, input_data[3 : 3 + N]))
    B = list(map(int, input_data[3 + N : 3 + 2 * N]))
    
    # To minimize the number of dishes eaten, we want to keep the running totals 
    # <= X and <= Y for as long as possible.
    # This means we want to pick dishes that are "small" in both A and B.
    # Specifically, to avoid exceeding X or Y quickly, we want to pick dishes
    # that contribute the least to the sums.
    # However, we can choose the order. The stop condition is:
    # stop as soon as (Sum A > X) OR (Sum B > Y).
    # To maximize the number of dishes eaten before stopping, we'd pick the smallest.
    # To MINIMIZE the number of dishes eaten, we want to reach the threshold as fast as possible.
    # This means we want to pick dishes that have the largest values.
    
    # Wait, let's re-read: "Find the minimum possible number of dishes that he will end up eating."
    # To stop as quickly as possible, we should pick dishes that make the sum exceed X or Y 
    # in the fewest steps.
    # This means we want to maximize the growth of the sums.
    # But we can't just pick the largest A's because they might have tiny B's.
    # Actually, the problem is: we can arrange ALL N dishes in any order.
    # We stop at index k if (Sum_{i=1 to k} A_i > X) OR (Sum_{i=1 to k} B_i > Y).
    # We want to find the minimum k across all permutations.
    
    # This is equivalent to: 
    # Can we eat k dishes without stopping? 
    # No, we want to stop as early as possible.
    # We stop at k if we can find k dishes such that their sum A > X OR their sum B > Y.
    # The minimum k is the smallest k such that there exists a subset of size k 
    # where (Sum A > X) OR (Sum B > Y).
    
    # To check if there exists a subset of size k with Sum A > X:
    # Just take the k largest A_i. If their sum > X, then k is possible.
    # Similarly for B.
    
    # So we need the smallest k such that:
    # (Sum of k largest A_i > X) OR (Sum of k largest B_i > Y).
    
    # Sort A and B in descending order
    sorted_A = sorted(A, reverse=True)
    sorted_B = sorted(B, reverse=True)
    
    # Compute prefix sums
    # Using map and slice to avoid explicit loops
    prefix_A = list(map(lambda i: sum(sorted_A[:i]), range(1, N + 1)))
    prefix_B = list(map(lambda i: sum(sorted_B[:i]), range(1, N + 1)))
    
    # We need the smallest k (1-indexed) such that prefix_A[k-1] > X or prefix_B[k-1] > Y.
    # We can use a generator expression with next() to find the first index.
    # If no such k exists, the answer is N.
    
    # To avoid the O(N^2) of the prefix sum logic above (sum(sorted_A[:i])), 
    # we use a more efficient way to get prefix sums without loops.
    # Since we can't use loops, we can use itertools.accumulate.
    from itertools import accumulate
    
    pref_A = list(accumulate(sorted_A))
    pref_B = list(accumulate(sorted_B))
    
    # Find the first index where the condition is met.
    # We use a generator and next() to find the first k.
    # range(1, N+1) corresponds to the number of dishes.
    # index i in pref_A is k = i + 1.
    
    ans = next((i + 1 for i in range(N) if pref_A[i] > X or pref_B[i] > Y), N)
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()
```

### 3. `lcb_medium` × `no_loops` (drop_overall=0.1381)

- pass_unconstrained: **0.8931**, pass_overall: **0.755**
- compliance_rate: **0.9801**, drop_among_compliant (biased): 0.1228

**Example problem:** `lcb/abc358_d`

**Unconstrained solution:**
```python
import sys

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    
    # A is the list of prices/candy counts for the N boxes
    A = list(map(int, input_data[2:2+N]))
    # B is the list of minimum candy requirements for the M people
    B = list(map(int, input_data[2+N:2+N+M]))
    
    # To minimize the total cost, we should try to satisfy the most demanding 
    # people (largest B_i) using the cheapest available boxes that meet their requirement.
    # However, a simpler greedy approach is:
    # 1. Sort the requirements B in descending order.
    # 2. Sort the available boxes A in ascending order.
    # 3. For the person with the largest requirement, we need a box with at least that many candies.
    # To keep costs low, we want to use the smallest possible A_i that satisfies B_j.
    
    # Let's refine the greedy strategy:
    # Sort B in descending order.
    # Sort A in ascending order.
    # For each B_j (from largest to smallest), we need to pick a box from A.
    # But wait, if we sort B descending, we must pick from the available A_i that are >= B_j.
    # To minimize total cost, we should prioritize satisfying the hardest requirements 
    # using the smallest possible valid boxes.
    
    # Correct Greedy Strategy:
    # 1. Sort B in descending order.
    # 2. Sort A in ascending order.
    # 3. Use a pointer or binary search to find the smallest A_i that satisfies the current B_j.
    # Actually, if we sort B descending, we can't just pick the smallest A_i because 
    # a small A_i might be needed for a smaller B_k later.
    # Wait, the price is the same as the candy count (A_i).
    # So if A_i >= B_j, the cost is A_i. To minimize sum(A_i), we want the smallest A_i such that A_i >= B_j.
    
    # Let's re-evaluate:
    # Person j needs A_i >= B_j. Cost is A_i.
    # To minimize total cost, for each person j, we want to assign the smallest possible A_i 
    # that is >= B_j and hasn't been used yet.
    # Does the order of people matter? 
    # Yes. If we satisfy the person with the largest B_j first, we take the smallest A_i >= B_j.
    # This is optimal because any A_k that could satisfy a smaller B_m might not satisfy B_j.
    
    B.sort(reverse=True)
    A.sort()
    
    # Since we need to find the smallest A_i >= B_j for the largest B_j, 
    # and A is sorted, we can use a pointer or binary search.
    # But we must remove used boxes. A simple pointer won't work because we might 
    # skip small boxes that are useful for smaller B_j.
    
    # Actually, the most efficient way:
    # Sort B in descending order.
    # Sort A in ascending order.
    # For each B_j (largest to smallest), we need an A_i >= B_j.
    # To minimize cost, we should use the smallest possible A_i that is >= B_j.
    # Since we are processing B_j from largest to smallest, any A_i that satisfies 
    # B_j will also satisfy all subsequent (smaller) B_k.
    # Therefore, the best strategy is to satisfy the largest B_j using the smallest 
    # available A_i that is >= B_j.
    
    # To implement this efficiently:
    # Sort A and B ascending.
    # For the smallest B_j, we want the smallest A_i >= B_j.
    # Let's use two pointers.
    
    A.sort()
    B.sort()
    
    total_cost = 0
    a_ptr = 0
    possible = True
    
    for b_val in B:
        # Find the first A[a_ptr] that is >= b_val
        while a_ptr < N and A[a_ptr] < b_val:
            a_ptr += 1
        
        if a_ptr == N:
            possible = False
            break
        
        total_cost += A[a_ptr]
        a_ptr += 1 # This box is now used
        
    if possible:
        print(total_cost)
    else:
        print("-1")

if __name__ == "__main__":
    solve()
```

**Constrained-compliant solution:**
```python
import sys
import heapq

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:]))

    # To satisfy the condition for each person i (B_i), 
    # we should process people with the largest requirements first.
    # For each person, we pick the cheapest available box that satisfies their B_i.
    
    # Sort A (available boxes) and B (requirements) in descending order.
    A_sorted = sorted(A, reverse=True)
    B_sorted = sorted(B, reverse=True)

    # We use a pointer-like approach via a generator or mapping.
    # Since loops are forbidden, we use a recursive function or 
    # a functional approach to simulate the process of picking the cheapest valid box.
    
    # However, the standard greedy approach for this problem is:
    # 1. Sort B descending.
    # 2. For each b in B, identify all boxes A_i >= b.
    # 3. From those, pick the one with the smallest A_i.
    
    # To avoid loops, we can use a recursive function with a heap.
    # Python's recursion limit needs to be increased.
    sys.setrecursionlimit(300000)

    def process(b_idx, a_idx, heap, total_cost):
        if b_idx == M:
            return total_cost
        
        # Add all boxes that satisfy the current requirement B[b_idx] to the heap
        # Since A and B are sorted descending, we only add boxes starting from a_idx
        def fill_heap(curr_a_idx):
            if curr_a_idx < N and A_sorted[curr_a_idx] >= B_sorted[b_idx]:
                heapq.heappush(heap, A_sorted[curr_a_idx])
                return fill_heap(curr_a_idx + 1)
            return curr_a_idx

        new_a_idx = fill_heap(a_idx)
        
        if not heap:
            return -1
        
        # Pick the cheapest box available in the heap
        cheapest = heapq.heappop(heap)
        res = process(b_idx + 1, new_a_idx, heap, total_cost + cheapest)
        return res

    # The logic above uses recursion to replace loops.
    # We start processing from the first person (largest requirement).
    print(process(0, 0, [], 0))

if __name__ == "__main__":
    solve()
```
