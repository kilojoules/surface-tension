The constraint to avoid explicit loops forces a shift from imperative state management to a functional paradigm. While not necessarily "cleaner" for complex logic, it encourages the use of higher-order functions and declarative data transformations. To solve this problem, I will use `functools.reduce` to maintain the state of remaining counts of each number and the target index across the sequence generation, and list comprehensions for iterative calculations.

```python
import sys
from functools import reduce
from math import factorial

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Since we need a specific index, and S can be enormous, 
    # we cannot calculate S directly.
    # However, the problem asks for the "middle" sequence.
    # A key observation: the lexicographical reverse of a sequence 
    # (replacing x with N-x+1) maps the i-th sequence to the (S-i+1)-th.
    # The middle sequence is the one that is "self-dual" or the 
    # start of the second half.
    
    # For a sequence A, let A' be the sequence where each element x is replaced by (N - x + 1).
    # If A is the i-th sequence, A' is the (S - i + 1)-th sequence.
    # We want the floor((S+1)/2)-th sequence.
    # If S is even, we want the (S/2)-th.
    # If S is odd, we want the ((S+1)/2)-th.
    
    # This specific index is exactly the sequence that comes just before 
    # the point where the sequence and its "dual" swap roles.
    # Specifically, we want the largest sequence A such that A <= A'.
    # A <= A' means at the first index i where A_i != A'_i, we have A_i < A'_i.
    # A_i < A'_i means A_i < N - A_i + 1, or 2 * A_i < N + 1.
    
    # To find the largest A such that A <= A', we try to place the largest 
    # possible numbers at the earliest positions, provided that the 
    # resulting sequence remains lexicographically <= its dual.
    
    # The condition A <= A' is satisfied if:
    # 1. For all i, A_i = N - A_i + 1 (only possible if N is odd and A_i = (N+1)/2)
    # 2. At the first index i where A_i != (N+1)/2, we have A_i < (N+1)/2.
    
    # Wait, the simplest way to find the floor((S+1)/2)-th sequence is:
    # It is the last sequence A such that A <= A'.
    # To maximize A, we want the first differing element from the dual to be 
    # as late as possible, and at that position, it must be the largest 
    # value that is still < (N+1)/2.
    
    # Correct logic for floor((S+1)/2)-th:
    # We want the largest sequence A such that A <= A'.
    # This means we want to fill the sequence from left to right.
    # At each position, we try the largest possible digit d (from N down to 1).
    # We can place d if there exists a completion of the sequence that is <= its dual.
    # The smallest possible completion is the one that is lexicographically smallest.
    # The largest possible completion is the one that is lexicographically largest.
    
    # Actually, the simplest characterization:
    # The sequence is the one that is "just barely" smaller than or equal to its dual.
    # We can greedily pick the largest digit d for the current position such that
    # the remaining counts allow the sequence to be <= its dual.
    # A sequence A is <= A' if at the first index i where A_i != (N+1)/2, A_i < (N+1)/2.
    # If all A_i = (N+1)/2, then A = A'.
    
    # Let's refine: we want the largest A such that A <= A'.
    # For the first position i, we try d = N, N-1, ..., 1.
    # If we pick d < (N+1)/2, then for all subsequent positions, we can pick the 
    # largest possible digits (since the condition A < A' is already satisfied).
    # If we pick d > (N+1)/2, then we must have already had some A_j < (N+1)/2 for j < i.
    # If we pick d = (N+1)/2, the condition depends on future elements.
    
    # State: (current_counts, has_become_smaller)
    # current_counts: list of remaining counts for 1..N
    # has_become_smaller: boolean
    
    def get_next_digit(state):
        counts, smaller = state
        # Try digits d from N down to 1
        # If smaller is True, we can pick the largest available d.
        # If smaller is False:
        #   - If d < (N+1)/2, we can pick it and smaller becomes True.
        #   - If d == (N+1)/2 (and N is odd), we can pick it and smaller remains False.
        #   - If d > (N+1)/2, we cannot pick it because that would make A > A' 
        #     unless smaller was already True.
        
        # Since we want the largest A, we check d from N down to 1.
        # But we must ensure a valid sequence can be formed.
        
        # If smaller is True: pick largest d with counts[d-1] > 0.
        # If smaller is False:
        #   - Try d = (N+1)//2 if N is odd and counts[d-1] > 0.
        #   - If that's not possible or we want to check if a larger d is possible:
        #     Actually, if smaller is False, we can't pick d > (N+1)/2.
        #     We can pick d = (N+1)//2 (if N odd), then smaller remains False.
        #     Or we can pick d < (N+1)//2, then smaller becomes True.
        
        # To maximize A, we first try d = (N+1)//2 (if N odd), then d < (N+1)//2.
        # Wait, if smaller is False, and we pick d < (N+1)//2, we can then pick 
        # the largest possible digits for the rest. 
        # If we pick d = (N+1)//2, we are still constrained.
        # So we compare:
        # 1. Pick d = (N+1)//2, then continue greedily.
        # 2. Pick d = max(j < (N+1)/2), then pick all remaining greedily.
        
        # Actually, the simplest greedy:
        # At each step, try d from N down to 1.
        # If smaller is True, any d with count > 0 is fine.
        # If smaller is False:
        #   - If d > (N+1)/2: Not allowed.
        #   - If d == (N+1)/2: Allowed, smaller remains False.
        #   - If d < (N+1)/2: Allowed, smaller becomes True.
        
        # But we must be careful. If we pick d = (N+1)/2, we are still restricted.
        # If we pick d < (N+1)/2, we are suddenly unrestricted.
        # Is it possible that picking d < (N+1)/2 now allows a much larger 
        # sequence later than picking d = (N+1)/2 now?
        # Example: N=3, K=1. (N+1)/2 = 2.
        # Options: 
        # - d=2: remaining {1, 3}. Next can be 3, then 1. Sequence (2, 3, 1).
        # - d=1: remaining {2, 3}. Next can be 3, then 2. Sequence (1, 3, 2).
        # (2, 3, 1) is larger. So we prefer d = (N+1)//2 if possible.
        
        # Correct Greedy:
        # For i = 1 to NK:
        #   For d = N down to 1:
        #     If count[d] > 0:
        #       If smaller == True: 
        #         return d, True
        #       Else (smaller == False):
        #         If d < (N+1)/2: return d, True
        #         If d == (N+1)/2 and N % 2 == 1: return d, False
        #         If d > (N+1)/2: continue (cannot pick)
        
        # Let's trace Sample 1: N=2, K=2. (N+1)/2 = 1.5.
        # i=1: d=2(>1.5, S=F)X, d=1(<1.5, S=F) -> pick 1, S=True.
        # i=2: d=2(S=T) -> pick 2, S=True.
        # i=3: d=2(S=T) -> pick 2, S=True.
        # i=4: d=1(S=T) -> pick 1, S=True.
        # Result: 1 2 2 1. Correct.
        
        # Sample 3: N=6, K=1. (N+1)/2 = 3.5.
        # i=1: d=6,5,4(>3.5, S=F)X, d=3(<3.5, S=F) -> pick 3, S=True.
        # i=2: d=6(S=T) -> pick 6, S=True.
        # i=3: d=5(S=T) -> pick 5, S=True.
        # i=4: d=4(S=T) -> pick 4, S=True.
        # i=5: d=2(S=T) -> pick 2, S=True.
        # i=6: d=1(S=T) -> pick 1, S=True.
        # Result: 3 6 5 4 2 1. Correct.
        
        # Sample 4: N=3, K=3. (N+1)/2 = 2.
        # i=1: d=3(>2, S=F)X, d=2(==2, S=F) -> pick 2, S=False.
        # i=2: d=3(>2, S=F)X, d=2(==2, S=F) -> pick 2, S=False.
        # i=3: d=3(>2, S=F)X, d=2(==2, S=F) -> pick 2, S=False.
        # i=4: d=3(>2, S=F)X, d=2(count=0), d=1(<2, S=F) -> pick 1, S=True.
        # i=5: d=3(S=T) -> pick 3, S=True.
        # i=6: d=3(S=T) -> pick 3, S=True.
        # i=7: d=3(S=T) -> pick 3, S=True.
        # i=8: d=1(S=T) -> pick 1, S=True.
        # i=9: d=1(S=T) -> pick 1, S=True.
        # Result: 2 2 2 1 3 3 3 1 1. Correct.

        # Implementation using reduce to avoid loops:
        pass

    # To implement this without loops, we use reduce over a range of NK.
    # The state will be (current_counts, smaller, result_sequence).
    
    initial_counts = [k] * n
    
    def step(state, _):
        counts, smaller, res = state
        # Find the best d
        # We need to find the largest d in N..1 that satisfies the condition.
        # We can use a list comprehension to find all valid d's and take the max.
        valid_ds = [
            d for d in range(1, n + 1)
            if counts[d-1] > 0 and (
                smaller or (d < (n +