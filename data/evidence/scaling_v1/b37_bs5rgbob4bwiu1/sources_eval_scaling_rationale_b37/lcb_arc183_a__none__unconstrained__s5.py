The constraint to avoid explicit loops forces a shift toward functional programming paradigms, emphasizing declarative data transformations over imperative state updates. While not always "cleaner" for complex algorithms, it encourages the use of higher-order functions and recursion, which can lead to more concise expressions of logic. For this specific problem, I will use `functools.reduce` to simulate the iterative process of building the sequence and recursion for the combinatorial logic.

```python
import sys
from functools import reduce

# Increase recursion depth for deep state transitions
sys.setrecursionlimit(10000)

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence. Due to the symmetry of lexicographical 
    # order, the middle sequence is the one that is its own "complement"
    # (replacing i with N-i+1) if we were looking for the exact center,
    # but specifically, the complement of the m-th sequence is the (S-m+1)-th.
    # The sequence at index floor((S+1)/2) is the one that, when mirrored
    # (replacing each x with N-x+1 and reversing the sequence), 
    # is the "ceiling" counterpart.
    
    # Actually, a much simpler property exists:
    # The sequence at index floor((S+1)/2) is the one that is 
    # lexicographically "halfway".
    # For any sequence A, its dual A' (where A'_i = N + 1 - A_i) 
    # satisfies: A < B iff A' > B'.
    # Thus, the sequence we are looking for is the one where 
    # A is "just smaller" than or equal to its dual A'.
    # This means at the first index i where A_i != A'_i, we must have A_i < A'_i.
    # Wait, the simplest way to find the middle sequence is to realize that
    # the sequence A is the floor((S+1)/2)-th if and only if 
    # A is the lexicographically smallest sequence such that 
    # A >= dual(A) is false, or more simply:
    # We want the sequence A such that we cannot find a sequence B < A 
    # where B is the dual of some C > A.
    
    # Correct logic for the middle sequence:
    # We want the sequence A such that the number of sequences < A 
    # is just under S/2.
    # This is equivalent to constructing the sequence greedily:
    # At each position, try digits v = 1, 2, ..., N.
    # Calculate how many sequences start with the current prefix.
    # If this count is less than the remaining target rank, subtract and move to v+1.
    # However, S can be massive, so we cannot compute it directly.
    
    # Key Insight: The "middle" sequence of all permutations of a multiset
    # is the one that is "self-dual" in a sense.
    # Specifically, for the middle sequence A, A_i + A_{NK-i+1} = N + 1
    # is NOT necessarily true, but the sequence is the one that 
    # balances the distribution.
    
    # Let's use the property: The target sequence A is the one where
    # for the first index i where A_i != (N + 1 - A_{NK-i+1}), 
    # we have A_i < (N + 1 - A_{NK-i+1}).
    # Actually, the most reliable way to find the middle sequence is:
    # A sequence A is the floor((S+1)/2)-th if A is the smallest sequence
    # such that A >= dual(A) is false? No.
    # Let's use the property: The middle sequence is the one that 
    # "looks" like the median.
    # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
    # S=6, floor(7/2)=3. Result: (1,2,2,1).
    # Dual of (1,2,2,1) is (2,1,1,2). (1,2,2,1) < (2,1,1,2).
    
    # The condition to be the floor((S+1)/2)-th sequence is:
    # A is the lexicographically smallest sequence such that A >= dual(A) is FALSE,
    # UNLESS S is odd, then A = dual(A).
    # Wait, if S is even, the middle two are A and dual(A). 
    # Since A < dual(A), A is the (S/2)-th and dual(A) is (S/2 + 1)-th.
    # floor((S+1)/2) for S even is S/2. So we want the larger of the two 
    # middle ones? No, floor(6.5) = 6? No, floor((6+1)/2) = 3.
    # For S=6, 3rd is (1,2,2,1). Dual is (2,1,1,2) which is 4th.
    # So we want the sequence A such that A < dual(A) and A is the 
    # largest such sequence.
    # This is equivalent to: A_i is chosen such that we stay "just below" 
    # the point where A becomes >= dual(A).
    
    # The simplest construction:
    # At each step i from 1 to NK:
    # Try v = 1, 2, ..., N.
    # If we pick v, we must ensure that the resulting sequence A 
    # satisfies A < dual(A) or (S is odd and A = dual(A)).
    # This is still hard. Let's use the property:
    # The middle sequence A satisfies:
    # For the first index i where A_i != N + 1 - A_{NK-i+1}, 
    # A_i < N + 1 - A_{NK-i+1}.
    # To make A the largest such sequence, we want A_i to be as large as possible.
    # This means we want A_i to be "just" smaller than N + 1 - A_{NK-i+1}.
    
    # Correct Greedy Approach:
    # We want the largest A such that A < dual(A).
    # This means at the first index i where A_i != dual(A)_i, we need A_i < dual(A)_i.
    # To maximize A, we want this first difference to occur as late as possible,
    # and at that index, A_i should be as large as possible while remaining < dual(A)_i.
    # For all j < i, A_j = dual(A)_j, which means A_j = N + 1 - A_{NK-j+1}.
    
    # Let's refine:
    # We want the largest A such that A < dual(A).
    # This means A_1 <= N + 1 - A_{NK}, A_2 <= N + 1 - A_{NK-1}, etc.
    # To maximize A, we try to set A_i = N + 1 - A_{NK-i+1} for as many i as possible.
    # But we must have A < dual(A), so at some index i, A_i < N + 1 - A_{NK-i+1}.
    # To maximize A, we want this i to be as large as possible.
    # The largest possible i is NK // 2 + 1 (the middle element).
    # If NK is even, the middle elements are at NK//2 and NK//2 + 1.
    # If NK is odd, the middle element is at (NK+1)//2.
    
    # For Sample 1: N=2, K=2. NK=4. Middle indices 2, 3.
    # Try to keep A_1 = 3 - A_4, A_2 = 3 - A_3.
    # To have A < dual(A), we need the first difference to be A_i < dual(A)_i.
    # To maximize A, we want A_1 = 3-A_4, A_2 < 3-A_3.
    # For A_2 < 3-A_3 to hold, and A_2 to be max, we need A_2 = 1, 3-A_3 = 2 => A_3 = 1.
    # Then A_1 = 3-A_4. To maximize A, A_1=2, A_4=1.
    # Sequence: (2, 1, 1, 2). Wait, Sample 1 says (1, 2, 2, 1).
    # Let's re-read: floor((6+1)/2) = 3. Sequences:
    # 1: 1,1,2,2 | Dual: 2,2,1,1 (6th)
    # 2: 1,2,1,2 | Dual: 2,1,2,1 (5th)
    # 3: 1,2,2,1 | Dual: 2,1,1,2 (4th)
    # So the 3rd is the smaller of the middle pair.
    # We want the LARGEST A such that A < dual(A).
    # In Sample 1, A=(1,2,2,1) is the largest sequence such that A < dual(A).
    # Check: (1,2,2,1) < (2,1,1,2). Correct.
    
    # How to construct the largest A < dual(A)?
    # We want A_i = N + 1 - A_{NK-i+1} for as many leading i as possible.
    # But we need A < dual(A), so at the first index i where they differ, A_i < dual(A)_i.
    # To maximize A, we want this i to be as large as possible.
    # The symmetry forces A_i = N + 1 - A_{NK-i+1} for i = 1, ..., floor(NK/2).
    # Then at i = floor(NK/2) + 1, we need A_i < N + 1 - A_{NK-i+1}.
    # To maximize A, we make A_i as large as possible, and A_{NK-i+1} as small as possible.
    
    # Let's trace N=2, K=2 again:
    # i=1: A_1 = 3 - A_4.
    # i=2: A_2 < 3 - A_3.
    # To maximize A, we want A_1 to be large. But A_1 is constrained by A_4.
    # This is a constraint satisfaction problem.
    # We need to pick A_1, ..., A_{NK} such that:
    # 1. Each number 1..N appears K times.
    # 2. A_i = N + 1 - A_{NK-i+1} for i < mid.
    # 3. A_{mid} < N + 1 - A_{NK-mid+1}.
    # 4. A is maximized.
    
    # To maximize A, we should pick A_1 as large as possible, then A_2, etc.
    # But A_1 is tied to A_{NK}. If we pick A_1 large, A_{NK} must be small.
    # This is actually simpler:
    # We want the largest A such that A < dual(A).
    # This means A_1 <= N + 1 - A_{NK}.
    # If A_1 < N + 1 - A_{NK}, then A is definitely < dual(A). To maximize A, we then make A_2...A_{NK} as large as possible.
    # If A_1 = N + 1 - A_{NK}, we move to A_2 and A_{NK-1}.
    
    # Correct Greedy Strategy to find largest A < dual(A):
    # For i = 1 to NK // 2:
    #   Try to set A_i = N + 1 - A_{NK-i+1}.
    #   This is possible if we have enough counts of both values.
    #   But we must ensure that we don't "run out" of options to make A < dual(A) later.
    #   Actually, the condition A < dual(A) is satisfied if at the first index i where A_i != dual(A)_i, A_i