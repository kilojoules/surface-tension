```python
import sys
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!^N).
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (a1, a2, ..., am), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # Lexicographically, the middle two sequences (if S is even) are:
    # The last sequence starting with 1 (or the smallest possible prefix) 
    # and the first sequence starting with N (or the largest possible prefix).
    
    # Specifically, for any sequence A, let A' be the sequence where A'_i = N + 1 - A_i.
    # A < A' is not always true, but the set of all sequences is symmetric.
    # The floor((S+1)/2)-th sequence is the "last" sequence that is 
    # lexicographically smaller than or equal to its complement.
    
    # A sequence A is lexicographically smaller than or equal to its complement A'
    # if at the first index i where A_i != A'_i, we have A_i < A'_i.
    # A_i < N + 1 - A_i  => 2 * A_i < N + 1 => A_i <= N // 2.
    
    # To find the floor((S+1)/2)-th sequence, we can construct it greedily.
    # At each position, we want to pick the largest possible digit that still allows
    # the resulting sequence to be <= its complement.
    
    # However, a simpler observation:
    # The middle of the lexicographical list of all permutations of a multiset
    # is reached by taking the "half-way" point.
    # Because of the symmetry (x -> N+1-x), the floor((S+1)/2)-th sequence
    # is the one that is the largest sequence A such that A <= A'.
    # This means at the first index i where A_i != (N+1)/2, we must have A_i < (N+1)/2.
    # To make A the largest such sequence, we want A_i to be as large as possible.
    # So for the first index i where we can't pick (N+1)/2, we pick the largest x < (N+1)/2.
    # Then for all subsequent indices, we pick the largest possible remaining numbers.
    
    # Let's refine this:
    # To get the largest A such that A <= A':
    # 1. Fill positions with (N+1)/2 as long as we have them (only if N is odd).
    # 2. At the first position where we cannot put (N+1)/2 (or if N is even),
    #    we must put a value x < (N+1)/2 to ensure A < A'.
    #    To maximize A, we pick the largest x < (N+1)/2 available.
    # 3. Once we have set A_i < A'_i, the condition A <= A' is satisfied regardless
    #    of the following elements. To maximize A, we fill the rest of the sequence
    #    with the remaining elements in descending order.
    
    # Special case: N=1. S=1, floor(2/2)=1. Sequence is (1,)*K.
    if N == 1:
        print(*( [1] * K ))
        return

    # Middle value
    mid = (N + 1) / 2
    
    # Count of each number
    counts = [K] * N
    
    # Result sequence
    res = []
    
    # 1. Fill with (N+1)//2 if N is odd
    if N % 2 == 1:
        m_val = (N + 1) // 2
        # We can put up to K of these at the start
        # But we need to leave room to eventually put a value < mid
        # Actually, we can put K of them, and then the next available value < mid.
        # But we must ensure there IS a value < mid available.
        # Since K > 0 and N > 1, there are always values < mid.
        
        # To maximize A such that A <= A', we want the first difference to be as late as possible.
        # The maximum number of (N+1)//2 we can put is K.
        # After K of those, the next element must be < (N+1)//2 to satisfy A < A'.
        # Wait, if we put K of (N+1)//2, the next element will be some x.
        # The complement will have K of (N+1)//2 and then (N+1-x).
        # Since x < N+1-x is false if x is the largest remaining, we need to be careful.
        
        # Correct logic:
        # To maximize A subject to A <= A':
        # Find the largest i such that we can have A_j = (N+1)/2 for j < i, and A_i < (N+1)/2.
        # This i is simply K + 1.
        # So: first K elements are (N+1)//2, then the next element is the largest available < (N+1)//2.
        # Then the rest are largest available.
        
        m_val = (N + 1) // 2
        res = [m_val] * K
        counts[m_val - 1] = 0
        
        # Next element: largest x < m_val
        # Since it's a good sequence, x is simply m_val - 1
        res.append(m_val - 1)
        counts[m_val - 2] -= 1
        
        # Remaining: descending order
        remaining = []
        for val in range(N, 0, -1):
            remaining.extend([val] * counts[val - 1])
        res.extend(remaining)
        
    else:
        # N is even. mid = N/2 + 0.5.
        # The first element A_1 must be <= N/2.
        # To maximize A, we pick A_1 = N/2.
        # Then we want to maximize the rest. But we must ensure A <= A'.
        # If we pick A_1 = N/2, then A'_1 = N/2 + 1.
        # Since A_1 < A'_1, the condition A < A' is already satisfied.
        # To maximize A, we fill the rest in descending order.
        
        # Wait, if we pick A_1 < N/2, we can potentially have a larger sequence?
        # No, because A_1 is the most significant digit.
        # Max A_1 such that A_1 <= N + 1 - A_1 is A_1 = N // 2.
        
        # Let's re-evaluate:
        # If A_1 < N/2, then A < A'.
        # If A_1 > N/2, then A > A'.
        # If A_1 = N/2 (only possible if N is even? No, N/2 is not integer).
        # If N=2, K=2: S=6. floor(7/2)=3.
        # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
        # 3rd is (1,2,2,1).
        # My logic: N=2 is even. A_1 = 2//2 = 1. Rest descending: (1, 2, 2, 1). Correct.
        
        # If N=3, K=3: S=3^3*3!/3! = 20? No. S = 9!/(3!^3) = 1680.
        # floor(1681/2) = 840.
        # My logic: N=3 is odd. m_val = 2.
        # res = [2,2,2] + [1] + [3,3,3,1,1] = [2,2,2,1,3,3,3,1,1].
        # Let's check Sample 4: N=3, K=3 -> 2 2 2 1 3 3 3 1 1. Correct.
        
        # For N even:
        # A_1 = N // 2.
        # Then A < A' is already satisfied because A_1 = N // 2 and A'_1 = N // 2 + 1.
        # To maximize A, fill the rest descending.
        
        # Actually, if N is even, we can't have A_i = A'_i for any i.
        # The first index i where they differ will determine if A < A'.
        # To maximize A such that A < A', we want A_i to be as large as possible for small i.
        # But we can't have A_i = A'_i.
        # So for the first index i, we must pick A_i < A'_i.
        # The largest possible value for A_i is N // 2.
        # Once we do that, A < A' is guaranteed.
        
        # But we can have A_i = A'_i for a while? No, only if N is odd and A_i = (N+1)/2.
        # If N is even, A_i is never equal to N + 1 - A_i.
        # So the very first element A_1 must be <= N // 2.
        # To maximize A, we pick A_1 = N // 2, and then the rest descending.
        
        # Let's trace N=2, K=2 again.
        # A_1 = 2 // 2 = 1.
        # Remaining: {1: 1, 2: 2}. Descending: 2, 2, 1.
        # Result: 1, 2, 2, 1. Correct.
        
        # Wait, if N is even, can we pick A_1 = N // 2 and then some other things?
        # Yes, if A_1 = N // 2, then A'_1 = N // 2 + 1.
        # Since A_1 < A'_1, then A < A' regardless of the rest.
        # To maximize A, we just fill the rest descending.
        
        # Is it possible to have A_1 > N // 2?
        # If A_1 > N // 2, then A_1 >= N // 2 + 1.
        # Then A'_1 = N + 1 - A_1 <= N // 2.
        # So A'_1 < A_1, which means A' < A.
        # We want A <= A', so we cannot have A_1 > N // 2.
        
        # Correct logic for N even:
        # A_1 = N // 2.
        # Rest = descending order of remaining elements.
        
        # Let's refine the "N is odd" case.
        # If N is odd, m = (N+1)//2.
        # We can have A_1 = m, A_2 = m, ..., A_K = m.
        # Then we must have A_{K+1} < m to ensure A < A'.
        # The largest such value is m-1.
        # Then the rest are descending.
        
        # This is exactly what I wrote. Let's implement it without the if/else structure.
        pass

# Since I cannot use if/else blocks for logic flow (only for initialization), 
# I will use a list comprehension or a map.
# But the prompt says "if/else" are not allowed for loops/logic? 
# No, it says "Return only Python source". It doesn't forbid if/else.
# It forbids for/while loops.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, K = map(int, input_data)
    
    # We use the logic derived:
    # If N is even: A_1 = N//2, then rest descending.
    # If N is odd: A_1...A_K = (N+1)//2, A_{K+1} = (N+1)//2 - 1, then rest descending.
    
    # To handle this without for/while loops:
    # We can use list multiplications and slicing.
    
    # Case N even:
    #