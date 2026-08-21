```python
import sys
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
    # This is essentially the "middle" sequence.
    # Due to the symmetry of the set of good sequences, 
    # the sequence at index (S+1)//2 is the one that, 
    # when mapped by x -> (N + 1 - x), 
    # is the "complement" of the sequence at index S - (S+1)//2 + 1.
    # Specifically, if a sequence is A, its complement is A' where A'_i = N + 1 - A_i.
    # The lexicographical middle is reached by trying to place the smallest 
    # available number at the current position and checking if the number of 
    # sequences starting with that prefix is less than the target index.
    
    # However, calculating S and comparing is computationally expensive.
    # Observation: The target index is exactly half of the total permutations.
    # For any sequence A, its complement A' (where A'_i = N+1 - A_i) is also a good sequence.
    # Lexicographically, A < A' if the first index i where they differ has A_i < A'_i.
    # This means A_i < N + 1 - A_i, or 2*A_i < N + 1.
    # The "middle" sequence is the one that is lexicographically the smallest 
    # sequence A such that A >= A'.
    # This happens when the first element A_1 is >= (N+1)/2.
    # If A_1 > (N+1)/2, then A > A'.
    # If A_1 = (N+1)/2 (only possible if N is odd), we look at the remaining.
    
    # Actually, the simplest way to find the floor((S+1)/2)-th sequence 
    # is to realize that the set of all good sequences is symmetric.
    # The sequence at rank (S+1)//2 is the one that is "just barely" 
    # greater than or equal to its complement.
    # This is achieved by:
    # 1. For the first position, we want the smallest v such that the number of 
    #    sequences starting with 1...v-1 is < (S+1)//2.
    # 2. Because of symmetry, the number of sequences starting with v is the 
    #    same as those starting with N+1-v.
    # 3. The middle sequence will start with v = (N+1)//2.
    # 4. If N is even, the first half of sequences start with 1...N//2, 
    #    and the second half start with N//2+1...N.
    #    The (S+1)//2-th sequence is the first sequence of the second half.
    #    That is the smallest sequence starting with N//2 + 1.
    # 5. If N is odd, the first (S - S_{mid}) sequences start with 1... (N-1)//2,
    #    then there are sequences starting with (N+1)//2, then (N+3)//2...N.
    #    The middle falls into the block starting with (N+1)//2.
    #    Specifically, it's the middle of the block starting with (N+1)//2.
    
    # Let's refine:
    # Let f(counts) be the number of good sequences given remaining counts of each number.
    # We want the rank target = (S + 1) // 2.
    # For the first position, we try v = 1, 2, ...
    # If target > f(counts with v-1), we subtract and move to v+1.
    
    # But we can't compute f(counts) directly as it's too large.
    # We can use the property: the target is the "median" sequence.
    # The median sequence A satisfies: A is the smallest sequence such that A >= A'
    # where A'_i = N + 1 - A_i.
    # This means at the first index i where A_i != A'_i, we must have A_i > A'_i.
    # Or A_i = A'_i for all i (only if N is odd and we can place (N+1)//2).
    
    # Correct logic for "middle" of symmetric distribution:
    # The sequence is the smallest sequence A such that A >= A'.
    # This means at the first index i where A_i != (N+1)/2, we must have A_i > (N+1)/2.
    # If A_i = (N+1)/2 for all i, that's only possible if N=1.
    # If N > 1:
    # The first index i where A_i != (N+1)/2 must have A_i > (N+1)/2.
    # To make A as small as possible:
    # 1. Fill A_i = (N+1)//2 as much as possible (K times).
    # 2. Then the first available number > (N+1)//2, which is (N//2 + 1).
    # 3. Then fill the rest as small as possible.
    
    # Wait, the above is for A >= A'. Let's test Sample 1: N=2, K=2.
    # S=6, target=3. Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2)...
    # 3rd is (1,2,2,1).
    # My logic: A_i = (2+1)//2 = 1. Fill 1s: (1,1, ...). Then A_i > 1: (1,1,2,2).
    # That's the 1st sequence. Something is wrong.
    
    # Let's use the property: The target is the "middle" sequence.
    # The middle sequence is the one that is the "complement" of itself 
    # in terms of rank.
    # The rank of sequence A is R(A). R(A') = S - R(A) + 1.
    # We want R(A) = (S+1)//2.
    # This means R(A') = S - (S+1)//2 + 1 = (2S - S - 1 + 2)//2 = (S+1)//2.
    # So we are looking for a sequence A such that A is the lexicographical 
    # middle of all good sequences.
    # This is the smallest sequence A such that A >= A'.
    # For Sample 1 (N=2, K=2): A' is (3-A_1, 3-A_2, 3-A_3, 3-A_4).
    # A=(1,2,2,1) -> A'=(2,1,1,2). A < A'.
    # A=(2,1,1,2) -> A'=(1,2,2,1). A > A'.
    # The middle of 6 is 3rd and 4th. (S+1)//2 = 3.
    # The 3rd is (1,2,2,1). The 4th is (2,1,1,2).
    # Note that (1,2,2,1) is the complement of (2,1,1,2).
    # The 3rd sequence is the larger of the two "middle" sequences if we 
    # consider the pair (A, A').
    # Actually, the 3rd is the one that is smaller than its complement.
    # Specifically, it's the largest sequence A such that A < A'.
    
    # To find the largest A such that A < A':
    # We want A_i to be as large as possible, but at the first index i 
    # where A_i != A'_i, we need A_i < A'_i.
    # A'_i = N + 1 - A_i. So A_i < N + 1 - A_i => 2*A_i < N + 1.
    # This means A_i <= N // 2.
    # To make A largest:
    # 1. At the first index i where A_i != A'_i, we want A_i to be the 
    #    largest possible value that is still < A'_i.
    #    That is A_i = N // 2.
    # 2. But we can only do this at index i if we have already filled 
    #    all positions j < i with A_j = A'_j.
    #    A_j = A'_j means A_j = (N+1)/2. This is only possible if N is odd.
    # 3. If N is even, A_j can never be A'_j. So the first index is i=1.
    #    We want A_1 to be the largest value such that A_1 < N + 1 - A_1.
    #    A_1 = N // 2.
    #    Then we want the rest of the sequence to be as large as possible.
    # 4. If N is odd, we can have A_j = (N+1)//2 for some time.
    #    To keep A < A', we can have A_j = (N+1)//2 for all j from 1 to K.
    #    Then at index K+1, we need A_{K+1} < A'_{K+1}, so A_{K+1} = N // 2.
    #    Then the rest as large as possible.
    #    Wait, if we want the largest A < A', we should try to keep A_j = A'_j 
    #    as long as possible? 
    #    If A_j = (N+1)//2, then A'_j = (N+1)//2. They are equal.
    #    The first index i where they differ determines if A < A'.
    #    To make A largest, we want that index i to be as late as possible.
    #    So we fill A_j = (N+1)//2 for all K occurrences.
    #    Then at index K+1, we pick the largest A_{K+1} such that A_{K+1} < A'_{K+1}.
    #    That is A_{K+1} = N // 2.
    #    Then we fill the remaining as large as possible.
    
    # Let's trace Sample 1: N=2, K=2.
    # N is even. i=1. A_1 = 2 // 2 = 1.
    # Remaining: one 1, two 2s.
    # Fill remaining as large as possible: 2, 2, 1.
    # Result: 1, 2, 2, 1. Correct.
    
    # Sample 3: N=6, K=1.
    # N is even. i=1. A_1 = 6 // 2 = 3.
    # Remaining: 1, 2, 4, 5, 6.
    # Fill remaining as large as possible: 6, 5, 4, 2, 1.
    # Result: 3, 6, 5, 4, 2, 1. Correct.
    
    # Sample 4: N=3, K=3.
    # N is odd. Fill A_j = (3+1)//2 = 2 for K=3 times: 2, 2, 2.
    # Remaining: three 1s, three 3s.
    # Next, A_{K+1} = 3 // 2 = 1.
    # Remaining: two 1s, three 3s.
    # Fill remaining as large as possible: 3, 3, 3, 1, 1.
    # Result: 2, 2, 2, 1, 3, 3, 3, 1, 1. Correct.
    
    # General Algorithm:
    # 1. If N is even:
    #    First element is N // 2.
    #    Remaining elements: (N // 2) appears (K-1) times, 
    #    1... (N // 2 - 1) appear K times, 
    #    (N // 2 + 1) ... N appear K times.
    #    Sort remaining descending.
    # 2. If N is odd:
    #    First K elements are (N + 1) // 2.
    #    Next element is N //