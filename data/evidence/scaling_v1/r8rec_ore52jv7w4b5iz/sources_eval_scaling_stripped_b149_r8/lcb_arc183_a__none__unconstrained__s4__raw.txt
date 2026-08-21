```python
import sys
from itertools import accumulate

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence 
    # (a_1, a_2, ..., a_{NK}) is the i-th, then the sequence
    # (N+1-a_1, N+1-a_2, ..., N+1-a_{NK}) is the (S-i+1)-th.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, we are looking for the sequence that marks the 
    # transition from the first half of all permutations to the second.
    
    # For N=1, the only sequence is (1, ..., 1).
    if N == 1:
        print(*(1 for _ in range(K)))
        return

    # The target index is floor((S+1)/2).
    # We can determine the sequence digit by digit.
    # At each position, we try digits d = 1, 2, ..., N.
    # We skip d if the number of permutations starting with the current prefix 
    # is less than the remaining target index.
    
    # However, S can be massive, so we cannot compute it directly.
    # We observe that the "middle" sequence is the one where we try to 
    # keep the sequence as "balanced" as possible.
    # More formally, the target index is exactly half of the total permutations.
    # This means we want the largest sequence that is <= its complement.
    # The complement of (a_1, ..., a_m) is (N+1-a_1, ..., N+1-a_m).
    # The middle sequence is the one that starts with the digit ceil(N/2).
    # If N is even, the first half starts with 1...N/2, second half with N/2+1...N.
    # The floor((S+1)/2)-th sequence will be the very last sequence that starts 
    # with the digit N/2.
    # If N is odd, the first (S-S_{N/2})/2 sequences start with 1...floor(N/2),
    # then S_{N/2} sequences start with (N+1)/2.
    # The target index falls into the block starting with (N+1)/2.
    
    # Let mid = (N + 1) // 2.
    # If N is even, the target is the last sequence starting with N/2.
    # If N is odd, the target is the "middle" of the sequences starting with (N+1)/2.
    
    # A simpler observation: the target sequence is the one that is 
    # lexicographically the largest among those that are <= their complement.
    # This is achieved by:
    # 1. Filling the first half of the sequence to be as small as possible
    #    while staying "below" the complement, then the second half as large as possible.
    # Actually, the property is: the target sequence is the one that 
    # starts with the digit 'mid = (N+1)//2', and then follows a specific 
    # pattern of being the "last" of the first half.
    
    # Correct logic for floor((S+1)/2)-th:
    # It is the sequence that is the lexicographical maximum of all sequences 
    # A such that A <= complement(A).
    # To maximize A such that A <= complement(A):
    # We want the first index i where A_i != complement(A)_i to have A_i < complement(A)_i,
    # and for all j < i, A_j = complement(A)_j.
    # But A_j = N+1 - A_j implies 2*A_j = N+1. This is only possible if N is odd and A_j = (N+1)/2.
    # If N is even, A_1 must be < complement(A)_1 for the sequence to be in the first half.
    # To maximize this, A_1 should be N/2. Then we want the largest possible sequence
    # starting with N/2.
    # If N is odd, A_1 can be (N+1)/2. Then we look at A_2 and its complement.
    
    # General rule:
    # While we have counts of numbers remaining:
    # If we can pick a number 'd' such that d < N+1-d, we want to pick the largest such 'd'
    # to stay in the first half but be as large as possible.
    # If we pick d < N+1-d, then to maximize the sequence, all subsequent 
    # elements should be as large as possible (descending order).
    # If we pick d = N+1-d (only possible if N is odd), we move to the next position.
    
    # Implementation:
    # We maintain counts of each number 1...N.
    # We try to pick d = (N+1)//2. 
    # If N is even, d = N/2 is the largest digit such that d < N+1-d.
    # If N is odd, d = (N+1)/2 is the only digit such that d = N+1-d.
    
    # Let's refine:
    # If N is even:
    # The first N/2 digits (1 to N/2) start the first half of S.
    # The target is the last sequence starting with N/2.
    # That is: N/2 (K times), then N, N-1, ..., 1 (each K times) - NO.
    # The last sequence starting with N/2 is:
    # N/2, then all remaining digits sorted descending.
    # Remaining: 1(K), 2(K), ..., N/2(K-1), N/2+1(K), ..., N(K).
    # Sorted descending: N(K), N-1(K), ..., 1(K), then the remaining N/2(K-1).
    # Wait, the remaining digits are just the multiset. Sorted descending is the largest.
    
    # If N is odd:
    # The sequences starting with 1... (N-1)/2 are the first half.
    # The sequences starting with (N+1)/2 are the middle.
    # The target is the middle of the sequences starting with (N+1)/2.
    # This is a recursive problem: find the middle sequence of the remaining multiset.
    
    # Let's use the property: the target sequence A satisfies A <= complement(A) 
    # and is the maximum such sequence.
    # 1. Find the first index i where we can't pick d = (N+1)/2.
    # 2. At that index i, pick the largest d < (N+1)/2.
    # 3. Fill the rest of the sequence greedily with the largest available numbers.
    # 4. For the indices j < i, we picked (N+1)/2.
    # 5. For the indices j > i, we fill greedily.
    # 6. But we must ensure the total counts are K.
    
    # Let's trace N=3, K=3. Target: 2 2 2 1 3 3 3 1 1 (Sample 4)
    # i=1: d=2. Count(2)=3. OK.
    # i=2: d=2. Count(2)=2. OK.
    # i=3: d=2. Count(2)=1. OK.
    # i=4: d=2. Count(2)=0. Cannot pick 2.
    # Largest d < 2 is 1. Pick d=1.
    # Remaining: 1(2), 3(3).
    # Fill remaining greedily descending: 3, 3, 3, 1, 1.
    # Result: 2, 2, 2, 1, 3, 3, 3, 1, 1. Matches sample!
    
    # Trace N=2, K=2. Target: 1 2 2 1 (Sample 1)
    # i=1: d=(2+1)//2 = 1.
    # Is d < N+1-d? 1 < 2. Yes.
    # Since we found a d < N+1-d, we pick the largest such d (which is 1),
    # then fill the rest descending.
    # Remaining: 1(1), 2(2).
    # Descending: 2, 2, 1.
    # Result: 1, 2, 2, 1. Matches sample!
    
    # Trace N=6, K=1. Target: 3 6 5 4 2 1 (Sample 3)
    # i=1: d=(6+1)//2 = 3.
    # 3 < 6+1-3 (3 < 4). Yes.
    # Largest d < 3.5 is 3.
    # Remaining: 1(1), 2(1), 4(1), 5(1), 6(1).
    # Descending: 6, 5, 4, 2, 1.
    # Result: 3, 6, 5, 4, 2, 1. Matches sample!

    # Algorithm:
    # 1. If N is even:
    #    First element is N//2.
    #    Remaining elements are (1..N) with counts (K, K, ..., K-1, ..., K)
    #    sorted descending.
    # 2. If N is odd:
    #    We can pick (N+1)//2 up to K times.
    #    After K times, we must pick a digit d < (N+1)//2.
    #    The largest such d is (N-1)//2.
    #    Then fill the rest descending.
    #    Wait, if N is odd, we can pick (N+1)//2 for the first K positions.
    #    Then we are left with digits 1... (N-1)//2 and (N+3)//2 ... N.
    #    The "middle" of these is the one that starts with the largest digit 
    #    that is less than its complement.
    #    The complement of d is N+1-d.
    #    For d < (N+1)//2, the complement is > (N+1)//2.
    #    So we pick the largest d < (N+1)//2, which is (N-1)//2.
    #    Then fill the rest descending.

    # Corrected Logic:
    # The target sequence is the largest sequence A such that A <= complement(A).
    # To maximize A:
    # We want A_i = complement(A)_i for as many i as possible.
    # A_i = N+1 - A_i  => 2*A_i = N+1 => A_i = (N+1)/2.
    # This is only possible if N is odd.
    # If N is odd, we can have A_i = (N+1)/2 for at most K positions.
    # Once we can no longer pick (N+1)/2, we must pick A_i < complement(A)_i
    # to ensure A < complement(A) (since we want the largest A <= complement(A)).
    # The largest A_i < complement(A)_i is the largest available digit d 
    # such that d < N+1-d.
    # After that, to maximize A, we fill the remaining positions with the 
    # largest available digits in descending order.

    mid_val = (N + 1) // 2
    
    # If N is even, mid_val is N//2. 
    # Since mid_val < N + 1 - mid_val (N/2 < N/2 + 1),
    # the first digit will be N//2, and the rest descending.
    # If N is odd, mid_val is (N+1)//2.
    # Since mid_val == N + 1 - mid_val, we can pick it K times.
    # Then we must pick a digit d < N + 1 - d.
    
    # Let's implement this.
    counts = {i: K for i in range(1, N + 1)}
    result = []
    
    # 1. Fill (N+1)//2 as long as N is odd and we have counts
    if N % 2 != 0:
        for _ in range(K):
