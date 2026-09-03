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

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # Note: S is symmetric. If a sequence is (a1, a2, ..., a_nk), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # Lexicographically, the middle two sequences are the ones that are 
    # "closest" to each other.
    # Specifically, the S/2-th and (S/2 + 1)-th sequences are complements of each other.
    # The floor((S+1)/2)-th sequence is the S/2-th sequence.
    
    # To find the S/2-th sequence:
    # At each position i, we try digits d = 1, 2, ..., N.
    # The number of sequences starting with the current prefix is (RemainingLength)! / Product(RemainingCounts!).
    # If this count is less than our current target rank, we subtract it and move to d+1.
    # Since we want the S/2-th, we can use the property that the sequence is 
    # "just before" the halfway point.
    
    # Because S can be massive, we cannot compute it directly.
    # However, we can determine the sequence digit by digit.
    # For the first digit:
    # If we pick 'd', the number of ways to fill the rest is W(d) = (NK-1)! / (K-1)! (K!)^{N-1}.
    # The total S = (NK)! / (K!)^N.
    # W(d) = S * (K / NK) = S / N.
    # Since all W(d) are equal for d=1...N, the first digit of the S/2-th sequence is:
    # If N=1: 1
    # If N>1: The digit d such that sum_{j=1}^{d-1} W(j) < S/2 <= sum_{j=1}^{d} W(j).
    # Since W(j) are all equal to S/N, this is d = ceil((S/2) / (S/N)) = ceil(N/2).
    # d = (N + 1) // 2.
    
    # After picking the first digit, we need to find the rank within the remaining set.
    # This is tricky because S is too large. 
    # Let's use the symmetry property:
    # The S/2-th sequence is the one immediately preceding the "middle" of the lexicographical list.
    # The middle two sequences are X and X', where X' is X reversed in value (d -> N+1-d).
    # X is the largest sequence that is lexicographically smaller than or equal to its complement.
    # To construct X:
    # For each position i = 1 to NK:
    # We want to pick the largest digit d such that the resulting prefix can still 
    # lead to a sequence <= its complement.
    # Actually, a simpler way:
    # The S/2-th sequence is the one where we try to pick the largest possible digits 
    # for the first half of the sequence, but stay under the 50% mark.
    # Correct logic: The S/2-th sequence is the one that is "just smaller" than its complement.
    # This means at the first index i where the sequence A differs from its complement A', 
    # we must have A[i] < A'[i].
    # To make A the largest such sequence, we want A[i] to be as large as possible, 
    # but since A[i] < A'[i] and A'[i] = N + 1 - A[i], we need A[i] < (N+1)/2.
    # So A[i] = (N-1)//2 if we want to stay in the first half? No.
    
    # Let's reconsider:
    # The sequence A is the S/2-th if we can partition the set of all sequences into pairs (A, A')
    # where A' is the complement. A < A' if at the first index i where they differ, A[i] < A'[i].
    # A[i] < N + 1 - A[i]  => 2*A[i] < N + 1.
    # To find the largest A such that A < A', we want A to match A' as long as possible.
    # But A and A' cannot be identical unless A[i] = (N+1)/2 for all i.
    # If N is even, A and A' always differ.
    # If N is odd, A can be equal to A' (the sequence of all (N+1)//2).
    
    # To get the S/2-th sequence:
    # For i = 1 to NK:
    # We want to pick the largest d such that the number of sequences starting with 
    # (prefix, d) is not enough to push us past S/2.
    # This is still hard. Let's use the property:
    # The S/2-th sequence is the one where we fill the sequence such that it is 
    # "as large as possible" while remaining "smaller than its complement".
    # This means for the first index i where A[i] != (N+1)/2, we must have A[i] < (N+1)/2.
    # To maximize this, we want A[i] to be (N-1)//2 at the latest possible index i,
    # and for all j < i, A[j] = (N+1)//2.
    # But we are constrained by the count K.
    # We can only have A[j] = (N+1)//2 for K times.
    # So for the first K indices, we can have (N+1)//2.
    # But we want the largest sequence < its complement.
    # The complement of (A_1, ..., A_{NK}) is (N+1-A_1, ..., N+1-A_{NK}).
    # To make A largest but A < A', we need the first index i where A_i != N+1-A_i to satisfy A_i < N+1-A_i.
    # To maximize A, we want this index i to be as large as possible.
    # The indices where A_i = N+1-A_i are those where A_i = (N+1)/2.
    # This can happen at most K times.
    # So for i = 1 to K, we can set A_i = (N+1)//2.
    # Then at i = K+1, we must pick A_{K+1} < (N+1)/2 to ensure A < A'.
    # To maximize A, we pick A_{K+1} = (N-1)//2.
    # Then for the remaining positions, we fill them with the largest available numbers.
    
    # Let's refine:
    # 1. If N is even:
    #    The first index i where A_i != N+1-A_i must have A_i < N+1-A_i.
    #    To maximize A, we want A_i to be as large as possible for as long as possible.
    #    But we can't have A_i = N+1-A_i.
    #    So at the first index i, we must pick A_i <= N//2.
    #    To maximize A, we want to push this constraint to the latest possible index.
    #    However, we have to pick K copies of each number.
    #    The largest possible sequence is (N, N, ..., N, N-1, ..., 1, 1).
    #    The S/2-th sequence is the one that is "just below" the middle.
    #    Due to symmetry, if we sort all sequences, the first half are those that start with 
    #    a digit < (N+1)/2, or start with (N+1)/2 and eventually have a digit < (N+1)/2.
    #    The largest sequence in the first half is:
    #    - Fill the first K positions with (N+1)//2 (if N is odd).
    #    - At the first position where we cannot pick (N+1)//2, or we decide to "drop" to the lower half,
    #      we pick the largest possible digit d < (N+1)/2.
    #    - Then fill all remaining positions with the largest available digits.
    
    # Correct Logic for S/2-th:
    # We want the largest sequence A such that A < complement(A).
    # A < complement(A) iff at the first index i where A_i != N+1-A_i, we have A_i < N+1-A_i.
    # To maximize A:
    # For i = 1 to NK:
    #   We want to pick the largest d such that there exists some j > i where A_j < N+1-A_j,
    #   and for all m < j, A_m = N+1-A_m.
    #   Wait, if we pick d > (N+1)/2, then A > complement(A) immediately.
    #   If we pick d < (N+1)/2, then A < complement(A) immediately.
    #   If we pick d = (N+1)/2, we move to the next index.
    
    # So, to get the largest A < complement(A):
    # 1. Fill as many (N+1)//2 as possible (up to K).
    # 2. At the first index where we can't or must break the symmetry to stay < complement,
    #    we pick the largest d < (N+1)/2.
    # 3. Then fill the rest of the sequence with the largest available numbers.
    
    # But we must be careful: if we pick d < (N+1)/2 at index i, we have already 
    # guaranteed A < complement(A). To maximize A, we should do this as late as possible.
    # The latest possible index to pick d < (N+1)/2 is when we are forced to because 
    # all remaining available numbers are < (N+1)/2.
    # But we can just pick d = (N+1)//2 for the first K times, then at index K+1, 
    # we must pick a number. If we pick d > (N+1)//2, we might exceed the middle.
    # Actually, the simplest way to describe the S/2-th sequence:
    # It is the sequence that matches the "middle" sequence (the one that is its own complement, if it exists)
    # for as long as possible, and then takes the smaller value.
    
    # Let mid = (N + 1) / 2.
    # The sequence is:
    # For i = 1 to NK:
    # If we can pick d = mid (i.e., N is odd and count[mid] > 0), we do it? 
    # No, that would lead to the middle sequence.
    # To be the largest sequence < complement:
    # We want to pick d > mid as much as possible, but we must eventually pick some d < mid
    # before we pick any d > mid.
    # That's not right. If we pick d > mid at index 1, then A > complement(A).
    # So we must pick d <= mid for the first index i where A_i != complement(A_i).
    # To maximize A, we want A_i = mid for as long as possible, then A_i < mid, then A_j = max available.
    
    # Let's trace Sample 1: N=2, K=2. S=6, S/2=3.
    # mid = 1.5.
    # i=1: d=1 (since 1 < 1.5). Remaining: {1:1, 2:2}.
    # Now we can pick anything. To maximize, pick d=2, 2, 1.
    # Result: 1 2 2 1. Correct.
    
    # Sample 3: N=6, K=1. S=720, S/2=360.
    # mid = 3.5.
    # i=1: d=3 (largest d < 3.5). Remaining: {1:1, 2:1, 4:1, 5:1, 6:1}.
    # Now maximize: 6, 5, 4, 2, 1.
    # Result: 3 6 5 4 2 1. Correct.
    
