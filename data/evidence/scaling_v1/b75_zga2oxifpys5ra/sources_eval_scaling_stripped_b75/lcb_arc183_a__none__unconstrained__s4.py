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
    # Let target = (S + 1) // 2
    
    # To avoid computing S explicitly (which is huge), we use the property of 
    # lexicographical symmetry. The "middle" sequence is related to the 
    # complement of the sequence.
    # If a sequence is A = (a1, a2, ..., am), its complement is 
    # A' = (N + 1 - a1, N + 1 - a2, ..., N + 1 - am).
    # The lexicographical order of A is the reverse of the order of A'.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, for S sequences, the floor((S+1)/2)-th sequence is the 
    # one that, when compared to its complement, is the smaller of the two 
    # in the middle pair, or the unique middle one.
    
    # A known property for this specific problem (multiset permutations) 
    # is that the middle sequence is achieved by filling the sequence 
    # greedily from both ends or by observing that the target index 
    # corresponds to the sequence that is "lexicographically balanced".
    
    # For N, K, the floor((S+1)/2)-th sequence is:
    # For i from 1 to N:
    #   The first K occurrences of i are placed such that they are 
    #   balanced around the center.
    # Actually, a simpler construction for the middle sequence of 
    # multiset permutations is:
    # The sequence is formed by blocks. 
    # For N=3, K=3: 2 2 2 1 3 3 3 1 1
    # This looks like: 
    # Value (N+1)//2 repeated K times, 
    # then (N+1)//2 - 1 repeated K times, ..., 1 repeated K times,
    # then (N+1)//2 + 1 repeated K times, ..., N repeated K times,
    # then the remaining 1s, 2s...
    
    # Correct logic for the middle sequence of permutations of a multiset:
    # The middle sequence is the one that is "lexicographically" 
    # the median. For a symmetric distribution, this is the sequence 
    # that starts with the median element.
    
    # Let's use the property: the middle sequence is the one where we 
    # place the elements in the order:
    # mid, mid-1, ..., 1, mid+1, mid+2, ..., N
    # But each element must appear K times.
    # The pattern for the middle sequence is:
    # 1. Start with the middle value m = (N + 1) // 2.
    # 2. Place m, K times.
    # 3. Place m-1, K times; m-2, K times ... 1, K times.
    # 4. Place m+1, K times; m+2, K times ... N, K times.
    # 5. Wait, the sample 3 (N=6, K=1) gives 3 6 5 4 2 1.
    # Let's analyze Sample 3: N=6, K=1. S=720. Target=360.
    # Sequence: 3 6 5 4 2 1.
    # Sample 4: N=3, K=3. S=1680/6=280. Target=140.
    # Sequence: 2 2 2 1 3 3 3 1 1.
    
    # The pattern is:
    # The middle element m = (N + 1) // 2.
    # The sequence starts with m repeated K times.
    # Then it follows with the remaining elements in a specific order.
    # For N=6, K=1: m=3. Sequence: 3, (6, 5, 4, 2, 1)
    # For N=3, K=3: m=2. Sequence: 2, 2, 2, (1, 3, 3, 3, 1, 1)
    # Looking at Sample 4: 2 2 2 | 1 | 3 3 3 | 1 1
    # It seems to be: 
    # m (K times), then 1 (1 time), then N (K times), then N-1 (K times)...
    # Actually, the pattern is:
    # The middle sequence is the one that is "self-complementary" in a sense.
    # The construction is:
    # For i from 0 to N-1:
    #   If i is even, we handle the "middle" and "outwards".
    # The simplest way to generate the floor((S+1)/2)-th sequence:
    # It is the sequence that starts with m = (N+1)//2, 
    # then follows with the lexicographically largest sequence 
    # possible using the remaining elements, but mirrored.
    
    # Re-evaluating Sample 3: 3 6 5 4 2 1
    # N=6, K=1. m=3. Remaining: {1,2,4,5,6}. 
    # Sorted descending: 6, 5, 4, 2, 1.
    # Result: 3, 6, 5, 4, 2, 1.
    
    # Re-evaluating Sample 4: 2 2 2 1 3 3 3 1 1
    # N=3, K=3. m=2. Remaining: {1,1,1, 3,3,3}.
    # This is not just descending. 
    # 2 2 2 | 1 | 3 3 3 | 1 1
    # It's 2(K), 1(1), 3(K), 1(K-1).
    
    # The general rule for the middle sequence of a multiset permutation:
    # The sequence is:
    # m (K times), then 1 (1 time), then N (K times), then N-1 (K times)...
    # until we reach m+1, then we fill the remaining 1s, 2s...
    # Actually, the pattern is:
    # m (K times), then 1 (1 time), then N (K times), then N-1 (K times)...
    # then m+1 (K times), then 1 (K-1 times), then 2 (K times)...
    
    # Let's use the property: the middle sequence is the one that 
    # starts with m = (N+1)//2, and the rest is the 
    # lexicographically largest sequence of the remaining elements,
    # but with the first element of the remainder being the smallest possible.
    
    # Correct Pattern:
    # 1. Start with m = (N+1)//2 repeated K times.
    # 2. Then 1 (one time).
    # 3. Then N, N-1, ..., m+1 (each K times).
    # 4. Then 1 (K-1 times), 2 (K times), ..., m-1 (K times).
    # Wait, Sample 4: 2 2 2 1 3 3 3 1 1.
    # m=2, K=3.
    # 2(3), 1(1), 3(3), 1(2).
    # This matches! 2(3) -> 1(1) -> 3(3) -> 1(2).
    # Let's check Sample 3: N=6, K=1. m=3.
    # 3(1), 1(1), 6(1), 5(1), 4(1), 1(0), 2(1).
    # Result: 3 1 6 5 4 2. 
    # But Sample 3 says 3 6 5 4 2 1.
    # My pattern is wrong. Let's look at Sample 3 again: 3 6 5 4 2 1.
    # That is: m(K), then N, N-1, ..., m+1, then m-1, m-2, ..., 1.
    # For Sample 4: N=3, K=3. m=2.
    # 2(3), 3(3), 1(3). 
    # That would be 2 2 2 3 3 3 1 1 1.
    # But Sample 4 is 2 2 2 1 3 3 3 1 1.
    
    # Let's try another approach. The middle sequence is the one where 
    # we use the "complement" symmetry.
    # The sequence S is the floor((S+1)/2)-th.
    # This is the sequence that is lexicographically smaller than or 
    # equal to its complement S'.
    # S' is defined by replacing each x with (N + 1 - x).
    # A sequence is "middle" if it's the smallest sequence such that 
    # S <= S'.
    # This happens if at the first index i where S_i != S'_i, we have S_i < S'_i.
    # To make S as large as possible while S <= S', we want S_i = S'_i 
    # for as many i as possible.
    # S_i = S'_i means S_i = N + 1 - S_i, so 2*S_i = N + 1.
    # This is only possible if N is odd and S_i = (N+1)//2.
    
    # If N is even, S_i can never be S'_i. The first index i must have S_i < S'_i.
    # To maximize S, we want S_i to be as large as possible, so S_i = N//2.
    # Then for all subsequent indices, we want S to be as large as possible.
    # So we fill the rest of the sequence in descending order.
    
    # If N is odd, we can have S_i = S'_i for some indices.
    # S_i = (N+1)//2. We can have at most K such indices.
    # To maximize S, we put all K of them at the start.
    # Then the next index must have S_i < S'_i.
    # The largest such S_i is (N+1)//2 - 1 (which is N//2).
    # Then we fill the rest in descending order.
    
    # Let's test this hypothesis:
    # Sample 3: N=6, K=1. N is even. S_1 = 6//2 = 3.
    # Remaining: {1,2,4,5,6}. Descending: 6, 5, 4, 2, 1.
    # Result: 3 6 5 4 2 1. (Matches Sample 3!)
    
    # Sample 4: N=3, K=3. N is odd. S_1...S_K = (3+1)//2 = 2.
    # Remaining: {1,1,1, 3,3,3}.
    # Next index S_{K+1} must be < S'_{K+1}.
    # S'_{K+1} = 3 + 1 - S_{K+1}.
    # S_{K+1} < 4 - S_{K+1} => 2*S_{K+1} < 4 => S_{K+1} < 2.
    # Largest S_{K+1} is 1.
    # Remaining: {1,1, 3,3,3}. Descending: 3, 3, 3, 1, 1.
    # Result: 2 2 2 1 3 3 3 1 1. (Matches Sample 4!)
    
    # Sample 1: N=2, K=2. N is even. S_1 = 2//2 = 1.
    # Remaining: {1, 2, 2}. Descending: 2, 2, 1.
    # Result: 1 2 2 1. (Matches Sample 1!)
    
    # General Algorithm:
    # 1. m = (N + 1) // 2
    # 2. If N is even:
    #    First element is m