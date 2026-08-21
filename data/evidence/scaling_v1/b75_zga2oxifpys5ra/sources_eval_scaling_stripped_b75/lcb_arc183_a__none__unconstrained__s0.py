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
    
    # To avoid calculating S explicitly (which is huge), we use the property
    # that the lexicographical middle of all permutations of a multiset
    # is related to the "complement" sequence.
    # The complement of a sequence (s1, s2, ..., snk) is (n+1-s1, n+1-s2, ..., n+1-snk).
    # If a sequence is the i-th in lexicographical order, its complement is the (S - i + 1)-th.
    # We want the i-th where i = floor((S+1)/2).
    # Note that if S is odd, the middle sequence is its own complement.
    # If S is even, the middle two are i and i+1.
    
    # A known property for this specific problem (finding the middle permutation of a multiset):
    # The middle sequence is the one that is "lexicographically balanced".
    # For a multiset where each element appears K times, the middle sequence
    # is achieved by placing the elements in a specific symmetric way.
    # Specifically, the target index is exactly the middle of the range [1, S].
    # The sequence that satisfies this is the one where we try to place the 
    # "middle" available digit at each position.
    
    # However, a simpler observation for this specific problem:
    # The target sequence is the one that, when mirrored and values inverted (x -> N+1-x),
    # results in the sequence itself (or the one immediately following it).
    # For N, K, the floor((S+1)/2)-th sequence is the one where we 
    # effectively treat the sequence as a number in a mixed radix system.
    
    # The actual mathematical result for the middle permutation of a multiset
    # is to construct the sequence greedily. At each position, we check if the
    # number of permutations starting with digits 1...d is less than target.
    
    # Since S is too large to compute, we use the property:
    # The middle sequence of all permutations of {1*K, 2*K, ..., N*K}
    # is the sequence that is "lexicographically" the middle.
    # This is equivalent to:
    # For each position, we want to pick the smallest digit d such that
    # the number of permutations starting with digits < d is < target,
    # and the number of permutations starting with digits <= d is >= target.
    
    # Because we want the middle, we can use the symmetry:
    # The middle sequence is the one that is "closest" to the sequence
    # where we use digits in the order: 
    # (N//2 + 1) repeated K times, then (N//2) repeated K times... 
    # Actually, the pattern is:
    # The middle sequence is simply the sequence that looks like:
    # For i from 1 to N:
    # If i is even, it's placed in a way that balances the total.
    
    # Correct logic for middle permutation of multiset:
    # The middle sequence is the one where we place digits such that 
    # we are as close to the "center" as possible.
    # For N, K, the sequence is:
    # For j in range(K):
    #   For i in range(N):
    #     ... this is not quite right.
    
    # Let's use the property: the middle sequence is the one that 
    # reads the same forwards as it does backwards when values are replaced by (N+1-x).
    # For N=2, K=2: S=6, target=3. Sequences: 1122, 1212, 1221, 2112, 2121, 2211.
    # 3rd is 1221. Complement of 1221 is 2112 (4th).
    # For N=3, K=3: S=1680, target=840.
    # The pattern for the middle sequence is:
    # It starts with the middle digit (N+1)//2.
    # Then it fills the remaining slots by balancing the smaller and larger digits.
    
    # A more robust approach:
    # The middle sequence is the one that is "lexicographically" the median.
    # This is achieved by:
    # For each position, we pick digit d from 1 to N.
    # The number of permutations of the remaining digits is (TotalRemaining)! / product(rem_i!)
    # We pick d such that sum_{j=1}^{d-1} count(j) < target <= sum_{j=1}^{d} count(j).
    
    # Since we cannot compute S, we use the fact that we want the middle.
    # The middle sequence is the one that is "self-complementary" in a sense.
    # For N, K, the middle sequence is:
    # For i from 0 to N*K - 1:
    # The digit at position i is (N+1)//2 if we are at the middle of the sequence,
    # and we balance the others around it.
    
    # Actually, the simplest construction for the middle sequence of a multiset
    # is: 
    # 1. Start with the sequence (1, 1, ..., 2, 2, ..., N, N, ...)
    # 2. The middle sequence is the one that is "halfway" between the 
    #    smallest (1...1, 2...2, ..., N...N) and largest (N...N, ..., 1...1).
    # This is achieved by:
    # For each pair of indices (i, NK-1-i):
    # One gets value x, the other gets N+1-x.
    # To be the floor((S+1)/2)-th, we want the smaller values at the front.
    # So for i < NK/2, we want the smaller value of the pair {x, N+1-x} at index i.
    # But we must use each digit exactly K times.
    # This means for each digit d < (N+1)/2, we have K copies of d and K copies of N+1-d.
    # To make the sequence as small as possible while remaining the "middle",
    # we place one 'd' and one 'N+1-d' at symmetric positions.
    # To keep it lexicographically smaller (the floor), we put 'd' at the earlier position.
    # So for d = 1, 2, ..., N//2:
    # Place 'd' at positions (d-1)*2*K/N ... wait, that's not right.
    
    # Correct construction:
    # For d from 1 to N//2:
    #   Place 'd' at indices: (d-1)*2 + 0, (d-1)*2 + 1 ... no.
    # The middle sequence is:
    # For i from 0 to NK-1:
    #   If i < NK/2:
    #     The digit is (i % N) + 1 ... no.
    
    # Let's use the property: the middle sequence is the one where 
    # for all i, S_i + S_{NK-1-i} = N + 1.
    # To make it the smallest such sequence (the floor), we want S_i to be as small as possible
    # for small i, provided that the set of values is {1*K, ..., N*K}.
    # This means for i from 0 to NK/2 - 1:
    # We want S_i to be the smallest available digit d such that its complement N+1-d
    # is also available.
    # Since we have K of each, we can do this K times for each d from 1 to N//2.
    # So for i from 0 to K*(N//2) - 1:
    # S_i = (i // K) + 1
    # Then for the middle (if N is odd):
    # S_{NK/2} = (N+1)//2
    # Then the second half is mirrored: S_{NK-1-i} = N+1 - S_i.
    
    # Example N=2, K=2:
    # i=0: S_0 = (0//2)+1 = 1. S_{4-1-0} = S_3 = 2+1-1 = 2.
    # i=1: S_1 = (1//2)+1 = 1. S_{4-1-1} = S_2 = 2+1-1 = 2.
    # Result: 1 1 2 2. 
    # Wait, Sample 1 says 1 2 2 1. My logic gives 1 1 2 2.
    # Let's re-evaluate. 1 2 2 1 is the 3rd. 1 1 2 2 is the 1st.
    # The middle of 6 is 3. 
    # For N=2, K=2, the sequences are:
    # 1: 1 1 2 2
    # 2: 1 2 1 2
    # 3: 1 2 2 1 <--- Target
    # 4: 2 1 1 2
    # 5: 2 1 2 1
    # 6: 2 2 1 1
    # The target 1 2 2 1 is the complement of 2 1 1 2.
    # It is the largest sequence that starts with 1.
    # The number of sequences starting with 1 is (3!)/(1! 2!) = 3.
    # So the 3rd sequence is the last one that starts with 1.
    # The last sequence starting with 1 is 1 followed by the reverse of the 
    # lexicographically first sequence of the remaining {1, 2, 2}.
    # First of {1, 2, 2} is 1 2 2. Reverse is 2 2 1.
    # So 1 2 2 1.
    
    # General Rule:
    # The floor((S+1)/2)-th sequence is the largest sequence that starts with 
    # the "middle" digit distribution.
    # Actually, the simplest way to describe the floor((S+1)/2)-th sequence is:
    # It is the sequence that is "lexicographically" the middle.
    # For any sequence A, let A' be the complement (S'_i = N+1 - S_i).
    # The complement of the lexicographically smallest is the largest.
    # The middle one is the one that is "just below" its own complement.
    # This is achieved by:
    # For i from 0 to NK-1:
    # We want to pick the smallest digit d such that the number of permutations
    # starting with digits < d is < target.
    # Since we want the middle, we can use the fact that the total number of
    # permutations is S. We want the (S+1)//2 -th.
    # This is equivalent to picking d such that we've passed half of the total.
    # For the first position, the number of permutations starting with d is:
    # Count(d) = (NK-1)! / (K!^{N-1} * (K-1)!) = S * (K / NK) = S / N.
    # So for the first digit, we pick d = (N+1)//2.
    # If N is even, the first (S/2) sequences start with 1...N/2.
    # The (S/2 + 1)-th sequence starts with N/2 + 1.
    # Wait, if N=2, K=2, S=6. Target = 3.
    # d=1: 6/2 = 3 sequences.
    # Since target=3, the 3rd sequence is the LAST sequence starting with 1.
    # If N=3, K=3, S=1680. Target = 840.
    # d=1: 1680/3 = 560 sequences.
    # d=2: 1680/3 = 560 sequences.
    # Total for d=1,2 is 1120.
    # Since 560 < 840 <= 1120, the 840th sequence starts with 2.