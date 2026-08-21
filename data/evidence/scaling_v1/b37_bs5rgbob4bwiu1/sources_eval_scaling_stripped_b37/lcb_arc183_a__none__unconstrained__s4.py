```python
import sys
from functools import reduce
from operator import mul

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is exactly the middle sequence if S is odd, 
    # or the (S/2)-th sequence if S is even.
    # In both cases, for S > 0, this is the sequence at index (S-1)//2 
    # in a 0-indexed sorted list of all good sequences.
    
    # To find the sequence at a specific rank, we determine the elements one by one.
    # For the current position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with the current prefix is 
    # (Remaining Total)! / Product(Remaining Count of each i!)
    
    # Since we need the middle sequence, we can use the property of symmetry.
    # The lexicographical middle of all permutations of a multiset is 
    # the "complement" of the sequence if we map v -> (N + 1 - v).
    # However, the most direct way to find the (S+1)//2-th sequence is to
    # realize that the set of all good sequences is symmetric.
    # If a sequence A is the i-th, then the sequence A' (where A'_j = N + 1 - A_j)
    # is the (S - i + 1)-th.
    # The middle sequence is the one where A is "closest" to its complement.
    
    # For N=1, the only sequence is (1,) * K.
    if N == 1:
        print(*( [1] * K ))
        return

    # We use a helper to calculate the number of permutations of a multiset.
    # Instead of calculating S, we can use the fact that we want the 
    # "median" sequence. A sequence A is the median if it is the 
    # smallest sequence such that the number of sequences <= A is >= S/2.
    
    # Because we cannot compute S directly (it's too large), we use the 
    # property that the middle sequence is the one that, when 
    # transformed by v -> N+1-v, results in the "opposite" rank.
    # The middle sequence A satisfies: A is the first sequence such that
    # A >= complement(A).
    # Actually, a simpler property: the middle sequence is the one 
    # that is lexicographically "half-way". 
    # For a multiset, the middle sequence is simply the one where 
    # we place the elements in a specific balanced way.
    # Specifically, the middle sequence is the one that starts with 
    # the value v such that the number of sequences starting with 
    # 1, ..., v-1 is < S/2 and starting with 1, ..., v is >= S/2.
    
    # Since we cannot compute S, we use the symmetry:
    # The middle sequence is the one that is "self-complementary" in rank.
    # This is achieved by picking the middle value for the first position,
    # and then recursively solving for the remaining.
    # If N is even, the middle two values are N//2 and N//2 + 1.
    # The first element of the (S+1)//2-th sequence is (N+1)//2.
    # Then we distribute the remaining counts.
    
    # Correct logic for the middle sequence of a symmetric distribution:
    # The first element is (N+1)//2. 
    # If N is even, the first element is N//2, but we must be careful.
    # Let's use the property: the middle sequence is the one that 
    # reads the same as its complement reversed? No.
    # The middle sequence is the one that is "centered".
    # For N=2, K=2: S=6. (S+1)//2 = 3rd. 
    # Sequences: 1122, 1212, 1221, 2112, 2121, 2211. 3rd is 1221.
    # For N=3, K=3: S=1680. (S+1)//2 = 840th.
    
    # The middle sequence is the one where we use the values 
    # 1...N in a balanced way.
    # The first element is (N+1)//2 if N is odd.
    # If N is even, the first element is N//2, and we are looking for 
    # the "last" sequence starting with N//2.
    
    # A known property: the middle sequence of all permutations of a multiset
    # is the one that is "lexicographically complementary" to itself.
    # The sequence A is the (S+1)//2-th if A is the smallest sequence 
    # such that A >= complement(A).
    # This means at the first index i where A_i != complement(A)_i, 
    # we must have A_i > complement(A)_i.
    # But we want the smallest such A.
    # This implies A_i = complement(A)_i for as many i as possible.
    # A_i = N + 1 - A_i  => 2*A_i = N + 1.
    # This is only possible if N is odd and A_i = (N+1)//2.
    
    # Let's refine: 
    # If N is even, the first element must be N//2. 
    # To be the (S/2)-th, it must be the largest sequence starting with N//2.
    # To be the largest sequence starting with N//2, the remaining elements
    # must be placed in descending order.
    # If N is odd, the first element is (N+1)//2.
    # We use (N+1)//2 once, then we have a symmetric problem with 
    # (N-1) values and K counts, plus one remaining (N+1)//2.
    
    # Actually, the simplest construction for the middle sequence:
    # 1. Place (N+1)//2 at the first position.
    # 2. Fill the remaining positions by placing the remaining 
    #    elements in a way that they are balanced.
    # The pattern is: 
    # For i from 1 to N:
    #   If i < (N+1)//2: place i at the end.
    #   If i > (N+1)//2: place i at the beginning.
    #   If i == (N+1)//2: place it in the middle.
    # This is getting complex. Let's use the property:
    # The middle sequence is the one that is the "largest" sequence 
    # that is "smaller than or equal to" its own complement.
    # That means it starts with the largest possible value v such that
    # v <= (N+1) - v, which is v = (N+1)//2.
    # Then we fill the rest of the sequence to be as large as possible,
    # provided the overall sequence remains <= its complement.
    
    # Correct construction:
    # The middle sequence is:
    # For i = 1 to N:
    #   If i < (N+1)//2: it appears K times.
    #   If i > (N+1)//2: it appears K times.
    #   If i == (N+1)//2: it appears K times.
    # The sequence is: 
    # (N//2) repeated K times, then (N//2 + 1) repeated K times... 
    # No, that's not it.
    
    # Let's use the property: The middle sequence is the one that 
    # starts with (N+1)//2, and then the remaining are arranged 
    # such that they are the largest possible sequence that is 
    # still "below" the mirror.
    # The mirror of a sequence A is A' where A'_i = N + 1 - A_i.
    # We want the largest A such that A <= A'.
    # This means for the first index i where A_i != A'_i, we need A_i < A'_i.
    # For all j < i, A_j = A'_j, which means A_j = (N+1)/2.
    # So A starts with some number of (N+1)//2, then a value v < (N+1)//2,
    # then the rest of the sequence is as large as possible (descending).
    
    # Let's trace N=2, K=2. S=6, target=3rd.
    # A_1: can it be 1? Yes. If A_1=1, A'_1=2. 1 < 2, so A < A'.
    # To make A the largest such sequence, we fill the rest descending:
    # Remaining: {1:1, 2:2}. Descending: 2, 2, 1.
    # Sequence: 1, 2, 2, 1. (Matches Sample 1!)
    
    # Let's trace N=3, K=3. S=1680, target=840th.
    # A_1: can it be 1? 1 < 3, so A < A'.
    # A_1: can it be 2? 2 == 2, so we check A_2.
    # If A_1=2, we have remaining {1:3, 2:2, 3:3}.
    # For A_2, can it be 2? 2 == 2, check A_3.
    # For A_3, can it be 2? 2 == 2, check A_4.
    # For A_4, can it be 1? 1 < 3, so A < A'.
    # To make A largest, fill the rest descending: 3,3,3,2,2,1,1.
    # Wait, the remaining were {1:3, 2:2, 3:3}. We used one '2' at A_1, A_2, A_3.
    # Remaining: {1:3, 2:0, 3:3}.
    # A_4 must be 1 (since 1 < 3). Then fill descending: 3,3,3,1,1.
    # Sequence: 2, 2, 2, 1, 3, 3, 3, 1, 1. (Matches Sample 4!)
    
    # General Algorithm:
    # 1. Fill A_i = (N+1)//2 for i = 1 to K.
    # 2. If N is even, the first element must be N//2.
    #    Then the remaining elements are filled in descending order.
    #    Wait, if N is even, (N+1)//2 is not an integer.
    #    If N=2, K=2: A_1 = 1. Remaining {1:1, 2:2}. Descending: 2, 2, 1.
    #    If N=4, K=1: S=24, target=12th.
    #    A_1=1: A < A'. A_1=2: 2 < 3, so A < A'.
    #    Largest A starting with 2: 2, 4, 3, 1.
    #    Check: A=(2,4,3,1), A'=(3,1,2,4). A < A' is True.
    #    Is there a larger one? A_1=3: 3 > 2, so A > A'.
    #    So A_1 must be 2.
    
    # Refined Algorithm:
    # 1. The first element A_1 is N // 2.
    # 2. If N is odd, we can have A_1 = (N+1)//2.
    #    Actually, the rule is:
    #    While we can place (N+1)//2, do so.
    #    Then place the largest value v < (N+1)//2.
    #    Then fill the rest descending.
    #    If N is even, (N+1)//2 is not available, so the "middle" value is N//2.
    
    # Let's formalize:
    # The middle value is M = (N + 1) // 2.
    # If N is odd:
    #   The sequence starts with M repeated K times.
    #   Then it places the largest value v < M (which is M-1).
    #   Then it fills the remaining values in descending order