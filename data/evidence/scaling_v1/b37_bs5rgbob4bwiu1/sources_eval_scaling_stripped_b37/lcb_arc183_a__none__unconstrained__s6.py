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
    # In both cases, it is the sequence at index (S-1)//2 in a 0-indexed sorted list.
    
    # To find the sequence at a specific rank, we determine the elements one by one.
    # For the first position, we try candidates v = 1, 2, ..., N.
    # The number of sequences starting with v is (NK-1)! / (K!^{N-1} * (K-1)!).
    # This simplifies to: Total_Sequences * (K / NK).
    
    # Instead of calculating S, we can use the symmetry of lexicographical order.
    # The "middle" sequence of a symmetric set of permutations is the one that
    # is its own "complement" (where complement of v is N - v + 1) reversed.
    # However, the problem asks for a specific rank.
    # The total number of sequences S is symmetric. The sequence at rank (S+1)//2
    # is the one that, when you replace each x with (N+1-x) and reverse the sequence,
    # you get the sequence at rank S - ((S+1)//2) + 1.
    
    # For N=1, the only sequence is (1,)*K.
    if N == 1:
        print(*( [1]*K ))
        return

    # The rank we are looking for is (S+1)//2.
    # Let's use the property: the middle sequence of all permutations of a multiset
    # is the one that is "lexicographically central".
    # For a multiset, the sequence at rank (S+1)//2 is the one where we 
    # greedily pick the smallest possible value such that the number of 
    # sequences lexicographically smaller than it is < (S+1)//2.
    
    # Since we cannot compute S (it's too large), we use the fact that
    # the middle sequence is the one that is "self-complementary" in a sense.
    # Specifically, if we have a sequence A, its complement A' is 
    # (N+1 - A_i). The reverse of A' is (A'_m, ..., A'_1).
    # The map A -> reverse(A') is an involution that reverses lexicographical order.
    # Therefore, the fixed point of this map (if it exists) or the two middle elements
    # are related by this symmetry.
    
    # The sequence at rank (S+1)//2 is the one that is "lexicographically" 
    # the middle one. This is achieved by picking the median value for the 
    # first position, and so on.
    # For a balanced distribution, the middle sequence is the one that 
    # starts with the value (N+1)//2, but we must be careful with parity.
    
    # Correct logic for "middle" sequence of multisets:
    # The sequence is the one that is its own complement-reverse.
    # A_i = N + 1 - A_{NK - i + 1}
    # For the first half of the sequence (i = 1 to NK//2):
    # We want the smallest A_i such that we don't exceed the halfway mark.
    # This effectively means we want the sequence that is "central".
    # The central sequence is constructed by:
    # For i = 1 to NK//2:
    # Try v = 1, 2, ..., N.
    # If we pick v, we must also pick N+1-v at position NK-i+1.
    # This reduces the problem to N, K-1 (for v and N+1-v) and NK-2 length.
    
    # Wait, the simplest way to find the (S+1)//2-th sequence is to realize
    # that it is the sequence that is "lexicographically" the middle.
    # For N=2, K=2: S=6. (6+1)//2 = 3rd.
    # Sequences: 1122, 1212, 1221, 2112, 2121, 2211. 3rd is 1221.
    # Note: 1221 is the reverse-complement of itself: 1->2, 2->1, rev(2112) = 2112? No.
    # Complement of 1221 is 2112. Reverse of 2112 is 2112.
    # Actually, the middle sequence is the one that is "equal" to its 
    # complement-reverse: A_i = N + 1 - A_{NK-i+1}.
    # To make it the smallest such sequence (since we want the floor),
    # we greedily pick the smallest v for A_i, provided that the 
    # remaining counts allow for a valid complement-reverse sequence.
    
    # For i = 1 to NK//2:
    # Try v = 1, 2, ..., N:
    # If we pick v, we must place N+1-v at NK-i+1.
    # This is possible if:
    # 1. Count of v > 0
    # 2. Count of N+1-v > 0 (if v != N+1-v)
    # 3. If v == N+1-v, we need at least 2 of them (unless it's the exact middle element).
    
    # Let's refine:
    # We want the smallest sequence A such that A = reverse(complement(A)).
    # For i = 1 to NK // 2:
    #   For v = 1 to N:
    #     If v can be placed at i and N+1-v at NK-i+1:
    #       Check if a valid sequence can be formed with remaining counts.
    #       (A valid sequence can always be formed if counts are non-negative).
    #       Since we want the (S+1)//2-th, and the map A -> reverse(complement(A))
    #       is an order-reversing involution, the fixed point is the middle.
    #       If S is even, there are two middle sequences; (S+1)//2 is the smaller one.
    #       The smaller one is the one that is "lexicographically" smaller than its 
    #       complement-reverse.
    
    # Actually, the property is: the sequence A is the (S+1)//2-th if 
    # A <= reverse(complement(A)) and for any B < A, B > reverse(complement(B)).
    # This is satisfied by the sequence that is "lexicographically smallest" 
    # among those where A <= reverse(complement(A)).
    # This is simply the sequence that is "half-way".
    # The construction:
    # For i = 1 to NK // 2:
    #   Try v = 1, 2, ..., N:
    #     If we pick v, we are forced to pick N+1-v at the mirror position.
    #     Does this choice keep us in the "lower half" or "upper half"?
    #     If v < N+1-v, we are definitely in the lower half.
    #     If v > N+1-v, we are definitely in the upper half.
    #     If v == N+1-v, we depend on the remaining sequence.
    
    # Correct Greedy Strategy for (S+1)//2-th:
    # For i = 1 to NK // 2:
    #   For v = 1 to N:
    #     If v < N+1-v:
    #       If count[v] > 0 and count[N+1-v] > 0:
    #         We can pick v. This will always be in the lower half.
    #         But we want the LARGEST such sequence that is still <= its complement-reverse.
    #         Wait, the question asks for the (S+1)//2-th.
    #         For N=2, K=2, S=6, rank 3. Sequences: 1122, 1212, 1221 | 2112, 2121, 2211.
    #         The 3rd is 1221.
    #         For i=1: v=1. 1 < 2+1-1=2. So 1 is the smallest.
    #         If we pick A_1 = 1, then A_4 = 2. Remaining: {1, 2}.
    #         For i=2: v=1. 1 < 2. If A_2=1, then A_3=2. Sequence 1122.
    #         If A_2=2, then A_3=1. Sequence 1212.
    #         Wait, 1221 is the 3rd. My manual trace is wrong.
    #         1122 (1), 1212 (2), 1221 (3).
    #         In 1221: A_1=1, A_4=2; A_2=2, A_3=1.
    #         At i=1, v=1. At i=2, v=2.
    
    # The logic: To get the (S+1)//2-th, we want the largest sequence A 
    # such that A <= reverse(complement(A)).
    # For i = 1 to NK // 2:
    #   For v = 1 to N:
    #     If v < N+1-v:
    #       If count[v] > 0 and count[N+1-v] > 0:
    #         # This v makes A < reverse(complement(A)) regardless of the rest.
    #         # To make A as large as possible, we should try the largest such v.
    #         # But we can't just pick the largest v, because a smaller v 
    #         # might allow a much larger sequence later.
    #         # Actually, if v < N+1-v, then any sequence starting with v 
    #         # is smaller than its complement-reverse.
    #         # So we want the largest v such that v < N+1-v and we can complete it.
    #         # Then we fill the rest of the sequence as largely as possible.
    #         # OR, if we pick v == N+1-v, we recursively check the middle.
    
    # Let's use the property: The (S+1)//2-th sequence is the one that is 
    # "lexicographically" the middle.
    # For i = 1 to NK:
    #   Try v = 1, 2, ..., N:
    #     Count how many sequences start with (prefix + v).
    #     If this count is < (S+1)//2, subtract it from rank and try v+1.
    #     If this count >= (S+1)//2, then the element is v.
    
    # Since we can't compute S, we use the fact that the number of sequences
    # starting with v is S * (count[v] / remaining_length).
    # Rank R = (S+1)//2.
    # For i = 1 to NK:
    #   For v = 1 to N:
    #     Ways(v) = (remaining_length - 1)! / product(count[j]!) * count[v]
    #     If R <= Ways(v): break
    #     Else: R -= Ways(v)
    
    # We can maintain R as a fraction R/S.
    # Initial R/S = 1/2.
    # For v = 1 to N:
    #   Prob(v) = count[v] / remaining_length
    #   If R/S <= Prob(v):
    #     # v is the element.
    #     # New R/S = (R/S) / Prob(v)
    #     # But we must be careful: R is the rank among sequences starting with v.
    #     # New R = R - (sum of Prob(u) for u < v) * S
    #     # New S = Prob(v) * S
    #     # New R/S = (R/S - sum_{u<v} Prob(u)) / Prob(v)
    
    # Let's use a Fraction to keep track of the relative