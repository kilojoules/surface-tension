The constraint to avoid explicit loops forces a transition from imperative programming to a functional paradigm, emphasizing the use of higher-order functions and recursion. This approach encourages the use of immutable-style transformations and declarative logic, leveraging Python's powerful sequence processing tools. To solve this problem, I will use `functools.reduce` to simulate the iterative process of building the sequence and a helper function to calculate the number of permutations of a multiset.

```python
import sys
from math import factorial
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    K = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2
    
    # Precompute factorials for multiset permutation formula
    # Since N, K <= 500, we need factorials up to 250,000.
    # However, we only need to calculate the count of sequences starting with a certain prefix.
    # The number of ways to arrange the remaining elements is:
    # (Total remaining)! / Product( (remaining count of each digit)! )
    
    # To avoid huge factorial calculations in every step, we can use a 
    # more efficient way to track the target index.
    
    # Let's define a function to calculate multiset permutations
    # Using a cache for factorials to speed up
    fact = [1] * (N * K + 1)
    # We cannot use a loop to precompute factorials. 
    # We use reduce to build the factorial list.
    # Note: This is a trick to bypass the 'no loop' constraint.
    # We create a list of indices and use reduce to accumulate products.
    # But wait, we can't use a loop to fill the list. 
    # Let's use a recursive-like approach via reduce.
    
    # Actually, we can just calculate the required values on the fly 
    # or use a mathematical property.
    # The number of sequences starting with digit 'd' is:
    # ((N*K - 1 - current_pos)!) / ( (K-1)! * (K!)^(N-1) ) 
    # if digit 'd' has remaining count > 0.
    
    # Let S_total be the total number of sequences.
    # target = (S_total + 1) // 2
    
    # Instead of precomputing all factorials, we can compute the 
    # proportion of sequences starting with digit d.
    # If we have counts [c1, c2, ..., cN], the total permutations are:
    # T = (sum(ci))! / product(ci!)
    # The number of permutations starting with digit d is:
    # T_d = (sum(ci) - 1)! / ((cd-1)! * product(cj! for j != d))
    # T_d = T * (cd / sum(ci))
    
    # We need to find the sequence corresponding to rank 'target'.
    # We can maintain the current target rank and the current counts of digits.
    
    # Since we can't use loops, we use functools.reduce to iterate through the positions.
    # The state will be (current_target, current_counts)
    
    def get_total_perms(counts):
        # Total = (sum(counts))! / product(c!)
        # We use a helper to compute this.
        # To avoid huge numbers and loops, we use the property:
        # T_d = T * (cd / sum(ci))
        pass

    # Let's redefine: we need the sequence at rank (S+1)//2.
    # S = (N*K)! / (K!)^N
    # target = (S + 1) // 2
    
    # We can use a recursive function to find the sequence.
    # But Python has a recursion limit. For NK=250,000, we need to increase it.
    sys.setrecursionlimit(300000)
    
    def find_sequence(target, counts, total_len):
        if total_len == 0:
            return []
        
        # We need to find which digit d the sequence starts with.
        # The number of sequences starting with digit d is:
        # count_d = (total_len - 1)! / ( (counts[d]-1)! * product(counts[j]!) )
        # This is equal to: Total_Permutations * (counts[d] / total_len)
        
        # However, calculating Total_Permutations is too slow.
        # Let's use the property:
        # The fraction of sequences starting with digit d is counts[d] / total_len.
        # So we can find d such that:
        # sum_{j < d} (Total * counts[j]/total_len) < target <= sum_{j <= d} (Total * counts[j]/total_len)
        
        # This is equivalent to:
        # target <= Total * (sum_{j=1}^d counts[j]) / total_len
        
        # But we need the absolute rank. 
        # Let's use the fact that we only need the middle sequence.
        # For a symmetric distribution, the middle sequence is the one that 
        # "mirrors" the first sequence if we replace 1 with N, 2 with N-1, etc.
        # Wait, the set of all good sequences is symmetric.
        # If (S1, ..., Sm) is a good sequence, then (N+1-S1, ..., N+1-Sm) is also a good sequence.
        # Lexicographically, if Seq A < Seq B, then Mirror(A) > Mirror(B).
        # The middle sequence is the one where Seq == Mirror(Seq) if S is odd.
        # If S is even, the (S/2)-th and (S/2 + 1)-th are the middle ones.
        # The problem asks for floor((S+1)/2).
        # If S is odd, it's the (S+1)/2-th (the exact middle).
        # If S is even, it's the S/2-th.
        
        # Let's check Sample 1: N=2, K=2. S=6. floor(7/2) = 3.
        # Sequences: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
        # 3rd is (1,2,2,1).
        # Mirror of (1,1,2,2) is (2,2,1,1).
        # Mirror of (1,2,1,2) is (2,1,2,1).
        # Mirror of (1,2,2,1) is (2,1,1,2).
        # The 3rd sequence is the one just before the mirror of the 1st.
        # Actually, the (S/2)-th sequence is the mirror of the (S/2 + 1)-th.
        # The (S/2)-th sequence is the largest sequence that starts with a digit < (N+1)/2
        # or is the "smaller" of the two middle ones.
        
        # For any sequence A, let A' be the mirrored sequence.
        # A < A' if the first index i where they differ has A_i < A'_i.
        # A_i < N + 1 - A_i  => 2*A_i < N + 1 => A_i < (N+1)/2.
        
        # The sequence we want is the largest sequence A such that A <= A'.
        # This means at the first index i where A_i != A'_i, we must have A_i < A'_i.
        # To make A as large as possible while A <= A', we want A_i to be as large as possible.
        # The condition A <= A' is satisfied if for the first i where A_i != A'_i, A_i < A'_i.
        # This is equivalent to saying A is lexicographically smaller than or equal to its mirror.
        
        # To maximize A subject to A <= A':
        # We want the first digit A_1 to be as large as possible, but A_1 <= N + 1 - A_1.
        # So A_1 <= (N+1)/2.
        # 1. If A_1 < (N+1)/2, then A < A' is already guaranteed. To maximize A, 
        #    we should pick the largest possible A_1 < (N+1)/2, and then fill the 
        #    rest of the sequence as large as possible (descending order).
        # 2. If A_1 = (N+1)/2 (only possible if N is odd), then we must check A_2 and A'_2.
        
        # Let's refine this:
        # We want the largest sequence A such that A <= Mirror(A).
        # For i = 1 to NK:
        #   Try digits d = N down to 1:
        #     If we pick d, can we still satisfy A <= Mirror(A)?
        #     The condition A <= Mirror(A) depends on the first index i where A_i != Mirror(A)_i.
        #     Mirror(A)_i = N + 1 - A_{NK - i + 1}.
        #     This is tricky because Mirror(A)_i depends on the end of the sequence.
        
        # Let's use the property: the (S+1)//2-th sequence is the largest sequence A 
        # such that A <= Mirror(A).
        # A <= Mirror(A) iff for the first i where A_i != Mirror(A)_i, A_i < Mirror(A)_i.
        # This is equivalent to: A is lexicographically smaller than or equal to its mirror.
        
        # Let's use the "middle" property:
        # The sequence is the one that is "half-way" through the sorted list.
        # For N=2, K=2, S=6, target=3. Sequence: (1, 2, 2, 1).
        # For N=6, K=1, S=720, target=360. Sequence: (3, 6, 5, 4, 2, 1).
        # Notice for N=6, K=1, the 360th is the largest sequence starting with 3.
        # The sequences starting with 1, 2, 3 take 120*3 = 360 slots.
        # So it's the largest sequence starting with 3.
        # Largest sequence starting with 3 is (3, 6, 5, 4, 2, 1).
        
        # General rule:
        # The total number of sequences is S.
        # We want the (S+1)//2-th.
        # The number of sequences starting with digit d is S * (count_d / total_len).
        # We want to find d such that sum_{j=1}^{d-1} (S * count_j / total_len) < (S+1)//2 <= sum_{j=1}^d (S * count_j / total_len).
        # This is: (S+1)//2 <= S * (sum_{j=1}^d count_j) / total_len.
        # Roughly: 1/2 <= (sum_{j=1}^d count_j) / total_len.
        # sum_{j=1}^d count_j >= total_len / 2.
        
        # Let's test this:
        # N=2, K=2: total_len=4. sum_{j=1}^d count_j >= 2.
        # d=1: count_1 = 2. 2 >= 2 is True. So d=1.
        # Now we need the ( (S+1)//2 - 0 )-th sequence among those starting with 1.
        # S_1 = 6 * (2/4) = 3. We need the 3rd sequence starting with 1.
        # The sequences starting with 1 are: (1,1,2,2), (1,2,1,2), (1,2,2,1).
        # The 3rd is (1,2,2,1).
        
        # N=6, K=1: total_len=6. sum_{j=1}^d count_j >= 3.
        # d=1: 1 >= 3 (F), d=2: 2 >= 3 (F), d=3: 3 >= 3 (T). So d=3.
        # S_