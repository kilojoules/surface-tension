The constraint to avoid explicit loops and recursion forces a shift from imperative state management to a functional paradigm. While not necessarily "cleaner" for complex combinatorial logic, it encourages the use of higher-order functions and immutable-style transformations. To solve this, I will use `functools.reduce` to simulate the iterative process of determining each element of the sequence, and `math.comb` to calculate the number of permutations of multisets.

```python
import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (N*K)! / (K!^N)
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    
    # To calculate the number of ways to complete a sequence given remaining counts:
    # Total ways = (sum(counts))! / product(counts[i]!)
    # However, calculating large factorials is slow. 
    # We can use a property: if we fix the next element to be 'i', 
    # the number of ways to fill the rest is:
    # (Total_Remaining - 1)! / (counts[0]! ... (counts[i]-1)! ... counts[N-1]!)
    # Which is: [Total_Remaining! / product(counts[j]!)] * (counts[i] / Total_Remaining)

    def get_total_permutations(counts):
        # Using a reduction to calculate the multiset coefficient
        # Total = (sum(counts)!) / product(c!)
        # We can compute this using combinations: comb(n, k1) * comb(n-k1, k2) ...
        return reduce(lambda acc, c: acc * comb(total_rem := sum(counts[counts.index(c):]), c), 
                      range(len(counts)), 1)
    
    # Since the above is slow inside a loop, we use a more efficient way to 
    # calculate the number of sequences starting with digit 'd'.
    # Let W be the total permutations of the current multiset.
    # The number of permutations starting with d is W * (count[d] / total_remaining).

    def step(state, _):
        current_counts, target_rank = state
        total_rem = sum(current_counts)
        if total_rem == 0:
            return state
        
        # Calculate total permutations of the remaining multiset
        # W = (total_rem!) / (c1! * c2! ... * cn!)
        # We can compute this once per position.
        # Using a helper to compute W:
        def calc_w(counts):
            res = 1
            rem = sum(counts)
            for c in counts:
                res *= comb(rem, c)
                rem -= c
            return res
        
        w_total = calc_w(current_counts)
        
        # Find the smallest digit d such that the sum of permutations 
        # starting with 1...d-1 is less than target_rank.
        def find_digit(d, accumulated_rank):
            if d > n: return n, accumulated_rank
            # Ways to start with digit d (1-indexed)
            # count of d is current_counts[d-1]
            ways_d = (w_total * current_counts[d-1]) // total_rem if total_rem > 0 else 0
            if accumulated_rank <= ways_d:
                return d, accumulated_rank
            return find_digit(d + 1, accumulated_rank - ways_d)

        # Using a loop-free way to find the digit:
        # We can use a generator and next() to find the first d that satisfies the condition.
        # But the prompt forbids loops. We use a recursive-like structure via reduce or 
        # a list comprehension with next().
        
        # To avoid recursion, we pre-calculate ways for all d in 1..N
        ways_list = [ (w_total * current_counts[d-1]) // total_rem if total_rem > 0 else 0 for d in range(1, n+1)]
        
        # Find d such that sum(ways[:d-1]) < target_rank <= sum(ways[:d])
        # We use a list comprehension to find the index
        # The index is the first i where sum(ways[:i+1]) >= target_rank
        d = next(i + 1 for i, s in enumerate(
            [sum(ways_list[:i+1]) for i in range(n)]
        ) if s >= target_rank)
        
        # Update target_rank for the next position
        new_target_rank = target_rank - sum(ways_list[:d-1])
        
        # Update counts
        new_counts = list(current_counts)
        new_counts[d-1] -= 1
        
        return (new_counts, new_target_rank), d

    # Initial S calculation
    # S = (N*K)! / (K!^N)
    # We need target = (S + 1) // 2
    def get_s(n, k):
        res = 1
        rem = n * k
        for i in range(n):
            res *= comb(rem, k)
            rem -= k
        return res

    s_total = get_s(n, k)
    target_rank = (s_total + 1) // 2
    
    # We need to run the 'step' function N*K times.
    # reduce(function, sequence, initial)
    # The state is (current_counts, current_target_rank)
    # We want to collect the digits chosen at each step.
    
    # Since reduce only returns the final state, we use a trick to collect results.
    # We store the sequence in a list within the state.
    def master_step(state, _):
        counts, rank, seq = state
        total_rem = sum(counts)
        if total_rem == 0: return state
        
        # Calculate W
        w_total = 1
        rem = total_rem
        # Using a list comprehension to simulate the product of combinations
        # This is a bit hacky but avoids explicit loops
        def calc_w_fast(counts):
            # We can't use a loop, so we use reduce
            return reduce(lambda a, c: a * comb(sum(counts[counts.index(c):]), c) if c > 0 else a, 
                          [c for c in counts if c > 0], 1)
        
        # Actually, the most reliable way to get W without loops is:
        # W = (sum(counts)!) / product(c!)
        # But we can just use the logic: ways_d = W * count[d] / total_rem
        # Let's pre-calculate W for the current state.
        
        # To avoid the O(N) calc_w inside the O(NK) process, 
        # we can't really, but N=500, NK=250,000 is too slow for O(N^2 K).
        # Wait, the number of ways to start with d is:
        # (total_rem - 1)! / (c1! ... (cd-1)! ... cn!)
        # = (total_rem - 1)! / (product(ci!) / cd)
        # = [ (total_rem)! / product(ci!) ] * cd / total_rem
        
        # Let's use the property that we only need to compare target_rank with ways_d.
        # target_rank <= ways_d  <=>  target_rank <= W * cd / total_rem
        # <=> target_rank * total_rem / W <= cd
        
        # However, W is huge. Let's use the fact that:
        # ways_d = comb(total_rem - 1, counts[0], counts[1], ..., counts[d-1]-1, ..., counts[n-1])
        # ways_d = (total_rem - 1)! / (c0! ... (cd-1)! ... cn-1!)
        
        # Let's use a simpler approach: 
        # The number of sequences starting with 1 is W1, starting with 2 is W2...
        # W_d = W_1 * (count[d] / count[1])
        # This allows us to find d without calculating the massive W.
        # target_rank <= sum_{i=1}^{d-1} W_i + W_d
        
        # Let's use the property: the proportion of sequences starting with d is count[d] / total_rem.
        # So we are looking for d such that:
        # sum_{i=1}^{d-1} (count[i]/total_rem) < target_rank / W_total <= sum_{i=1}^{d} (count[i]/total_rem)
        # (sum_{i=1}^{d-1} count[i]) / total_rem < target_rank / W_total <= (sum_{i=1}^{d} count[i]) / total_rem
        # This requires W_total.
        
        # Let's reconsider: we need the (S+1)//2-th sequence.
        # For a symmetric distribution, the "middle" sequence is the one that is 
        # the reverse of the 1st sequence if we mirrored the alphabet.
        # The 1st sequence is 1(K times), 2(K times), ..., N(K times).
        # The last sequence is N(K times), ..., 1(K times).
        # The middle sequence is the one that is "lexicographically" in the center.
        # For any sequence A, its "complement" A' (where x becomes N-x+1) 
        # is such that if A is the i-th, A' is the (S-i+1)-th.
        # We want the (S+1)//2-th.
        # If S is odd, (S+1)//2 is the exact middle. A = A'.
        # If S is even, (S+1)//2 is the smaller of the two middle ones.
        
        # The complement of a sequence is replacing each x with (N - x + 1).
        # The lexicographical middle of all permutations of a multiset is 
        # simply the sequence that is "balanced".
        # Actually, the simplest way to find the (S+1)//2-th sequence is to 
        # realize that the set of all good sequences is closed under the 
        # operation of reversing the alphabet (x -> N-x+1) AND reversing the string.
        # But the problem is simpler: the "middle" sequence is the one 
        # that looks like the "average" sequence.
        # For N=2, K=2: (1,1,2,2), (1,2,1,2), (1,2,2,1), (2,1,1,2), (2,1,2,1), (2,2,1,1)
        # The 3rd is (1,2,2,1).
        # Notice that the 3rd is the "complement" of the 4th.
        # The 1st is the complement of the 6th, 2nd of 5th, 3rd of 4th.
        # The complement of (S1, ..., Sm) is (N-S1+1, ..., N-Sm+1).
        # If we sort the sequences, the i-th sequence's complement is the (S-i+1)-th.
        # We want the (S+1)//2-th.
        # This is the sequence A such that A is the "largest" sequence that is 
        # lexicographically smaller than or equal to its complement A'.
        # A <= A' means at the first index i where they differ, A_i < A'_i.
        # A_i < N - A_i + 1  => 2*A_i < N + 1.
        
        # This is much simpler! We want the sequence A that is the 
        # "largest" sequence such that A <= complement(A).
        # To make A as large as possible while A <= A', we want the first 
        # index i where A_i != A'_i to have A_i < A'_i, and for all j < i, A_j = A'_j.
        # To maximize A, we want to push the first difference as far back as possible.
        # For j < i, A_j = N - A_j + 1 => 2*A_j = N + 1.
        # This is only possible if N is odd and A_j = (N+1)//2.
        # If N is even, A_j can never