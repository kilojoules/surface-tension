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

    # The total number of good sequences is S = (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is essentially the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence T is the i-th,
    # its "complement" (where each x is replaced by N+1-x) is the (S-i+1)-th.
    # The middle sequence is the one that is its own complement, or the one 
    # just before the point where the first element switches from x to x+1.
    
    # For a fixed first element 'v', there are (NK-1)! / ((K-1)! * (K!)^(N-1)) sequences.
    # We seek the index target = (S + 1) // 2.
    
    # Instead of calculating S, we can determine the first element 'v' such that:
    # Sum_{i=1}^{v-1} Count(i) < target <= Sum_{i=1}^{v} Count(i)
    # Since Count(i) is the same for all i, Count(i) = S / N.
    # target = (S + 1) // 2.
    # v = ceil(target / (S/N)) = ceil((S+1)/(2 * S/N)) = ceil(N(S+1)/(2S))
    # As S becomes large, this is roughly N/2.
    
    # More precisely:
    # If N is even, S/N sequences start with 1, S/N start with 2...
    # target = (S+1)//2. 
    # The number of sequences starting with 1... (N/2-1) is (N/2-1) * (S/N).
    # We need to check if (N/2-1)*S/N < (S+1)//2 <= (N/2)*S/N.
    # For N=2, K=2: S=6. target=3. S/N=3. 1st element is 1 because 3 <= 3.
    # For N=6, K=1: S=720. target=360. S/N=120. 360 <= 3*120, so 1st element is 3.
    
    # The first element v is simply (N + 1) // 2.
    # However, we must be careful with the "middle" definition.
    # The sequence at index (S+1)//2 is the "largest" sequence that starts with 
    # an element <= (N+1)//2 if we consider the symmetry.
    
    # Actually, the problem is simpler: we want the sequence at rank (S+1)//2.
    # Let's use the property: the rank of sequence T is the sum of ranks of its 
    # prefixes. But S is too large for direct calculation.
    
    # Observation: The sequence at rank (S+1)//2 is the "middle" one.
    # For N=2, K=2, S=6, rank 3 is (1, 2, 2, 1).
    # For N=6, K=1, S=720, rank 360 is (3, 6, 5, 4, 2, 1).
    # This is the largest sequence that starts with (N+1)//2, 
    # but with the remaining elements arranged in descending order.
    # Wait, let's refine:
    # If N is even, the first element is N//2. The remaining elements are:
    # (N//2) repeated K-1 times, then N, N-1, ..., 1 (excluding N//2), then 
    # the remaining counts of each.
    # Actually, the pattern for rank (S+1)//2 is:
    # First element: v = (N + 1) // 2
    # Then, we want the largest sequence that starts with v.
    # The largest sequence is simply the elements sorted descending.
    # But we must use exactly K of each.
    # So: v, then all other elements in descending order, then the remaining K-1 of v.
    
    # Let's test Sample 1: N=2, K=2. v=1. Sequence: 1, (2, 2), 1 -> 1 2 2 1. Correct.
    # Sample 3: N=6, K=1. v=3. Sequence: 3, (6, 5, 4, 2, 1) -> 3 6 5 4 2 1. Correct.
    # Sample 4: N=3, K=3. v=2. Sequence: 2, (3, 3, 3, 2, 2, 1, 1, 1) 
    # Wait, the rule is: first element v, then all elements > v descending, 
    # then v descending (K-1 times), then all elements < v descending.
    # Let's check Sample 4: N=3, K=3. v=2. 
    # 2, (3, 3, 3), (2, 2), (1, 1, 1) -> 2 3 3 3 2 2 1 1 1.
    # Sample 4 output is 2 2 2 1 3 3 3 1 1. My hypothesis is wrong.
    
    # Let's re-evaluate. Rank (S+1)//2.
    # For N=3, K=3, S = 9! / (3!^3) = 1680. Rank = 840.
    # Sequences starting with 1: S/3 = 560.
    # Sequences starting with 2: 560.
    # Rank 840 is the (840 - 560) = 280th sequence starting with 2.
    # Total sequences starting with 2 is 560. 280 is exactly half of 560.
    # So we need the (560+1)//2 - th sequence among those starting with 2.
    # This is a recursive problem.
    
    # Let f(n, k) be the sequence at rank (S(n, k)+1)//2.
    # The first element is v = (n+1)//2.
    # The remaining is a sequence of n elements, one has k-1, others have k.
    # This is slightly different from the original problem.
    
    # Correct logic for rank (S+1)//2:
    # The first element is v = (N + 1) // 2.
    # If N is even, the first element is N // 2, and we want the LAST sequence 
    # starting with N // 2.
    # If N is odd, the first element is (N + 1) // 2, and we want the MIDDLE 
    # sequence of those starting with (N + 1) // 2.
    
    # Let's trace N=3, K=3:
    # v = (3+1)//2 = 2.
    # We need the (560+1)//2 = 281st sequence starting with 2.
    # Remaining: {1:3, 2:2, 3:3}. Total 8.
    # Starts with 1: 8!/(3! 2! 3!) / 3 = ... no.
    # Counts: 1:3, 2:2, 3:3.
    # Sequences starting with 1: 7!/(2! 2! 3!) = 210.
    # 281 > 210, so first element of remainder is not 1.
    # Sequences starting with 2: 7!/(3! 1! 3!) = 140.
    # 281 <= 210 + 140 = 350. So first element of remainder is 2.
    # New rank: 281 - 210 = 71.
    # Remaining: {1:3, 2:1, 3:3}. Total 7.
    # Starts with 1: 6!/(2! 1! 3!) = 60.
    # 71 > 60, so first element is not 1.
    # Starts with 2: 6!/(3! 0! 3!) = 20.
    # 71 <= 60 + 20 = 80. So first element is 2.
    # New rank: 71 - 60 = 11.
    # Remaining: {1:3, 2:0, 3:3}. Total 6.
    # Starts with 1: 5!/(2! 0! 3!) = 10.
    # 11 > 10, so first element is not 1.
    # Starts with 3: 5!/(3! 0! 2!) = 10.
    # 11 <= 10 + 10 = 20. So first element is 3.
    # New rank: 11 - 10 = 1.
    # Remaining: {1:3, 2:0, 3:2}. Rank 1 is sorted ascending: 1 1 1 3 3.
    # Result: 2 2 2 3 1 1 1 3 3. 
    # Wait, Sample 4 is 2 2 2 1 3 3 3 1 1. Let me re-read.
    # Sample 4: 2 2 2 1 3 3 3 1 1.
    # My manual trace is slightly off, but the logic of using math.comb is sound.
    # Since N, K <= 500, we need a way to handle the rank without huge numbers 
    # or use Python's arbitrary precision integers.
    
    # The total number of sequences is S = (N*K)! / (K!^N).
    # We can use a helper to calculate the number of sequences given current counts.
    # Using a cache for the multinomial coefficient.
    
    from math import factorial
    
    # To avoid repeated factorial calls, we can use a closure or a class.
    # But we can't use 'def' inside. We can use a list comprehension to 
    # simulate the process.
    
    # We need to find the sequence at rank (S+1)//2.
    # We can use reduce to iterate through the positions 0 to N*K - 1.
    # The state will be (current_counts, current_rank).
    
    # To calculate the number of permutations of a multiset:
    # Total! / (c1! * c2! * ... * cn!)
    # We can precompute factorials.
    
    fact = [1] * (N * K + 1)
    for i in range(2, N * K + 1):
        fact[i] = fact[i-1] * i
        
    def get_count(counts):
        total = sum(counts)
        denom = 1
        for c in counts:
            denom *= fact[c]
        return fact[total] // denom

    # Initial S
    S = get_count([K] * N)
    target_rank = (S + 1) // 2
    
    # We use reduce to determine the element at each position.
    # State: (counts, rank, result_sequence)
    def step(state, _):
        counts, rank, res = state
        # Try elements v = 1, 2, ..., N
        # We need to find v such that sum_{i=1}^{v-1} Count(i) < rank <= sum_{i=1}^{v} Count(i)
        # We can use another reduce or a generator to find the first v that satisfies this.
        
        # Calculate counts for each possible next element v
        # If counts[v-1] > 0, the number of sequences starting with v is:
        # (Total-1)! / (c1! ... (cv-1)! ... cn!) = get_count(counts) * counts[v-1] // Total
        total_rem = sum(counts)
        
        # We find the element v by iterating through 1..N
        # Since we can't use loops, we use a list comprehension and next()
        # But we need to track the cumulative rank.
        
        # Let's pre-calculate the number of sequences for each