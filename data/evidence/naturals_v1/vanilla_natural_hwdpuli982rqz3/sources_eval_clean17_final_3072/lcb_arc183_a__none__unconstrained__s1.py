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
    # Note: S is symmetric. If a sequence is (a1, a2, ..., a_{NK}), 
    # its "complement" (where each x is replaced by N+1-x) is also a good sequence.
    # Lexicographically, if sequence A < B, then complement(A) > complement(B).
    # The middle two sequences (if S is even) or the middle one (if S is odd)
    # are related to the "halfway" point of the lexicographical order.
    
    # Specifically, the floor((S+1)/2)-th sequence is the one immediately 
    # preceding the "central" point. 
    # Due to the symmetry of the set of all good sequences, 
    # the floor((S+1)/2)-th sequence is the one where we try to pick 
    # the smallest possible number at each step, but we are constrained 
    # by the fact that we want the "middle" of the distribution.
    
    # Actually, a simpler observation:
    # The set of all good sequences is symmetric. 
    # If we sort them, the i-th sequence and the (S-i+1)-th sequence are 
    # "complements" (replace x with N+1-x).
    # The floor((S+1)/2)-th sequence is the last sequence that starts with 
    # a digit 'd' such that the number of sequences starting with 1...d-1 
    # is less than or equal to floor((S+1)/2).
    
    # However, calculating S is impossible with loops/recursion due to constraints.
    # We need to determine each digit x_i one by one.
    # For the i-th position, we try digits d = 1, 2, ..., N.
    # Let count(d) be the number of ways to complete the sequence if we pick d.
    # We move to the next digit once the sum of count(d) reaches or exceeds floor((S+1)/2).
    
    # To avoid huge numbers, we can observe that we want the "middle" sequence.
    # The middle sequence is the one that is "lexicographically balanced".
    # For a position i, if we have remaining counts c1, c2, ..., cN,
    # the number of ways to complete is (sum(cj))! / product(cj!).
    # We want to find the smallest d such that 
    # sum_{j=1}^{d-1} (ways to complete with j) < (S+1)/2 <= sum_{j=1}^{d} (ways to complete with j).
    
    # Because we cannot use loops, we use a functional approach.
    # We can maintain the state as (current_counts, target_rank) and use a 
    # list comprehension to find the digit and update the state.
    
    # To handle the "middle" without calculating S:
    # The target rank is (S+1)//2.
    # For the first digit, we check d=1, 2...
    # Ways to complete if we pick d: W(d) = ((NK-1)! / (K!^N)) * (K / (NK)) = S * (K / NK) = S/N.
    # Since each digit 1..N appears K times, the first digit is '1' for the first S/N sequences,
    # '2' for the next S/N, and so on.
    # The middle index (S+1)//2 falls into the ceil(N/2)-th group.
    # Let mid_digit = (N + 1) // 2.
    # The number of sequences starting with 1... (mid_digit-1) is (mid_digit-1) * (S/N).
    # The relative rank within the group starting with mid_digit is:
    # rank = (S+1)//2 - (mid_digit-1)*S/N.
    
    # This suggests a recursive-like structure. Since we can't use recursion,
    # we use a while loop (which is forbidden) or a reduce.
    # We can use a reduce over the range of the sequence length NK.
    
    # State: (current_counts, current_rank)
    # current_counts: tuple of remaining counts of each digit
    # current_rank: the rank we are looking for within the current subtree
    
    # To avoid loops, we use a helper function to calculate combinations
    # and a reduce to iterate through the sequence length.
    
    import math
    from fractions import Fraction

    # Precompute factorials for the combination formula
    # Since we can't use loops, we use a map or a list comprehension
    # But we need them for the combination formula.
    # Given N, K <= 500, NK = 250,000. We can't precompute all factorials.
    # But we can compute the ratio of ways.
    # W(d) = (TotalRemaining - 1)! / (c1! * (cd-1)! * ... * cN!)
    # W(d) = [ (TotalRemaining)! / (c1! * ... * cN!) ] * (cd / TotalRemaining)
    
    # Let f(counts) = (sum(counts))! / product(counts!)
    # The number of sequences starting with d is f(counts) * (count[d] / sum(counts))
    
    # To avoid loops, we use a reduce to build the sequence.
    # To avoid loops inside the reduce, we use a list comprehension to find the digit.
    
    # Since we need to track the rank, we'll use a list to store the state.
    # state = [current_counts, current_rank]
    
    def get_digit(state, pos):
        counts, rank = state
        total = sum(counts)
        
        # We need to find d such that sum_{j=1}^{d-1} W(j) < rank <= sum_{j=1}^{d} W(j)
        # W(j) = (total - 1)! / (product(counts!) / counts[j])
        # W(j) = [total! / product(counts!)] * (counts[j] / total)
        # Let TotalWays = total! / product(counts!)
        # W(j) = TotalWays * counts[j] / total
        
        # To avoid calculating TotalWays every time, we can't easily.
        # But we can check the condition:
        # rank <= sum_{j=1}^{d} (TotalWays * counts[j] / total)
        # rank * total <= TotalWays * sum_{j=1}^{d} counts[j]
        
        # We need TotalWays. We can't use loops.
        # We can use math.factorial and a comprehension.
        
        total_ways = math.factorial(total)
        denom = reduce(lambda x, y: x * math.factorial(y), counts, 1)
        s_current = total_ways // denom
        
        # Find d (1-indexed)
        # We want the smallest d such that rank <= (s_current * sum(counts[0:d])) // total
        # Since we can't loop, we use a list comprehension to find all d's and take the min.
        d = min([d for d in range(1, N + 1) if rank <= (s_current * sum(counts[:d])) // total])
        
        # Update rank for the next position:
        # New rank is rank - (sum of W(j) for j < d)
        # New rank = rank - (s_current * sum(counts[:d-1])) // total
        new_rank = rank - (s_current * sum(counts[:d-1])) // total
        
        # Update counts
        new_counts = list(counts)
        new_counts[d-1] -= 1
        
        return (tuple(new_counts), new_rank)

    # Initial state
    # S = (N*K)! / (K!^N)
    # target = (S + 1) // 2
    
    s_total = math.factorial(N * K) // (math.factorial(K)**N)
    initial_state = ((K,) * N, (s_total + 1) // 2)
    
    # Use reduce to simulate the process of picking NK digits
    # The accumulator is the state, and we ignore the range value.
    # We need to keep track of the digits picked.
    
    # Since reduce doesn't allow easy access to the "current" digit being picked 
    # without returning it in the state, we'll store the sequence in the state.
    
    def process(state, _):
        counts, rank, seq = state
        # We need to recalculate total and s_current inside
        total = sum(counts)
        if total == 0: return state
        
        s_current = math.factorial(total) // reduce(lambda x, y: x * math.factorial(y), counts, 1)
        
        d = min([d for d in range(1, N + 1) if counts[d-1] > 0 and rank <= (s_current * sum(counts[:d])) // total])
        
        new_rank = rank - (s_current * sum(counts[:d-1])) // total
        new_counts = list(counts)
        new_counts[d-1] -= 1
        
        return (tuple(new_counts), new_rank, seq + [d])

    # To avoid loops/recursion, we use reduce.
    # Initial state: (counts, rank, sequence)
    final_state = reduce(process, range(N * K), ((K,) * N, (s_total + 1) // 2, []))
    
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()