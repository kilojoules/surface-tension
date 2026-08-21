```python
import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    m = int(input_data[1])
    a = list(map(int, input_data[2:]))
    
    # Calculate prefix sums modulo M
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # The distance from s to t (s < t) is (P[t] - P[s]) % M
    # The distance from s to t (s > t) is (TotalSum - P[s] + P[t]) % M
    
    # We use reduce to build the prefix sum list without explicit loops
    # prefix_sums[i] is the distance from rest area 1 to rest area i+1
    prefix_sums = reduce(
        lambda acc, x: acc + [ (acc[-1] + x) % m ],
        a,
        [0]
    )
    
    # The total distance around the lake modulo M
    total_sum_mod = prefix_sums[-1] % m
    
    # We are looking for pairs (s, t) such that dist(s, t) % M == 0
    # Case 1: s < t
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # Case 2: s > t
    # (TotalSum - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == TotalSum % M
    
    # Let's count occurrences of each prefix sum modulo M
    # We only care about P[0]...P[N-1] because P[N] is the return to start
    # Note: The problem says s != t.
    # For a fixed s, we want t such that:
    # If t > s: P[t-1] \equiv P[s-1] (mod M)
    # If t < s: P[t-1] \equiv P[s-1] - TotalSum (mod M)
    
    # Let counts be a dictionary of {remainder: count}
    # We can't use a loop to build the dictionary, so we use a trick with a list
    # and a comprehension, or just use the fact that we can use a list as a frequency array
    # since M <= 10^6.
    
    # However, we cannot use a loop to populate the frequency array.
    # We can use a generator expression inside a sum().
    
    # Let's refine the logic:
    # For each s \in {1...N}, we seek t \in {1...N}, t \neq s.
    # The distance from s to t is:
    # If s < t: (P[t-1] - P[s-1]) % M
    # If s > t: (P[N-1] + A_N - P[s-1] + P[t-1]) % M  <-- Wait, P[N] is total sum.
    # Let P[i] = sum(A_1...A_i) % M. 
    # Dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # Dist(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    
    # Let X = P[s-1] and Y = P[t-1].
    # Condition: 
    # 1. s < t and Y == X
    # 2. s > t and Y == (X - P[N]) % M
    
    # Let's use a frequency array for P[0]...P[N-1]
    # Since we can't use loops, we use a list comprehension to count
    # But we can't use a loop to fill the frequency array.
    # Actually, we can use a dictionary and a generator to count.
    
    # Correct approach to count without loops:
    # Use a list to store P[0]...P[N-1].
    # Use a dictionary to store frequencies of each value in P.
    # But wait, I can't use a loop to build the dictionary.
    # I can use a trick: sorted list and groupby.
    
    from itertools import groupby
    
    # P contains P[0]...P[N-1]
    P = prefix_sums[:-1] 
    total = prefix_sums[-1]
    
    # Group identical values to get frequencies: {val: count}
    # sorted(P) is allowed. groupby is allowed.
    freqs = {k: len(list(g)) for k, g in groupby(sorted(P))}
    
    # For a fixed s, we want t != s such that:
    # If t > s, P[t-1] == P[s-1]
    # If t < s, P[t-1] == (P[s-1] - total) % M
    
    # Total pairs = Sum_{s=1 to N} [ (count of P[s-1]) - 1 (for t>s) 
    #                               + (count of (P[s-1] - total)%M) (for t<s) ]
    # This is slightly wrong because the "t > s" and "t < s" depends on indices.
    
    # Let's reconsider:
    # A pair (s, t) is valid if:
    # 1. s < t and P[t-1] \equiv P[s-1] (mod M)
    # 2. s > t and P[t-1] \equiv (P[s-1] - total) (mod M)
    
    # For a fixed value v, let C(v) be the number of times v appears in P[0...N-1].
    # The number of pairs (s, t) with s < t and P[s-1] = P[t-1] = v is C(v)*(C(v)-1)//2.
    # The number of pairs (s, t) with s > t and P[t-1] = (P[s-1] - total) % M is:
    # This is harder because it depends on the relative positions.
    
    # Actually, the total number of pairs (s, t) is:
    # Sum_{v} [ C(v) * C((v - total) % M) ]
    # But we must subtract cases where s=t, which happens if v == (v - total) % M,
    # i.e., total % M == 0. In that case, we subtract C(v) for each v.
    # Wait, if total % M == 0, then (s, t) is valid if P[s-1] == P[t-1].
    # There are C(v)*(C(v)-1) such pairs for each v.
    # If total % M != 0, then s < t and s > t are disjoint conditions.
    # For a fixed s, t is valid if:
    # (t > s and P[t-1] == P[s-1]) OR (t < s and P[t-1] == (P[s-1] - total) % M)
    
    # Let's use the property:
    # Total = Sum_{s=1 to N} (Count of t > s where P[t-1] == P[s-1]) 
    #       + Sum_{s=1 to N} (Count of t < s where P[t-1] == (P[s-1] - total) % M)
    
    # This is equivalent to:
    # Sum_{v} [ C(v)*(C(v)-1)//2 ]  <-- This is for s < t and P[s-1] == P[t-1]
    # + Sum_{s=1 to N} [ Count of t < s where P[t-1] == (P[s-1] - total) % M ]
    
    # To calculate the second term without loops:
    # We can use a technique with a custom object or a mutable container inside a list comprehension.
    # But a cleaner way is to realize that the second term is:
    # Sum_{v} [ C(v) * C((v - total) % M) ] 
    # BUT only for pairs where the index of (v-total) is less than the index of v.
    
    # Actually, there is a much simpler way.
    # The condition is: (P[t-1] - P[s-1]) % M == 0 if s < t
    # and (P[N] - P[s-1] + P[t-1]) % M == 0 if s > t.
    # This is equivalent to:
    # s < t: P[t-1] \equiv P[s-1] (mod M)
    # s > t: P[t-1] \equiv P[s-1] - P[N] (mod M)
    
    # Let's use the "mutable container" trick to count t < s in one pass.
    # We use a list of size M to store counts of P[i] seen so far.
    
    # Since we can't use loops, we can use a list comprehension that updates a 
    # frequency array and returns the count.
    # However, updating a list in a comprehension is generally discouraged.
    # A better way:
    # The total count is Sum_{v} [ C(v) * C((v - total) % M) ]
    # If total % M == 0, this is Sum C(v)^2. But we need s != t, so Sum C(v)(C(v)-1).
    # If total % M != 0, then for any pair {s, t}, only one of (s,t) or (t,s) 
    # can satisfy the condition.
    # Specifically, if P[t-1] == P[s-1], then (s,t) is valid if s < t.
    # If P[t-1] == (P[s-1] - total) % M, then (s,t) is valid if s > t.
    
    # Let's test this:
    # If total % M != 0:
    # For every pair of indices {i, j} with i < j:
    # Pair (i+1, j+1) is valid if P[j] == P[i]
    # Pair (j+1, i+1) is valid if P[i] == (P[j] - total) % M
    # These two conditions are: P[j] == P[i] AND P[i] == (P[j] - total) % M
    # Which implies total % M == 0.
    # So if total % M != 0, the conditions are mutually exclusive.
    # Total = Sum_{v} [ C(v)*(C(v)-1)//2 ] + Sum_{v} [ C(v) * C((v - total) % M) ]
    # Wait, the second term is Sum_{j > i} [ P[i] == (P[j] - total) % M ].
    # This is not simply C(v)*C(v-total).
    # Actually, if total % M != 0, then for any two indices i, j:
    # Either (P[j]-P[i])%M == 0 or (P[i]-P[j]+total)%M == 0 or neither.
    # If P[j] == P[i], then (P[j]-P[i])%M == 0.
    # If P[i] == (P[j]-total)%M, then (P[i]-P[j]+total)%M == 0.
    # These are the same condition: P[j] - P[i] \equiv 0 \pmod M vs P[j] - P[i] \equiv total \pmod M.
    
    # Let's use the property:
    # Total = Sum_{i < j} [ (P[j]-P[i]) % M == 0 ] + Sum_{i < j} [ (P[i]-P[j]+total) % M == 0 ]
    # Total = Sum_{v} [ C(v)*(C(v)-1)//2 ] + Sum_{i < j} [ P[j] - P[i] \equiv total \pmod M ]
    
    # The second term Sum_{i < j} [ P[j] - P[i] \equiv total \pmod M ]
    # can be computed by iterating through the array and keeping track of counts.
    # Since we can't use loops, we can use a trick with a list and a function.
