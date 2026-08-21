import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use a trick to unpack the iterator
    # We can't use a for loop, but we can use a list comprehension 
    # or map to consume the input.
    
    # To handle the input without a loop, we convert the map to a list
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Let S[i] be the distance from rest area 1 to rest area i+1.
    # S[0] = 0
    # S[1] = A[0]
    # S[2] = A[0] + A[1] ...
    # The distance from s to t (s < t) is (S[t-1] - S[s-1])
    # The distance from s to t (s > t) is (Total_Sum - S[s-1]) + S[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate(A) gives [A0, A0+A1, A0+A1+A2, ...]
    # We prepend 0 to represent the starting point (rest area 1)
    prefix_sums = list(accumulate(A))
    # S contains the distance from area 1 to area i (for i=2...N)
    # We add 0 to the front to represent area 1
    S = [0] + [x % M for x in prefix_sums]
    
    # Total distance around the lake modulo M
    total_sum_mod = prefix_sums[-1] % M
    
    # We want (S[t-1] - S[s-1]) % M == 0 for s < t
    # and (total_sum_mod - S[s-1] + S[t-1]) % M == 0 for s > t
    
    # Let's count occurrences of each remainder in S
    # S has N elements (for areas 1 to N)
    counts = Counter(S)
    
    # For a fixed remainder r, any two areas with that remainder 
    # form a pair (s, t) where s < t and distance is a multiple of M.
    # Number of such pairs is r_count * (r_count - 1) // 2
    # However, we also need to consider s > t.
    
    # Let's analyze the condition:
    # If s < t: (S[t-1] - S[s-1]) % M == 0  => S[t-1] == S[s-1] (mod M)
    # If s > t: (total_sum_mod + S[t-1] - S[s-1]) % M == 0 => S[s-1] - S[t-1] == total_sum_mod (mod M)
    
    # Let, for each remainder r, count[r] be the number of times it appears in S.
    # Pairs (s, t) with s < t:
    # For each r, we have count[r] * (count[r] - 1) // 2 pairs.
    
    # Pairs (s, t) with s > t:
    # We need S[s-1] - S[t-1] ≡ total_sum_mod (mod M)
    # For each r, we need a t such that S[t-1] ≡ r - total_sum_mod (mod, M)
    # The number of such pairs is count[r] * count[(r - total_sum_mod) % M]
    # But we must ensure s > t. This is tricky without loops.
    
    # Alternative approach:
    # For every pair (s, t) with s != t:
    # Let x = S[s-1] and y = S[t-1].
    # If s < t, we need (y - x) % M == 0.
    # If s > t, we need (total_sum_mod + y - x) % M == 0.
    
    # Let's use the property:
    # Total pairs = Sum_{r=0 to M-1} [ count[r] * count[(r + 0) % M] ] 
    # But this counts s=t and doesn't distinguish s < t and s > t.
    
    # Correct Logic:
    # For each s, we seek t such that:
    # 1. t > s and S[t-1] ≡ S[s-1] (mod M)
    # 2. t < s and S[t-1] ≡ S[s-1] - total_sum_mod (mod M)
    
    # Let's iterate through the array S and maintain counts of remainders seen so far.
    # Since we can't use loops, we use a list comprehension to calculate 
    # for each index i, how many j < i satisfy the condition.
    # But we can't maintain state. 
    
    # Let's use the global counts:
    # For a fixed s, the number of t > s such that S[t-1] == S[s-1] is:
    # (count[S[s-1]] - 1) - (number of j < s-1 where S[j] == S[s-1])
    # This is still a loop.
    
    # Let's use the mathematical property:
    # Total = Sum_{r=0 to M-1} (count[r] * (count[r] - 1) // 2)  <-- this is s < t
    # Total += Sum_{r=0 to M-1} (count[r] * count[(r - total_sum_mod) % M])
    # Wait, the second term is for s > t.
    # For a fixed s, we need t < s such that S[t-1] ≡ S[s-1] - total_sum_mod (mod M).
    # Let r = S[s-1]. We need S[t-1] ≡ r - total_sum_mod (mod M).
    # Let r' = (r - total_sum_mod) % M.
    # For a fixed r, there are count[r] positions for s and count[r'] positions for t.
    # This counts all pairs (s, t) such that S[s-1] - S[t-1] ≡ total_sum_mod (mod M).
    # Does this include s < t? 
    # If s < t, then S[s-1] - S[t-1] ≡ - (S[t-1] - S[s-1]) ≡ 0 (mod M) 
    # only if total_sum_mod ≡ 0 (mod M).
    
    # Case 1: total_sum_mod == 0
    # Then s < t requires S[t-1] == S[s-1] and s > t requires S[t-1] == S[s-1].
    # Total = count[r] * (count[r] - 1) for all r.
    
    # Case 2: total_sum_mod != 0
    # s < t requires S[t-1] == S[s-1].
    # s > t requires S[t-1] == (S[s-1] - total_sum_mod) % M.
    # These two conditions are mutually exclusive because total_sum_mod != 0.
    # Total = Sum (count[r] * (count[r]-1)//2) + Sum (count[r] * count[(r - total_sum_mod)%M])
    # Wait, the second sum is over all s, t. Since s > t, we only count 
    # pairs where the index of r is greater than the index of r'.
    # Actually, if total_sum_mod != 0, then for any s, t:
    # if S[s-1] == S[t-1], then only one of (s < t) or (t < s) can satisfy the "multiple of M" 
    # condition (specifically, the one where we walk clockwise from the smaller index to the larger).
    # If S[s-1] - S[t-1] == total_sum_mod, then only s > t can satisfy it.
    
    # Let' same_rem = sum(c * (c - 1) // 2 for c in counts.values())
    # Let' diff_rem = sum(counts[r] * counts[(r - total_sum_mod) % M] for r in counts)
    # If total_sum_mod == 0:
    #    The condition for s < t is S[t-1] == S[s-1].
    #    The condition for s > t is S[t-1] == S[s-1].
    #    Total = same_rem * 2
    # If total_sum_mod != 0:
    #    s < t: S[t-1] == S[s-1] -> same_rem
    #    s > t: S[t-1] == (S[s-1] - total_sum_mod) % M -> 
    #    For each s, we need t < s. 
    #    This is sum_{i=0 to N-1} (count of (S[i] - total_sum_mod) seen before index i).
    
    # To avoid loops for the s > t case when total_sum_mod != 0:
    # We can use the fact that for any r, count[r] * count[(r - total_sum_mod) % M]
    # counts all pairs (s, t) such that S[s-1] - S[t-1] ≡ total_sum_mod.
    # Since total_sum_mod != 0, S[s-1] cannot equal S[t-1].
    # Thus, for any such pair, either s > t or s < t.
    # If s < t, then S[t-1] - S[s-1] ≡ -total_sum_mod ≡ (M - total_sum_mod).
    # So the product count[r] * count[(r - total_sum_mod) % M] 
    # correctly counts pairs (s, t) with s > t.
    
    same_rem = sum([c * (c - 1) // 2 for c in counts.values()])
    
    # Use a conditional expression to handle total_sum_mod == 0
    ans = (same_rem * 2) if total_sum_mod == 0 else (
        same_rem + sum([counts[r] * counts[(r - total_sum_mod) % M] for r in counts])
    )
    
    sys.stdout.write(str(ans) + '\n')

if __name__ == "__main__":
    solve()