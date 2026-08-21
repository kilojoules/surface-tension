import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums modulo M
    # P[i] = (A_1 + ... + A_i) % M
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    
    # Use reduce to create the prefix sum list without loops
    # P will have N elements: P[0] is sum of first 0 elements (0), P[1] is A[0], etc.
    # To avoid loops, we can use a list comprehension or reduce.
    # However, since we need P[i] based on P[i-1], reduce is appropriate.
    
    # We generate the prefix sums: [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    # Note: we only need the first N prefix sums for the starting points.
    # The total sum is needed for the wrap-around cases.
    
    def accumulate_sums(acc, x):
        return acc + [ (acc[-1] + x) % M ]
    
    P = reduce(accumulate_sums, A, [0])
    # P now contains [0, P1, P2, ..., PN]
    # Total sum modulo M is P[-1]
    total_sum_mod = P[-1]
    
    # We are looking for pairs (s, t) such that distance is 0 mod M.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # For s > t: (total_sum_mod - P[s-1] + P[t-1]) % M == 0 => P[s-1] == (P[t-1] + total_sum_mod) % M
    
    # Let's count occurrences of each remainder in P[0...N-1]
    # We use a dictionary to count frequencies of P[i] for i in 0...N-1
    # Since we can't use loops, we use a dictionary with reduce or a Counter
    from collections import Counter
    counts = Counter(P[:-1])
    
    # For a fixed s, we want to find t != s such that:
    # If t > s: P[t-1] == P[s-1]
    # If t < s: P[t-1] == (P[s-1] - total_sum_mod) % M
    
    # Total pairs = Sum over all r in counts:
    # count(r) * (count(r) - 1)  <-- This covers s < t where P[s-1] == P[t-1]
    # PLUS
    # count(r) * count((r - total_sum_mod) % M) <-- This covers s > t
    # BUT we must subtract the cases where s > t and (r - total_sum_mod) % M == r
    # because that would imply total_sum_mod == 0, and we already counted s < t.
    # Actually, the simplest way:
    # For every s, the number of t's is:
    # (number of t > s with P[t-1] == P[s-1]) + (number of t < s with P[t-1] == (P[s-1] - total_sum_mod) % M)
    
    # Let's use the property:
    # Total = Sum_{r=0 to M-1} [ count(r) * count((r + total_sum_mod) % M) ]
    # Then subtract the cases where s == t (which happens when r == (r + total_sum_mod) % M)
    # Wait, the condition s != t is strict.
    # If total_sum_mod == 0, then (r + total_sum_mod) % M == r.
    # The formula becomes Sum [ count(r) * count(r) ], then subtract N (for s=t).
    # If total_sum_mod != 0, then (r + total_sum_mod) % M != r.
    # The formula is Sum [ count(r) * count((r + total_sum_mod) % M) ].
    
    # To implement the sum without a loop, we use map and sum.
    # We iterate over the unique remainders present in the Counter.
    
    ans = sum(map(lambda r: counts[r] * counts[(r + total_sum_mod) % M], counts.keys()))
    
    # If total_sum_mod % M == 0, we have counted pairs (s, s) which are not allowed.
    # There are N such pairs.
    final_ans = ans - N if total_sum_mod % M == 0 else ans
    
    print(final_ans)

if __name__ == "__main__":
    solve()