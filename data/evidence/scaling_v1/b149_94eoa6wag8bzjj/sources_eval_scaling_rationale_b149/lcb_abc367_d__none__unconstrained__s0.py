import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension to capture the remaining A_i
    data_iter = iter(input_data)
    N = next(data_iter)
    M = next(data_iter)
    A = [x for x in data_iter]

    # Let P_i be the clockwise distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s).
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t).
    # We want (Distance) % M == 0.
    
    # Compute prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + A[:-1]) gives 0, A_1, A_1+A_2, ...
    # We only need the first N prefix sums.
    P = list(accumulate([0] + A[:-1]))
    total_sum = sum(A)
    
    # We are looking for pairs (s, t) such that:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # 2. s > t and (total_sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == total_sum % M
    
    # Let R_i = P_i % M.
    # For s < t: R_t == R_s
    # For s > t: R_s - R_t == total_sum % M  => R_t == (R_s - total_sum) % M
    
    R = [p % M for p in P]
    counts = Counter(R)
    
    # For a fixed s, we need to count t such that:
    # If t > s: R_t == R_s
    # If t < s: R_t == (R_s - total_sum) % M
    
    # However, it's easier to think about the total pairs:
    # For each value v in 0...M-1, let count(v) be the number of times it appears in R.
    # The number of pairs (s, t) with s < t and R_s == R_t is count(v) * (count(v) - 1) // 2.
    # The number of pairs (s, t) with s > t and R_s - R_t == total_sum % M is:
    # Let diff = total_sum % M. We need R_s - R_t = diff (mod M).
    # This is equivalent to R_t = (R_s - diff) % M.
    
    # Let's use a different approach to avoid loops and recursion:
    # Total pairs = Sum_{v=0 to M-1} [ count(v) * count((v - total_sum) % M) ]
    # But we must exclude cases where s == t.
    # If total_sum % M == 0, then (v - total_sum) % M == v, so we get count(v)^2.
    # Since s != t, we subtract the cases where s == t, which is N.
    
    diff = total_sum % M
    
    # Use a list comprehension to sum the products for all unique remainders present in R
    # result = sum(counts[v] * counts[(v - diff) % M] for v in counts)
    # If diff == 0, this includes s == t cases.
    
    ans = sum(counts[v] * counts[(v - diff) % M] for v in counts)
    
    # If diff == 0, we counted pairs (s, s), so subtract N.
    # Otherwise, s can never equal t because R_s == (R_s - diff) % M is only true if diff == 0.
    final_ans = ans - N if diff == 0 else ans
    
    print(final_ans)

if __name__ == "__main__":
    solve()