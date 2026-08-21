import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = list(map(int, sys.stdin.read().split()))
    
    # N: number of rest areas, M: the divisor
    # A: list of distances between rest area i and i+1
    N, M = input_data[0], input_data[1]
    A = input_data[2:]
    
    # Let P[i] be the clockwise distance from rest area 1 to rest area i.
    # P[1] = 0
    # P[2] = A[0]
    # P[3] = A[0] + A[1] ...
    # The distance from s to t (s < t) is P[t] - P[s].
    # The distance from s to t (s > t) is (Total_Sum - P[s]) + P[t].
    
    # Calculate prefix sums: P[0]=0, P[1]=A_1, P[2]=A_1+A_2...
    # We only need the sums modulo M.
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Total sum of all A_i modulo M
    total_sum_mod = P[N]
    
    # We are looking for pairs (s, t) such that distance(s, t) % M == 0.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # For s > t: (total_sum_mod - P[s-1] + P[t-1]) % M == 0 
    #            => P[s-1] % M == (P[t-1] + total_sum_mod) % M
    
    # Let R be the list of P[i] % M for i = 0 to N-1
    R = P[:N]
    counts = Counter(R)
    
    # For s < t, the number of pairs is sum(count * (count - 1) // 2) for each unique remainder.
    # However, the problem asks for pairs (s, t) where s != t.
    # Let's evaluate the condition: distance(s, t) % M == 0.
    # This is equivalent to: P[t-1] - P[s-1] ≡ 0 (mod M) if s < t
    # And: total_sum_mod + P[t-1] - P[s-1] ≡ 0 (mod M) if s > t.
    
    # Let x = P[s-1] and y = P[t-1].
    # If s < t: y - x ≡ 0 (mod M)  => y ≡ x (mod M)
    # If s > t: total_sum_mod + y - x ≡ 0 (mod M) => x ≡ y + total_sum_mod (mod M)
    
    # Total pairs = Σ_{x ∈ R} (count(x) * count(x - total_sum_mod))
    # But we must exclude cases where s=t (not allowed) and handle the s < t vs s > t logic.
    
    # Let's refine:
    # For a fixed t, we need s such that:
    # 1. s < t and P[s-1] ≡ P[t-1] (mod M)
    # 2. s > t and P[s-1] ≡ P[t-1] + total_sum_mod (mod M)
    
    # Total pairs = Σ_{y ∈ R} [ (count(y) - 1) if total_sum_mod == 0 
    #                           else (count(y) if total_sum_mod != 0 and we count s > t 
    #                                 but we need to be careful about the mapping) ]
    
    # Correct logic:
    # For each t ∈ {1...N}, we want s ∈ {1...N}, s ≠ t such that:
    # If s < t: P[s-1] ≡ P[t-1] (mod M)
    # If s > t: P[s-1] ≡ P[t-1] + total_sum_mod (mod M)
    
    # Let's use the property:
    # Total = Σ_{y ∈ R} (count(y) * count((y + total_sum_mod) % M))
    # If total_sum_mod == 0, this counts s=t cases, so subtract N.
    # If total_sum_mod != 0, it never counts s=t because y != (y + total_sum_mod) % M.
    
    ans = sum(counts[y] * counts[(y + total_sum_mod) % M] for y in counts)
    
    # Subtract N if total_sum_mod is 0 because s cannot be t
    final_ans = ans - N if total_sum_mod == 0 else ans
    
    print(final_ans)

if __name__ == "__main__":
    solve()