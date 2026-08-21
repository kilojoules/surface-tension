import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    # and a list comprehension/slice for the rest.
    data = list(input_data)
    N = data[0]
    M = data[1]
    A = data[2:]

    # Let P_i be the distance from rest area 1 to rest area i.
    # P_1 = 0
    # P_2 = A_1
    # P_3 = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P_t - P_s)
    # The distance from s to t (s > t) is (Total_Sum - P_s + P_t)
    
    # Compute prefix sums: P_1, P_2, ..., P_N
    # accumulate([0] + A) gives 0, A_1, A_1+A_2, ...
    # We only need the first N prefix sums.
    P = list(accumulate([0] + A))[:N]
    
    # Total distance around the lake
    total_sum = sum(A)
    
    # We want (dist from s to t) % M == 0
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # If s > t: (total_sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == total_sum % M
    
    # Let R_i = P_i % M
    R = [p % M for p in P]
    counts = Counter(R)
    
    # For s < t, we need R_s == R_t. 
    # For a given remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t) where s != t.
    # Let's evaluate the condition: 
    # Dist(s, t) = (P_{t-1} - P_{s-1}) mod Total_Sum
    # We want Dist(s, t) % M == 0.
    # This is equivalent to:
    # If s < t: P_{t-1} ≡ P_{s-1} (mod M)
    # If s > t: P_{t-1} ≡ P_{s-1} - total_sum (mod M)
    
    # Let's redefine: we are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Let X_i = P_{i-1} % M for i = 1...N.
    # Condition: 
    # If s < t: X_t - X_s ≡ 0 (mod M)  => X_t ≡ X_s (mod M)
    # If s > t: (total_sum + X_t - X_s) ≡ 0 (mod M) => X_s - X_t ≡ total_sum (mod M)
    
    # Let S = total_sum % M.
    # We need to count pairs (s, t) such that:
    # 1. s < t and X_s ≡ X_t (mod M)
    # 2. s > t and X_s - X_t ≡ S (mod M)
    
    # For a fixed value v, let count(v) be the number of times v appears in X.
    # Contribution from s < t: sum(count(v) * (count(v) - 1) // 2)
    # Contribution from s > t: 
    # We need X_s - X_t ≡ S (mod M), which is X_s ≡ X_t + S (mod M).
    # For each t, we need to count s > t such that X_s ≡ (X_t + S) (mod M).
    # This is tricky without loops. Let's use the property:
    # Total pairs = Sum_{v} [count(v) * count((v + S) % M)]
    # But we must exclude cases where s = t (which would happen if S == 0).
    # If S == 0, the condition is always X_s ≡ X_t (mod M).
    # Then for each v, we have count(v) * (count(v) - 1) pairs.
    # If S != 0, the condition s < t and X_s == X_t is different from s > t and X_s - X_t == S.
    # Actually, the total count is simply:
    # Sum_{v=0 to M-1} (count(v) * count((v - S) % M))
    # Wait, let's re-verify:
    # For a fixed pair {s, t} with s < t:
    # Clockwise s -> t is (X_t - X_s) % M. This is 0 if X_t ≡ X_s (mod M).
    # Clockwise t -> s is (total_sum + X_s - X_t) % M. This is 0 if X_s - X_t ≡ -total_sum ≡ -S (mod M).
    # Note that -S ≡ M - S (mod M).
    # So for every pair {s, t} with s < t:
    # - It's a valid (s, t) if X_s ≡ X_t (mod M)
    # - It's a valid (t, s) if X_s - X_t ≡ S (mod M)
    
    # Total = Sum_{v=0 to M-1} [count(v) * count((v - S) % M)]
    # If S == 0, this counts pairs (s, t) where X_s == X_t, including s=t.
    # So we subtract N.
    # If S != 0, X_s == (X_s - S) % M is impossible, so s=t is never counted.
    
    S = total_sum % M
    # Use a generator expression to sum the products
    ans = sum(counts[v] * counts[(v - S) % M] for v in counts)
    
    # Subtract N if S == 0 because s cannot be equal to t
    final_ans = ans - N if S == 0 else ans
    
    print(final_ans)

if __name__ == "__main__":
    solve()