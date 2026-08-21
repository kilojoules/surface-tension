```python
import sys
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of distances modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A1%M, (A1+A2)%M, ..., (A1+...+AN)%M]
    # We use accumulate to avoid loops and map to keep values within modulo M
    P = list(accumulate((x % M for x in A), initial=0))
    
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1] (mod M)
    # For s < t, we are looking for pairs in P[0...N-1] with the same value.
    # Note: P[N] is the total distance around the lake.
    
    # Let's count occurrences of each remainder in P[0...N-1]
    # We use a list as a frequency array since M is at most 10^6
    counts = [0] * M
    for i in range(N):
        counts[P[i]] += 1
    
    # For any two indices i, j in {0...N-1} where i < j, 
    # if P[i] == P[j], then the distance from area i+1 to j+1 is a multiple of M.
    # Number of such pairs is sum(count * (count - 1) // 2)
    ans = sum(c * (c - 1) // 2 for c in counts)
    
    # Now consider the wrap-around cases: s > t
    # Distance from s to t is (Total_Dist - (P[s-1] - P[t-1])) % M
    # Let Total = P[N]. We want (Total - P[s-1] + P[t-1]) % M == 0
    # Which means P[s-1] - P[t-1] == Total (mod M)
    # Or P[t-1] == (P[s-1] - Total) (mod M)
    
    # Let Total = P[N]. For a fixed s, we need t < s such that 
    # P[t-1] == (P[s-1] - Total) % M.
    # This is equivalent to counting pairs (i, j) with 0 <= i < j < N
    # such that P[i] == (P[j] - Total) % M.
    # However, the simple combinatorial approach above only works if Total % M == 0.
    # If Total % M != 0, the "wrap-around" pairs are different from "forward" pairs.
    
    # Correct logic for all pairs (s, t) where s != t:
    # Distance(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # Distance(s, t) = (P[N] - P[s-1] + P[t-1]) % M if s > t
    
    # Let's redefine: we want (P[t-1] - P[s-1]) % M == 0 for s < t
    # AND (P[N] + P[t-1] - P[s-1]) % M == 0 for s > t.
    
    # Let Total = P[N] % M.
    # For s < t: P[t-1] == P[s-1] (mod M)
    # For s > t: P[t-1] == (P[s-1] - Total) (mod M)
    
    # Let's use the frequency array of P[0...N-1].
    # For a fixed value v, there are counts[v] indices.
    # Forward pairs: counts[v] * (counts[v] - 1) // 2
    # Backward pairs: 
    # For each j, we need i < j such that P[i] == (P[j] - Total) % M.
    # This is tricky because the condition depends on the index.
    # Actually, the total number of pairs is:
    # Sum_{i=0 to N-1} Sum_{j=0 to N-1, j!=i} [Dist(i+1, j+1) % M == 0]
    
    # Let's use the property: 
    # Dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # Dist(s, t) = (P[N] + P[t-1] - P[s-1]) % M if s > t
    
    # Total pairs = Sum_{0 <= i < j < N} [P[j] == P[i]] 
    #             + Sum_{0 <= j < i < N} [P[j] == (P[i] - Total) % M]
    
    # The second term: for a fixed i, we need count of j < i such that P[j] == (P[i] - Total) % M.
    # This can be solved by iterating through P and maintaining a running count.
    
    # Since we cannot use loops, we can use a trick with a generator or map.
    # But wait, the second term is just:
    # Sum_{i=0 to N-1} (count of P[j] == (P[i] - Total) % M for j < i)
    
    # Let's use a different approach for the second term.
    # Let Total = P[N] % M.
    # We want pairs (i, j) with 0 <= j < i < N such that P[j] == (P[i] - Total) % M.
    # This is equivalent to P[i] - P[j] == Total (mod M).
    
    # If Total == 0:
    # The condition is P[i] == P[j]. 
    # Total pairs = 2 * (counts[v] * (counts[v] - 1) // 2) = counts[v] * (counts[v] - 1)
    
    # If Total != 0:
    # Forward: P[j] == P[i] (i < j)
    # Backward: P[j] == (P[i] - Total) % M (j < i)
    # Note that if P[j] == (P[i] - Total) % M, then P[i] == (P[j] + Total) % M.
    # This means for every pair (i, j) that satisfies the backward condition,
    # it's a pair where one value is v and the other is (v + Total) % M.
    # Specifically, if we have counts[v] and counts[(v + Total) % M],
    # the total number of pairs (i, j) with i != j such that P[i] - P[j] == Total (mod M)
    # is simply counts[v] * counts[(v + Total) % M].
    # Wait, that's for all i, j. But we have the constraint j < i.
    # Actually, for any two indices i, j, either (i < j and Dist is P[j]-P[i])
    # or (i > j and Dist is Total + P[j]-P[i]).
    # So we want:
    # 1. i < j and P[j] - P[i] == 0 (mod M)
    # 2. i > j and P[j] - P[i] == -Total (mod M) => P[i] - P[j] == Total (mod M)
    
    # Let's use the property:
    # For any pair {i, j} with i < j:
    # Clockwise i+1 to j+1 is (P[j] - P[i]) % M
    # Clockwise j+1 to i+1 is (P[N] + P[i] - P[j]) % M
    
    # We want:
    # (P[j] - P[i]) % M == 0  OR  (P[N] + P[i] - P[j]) % M == 0
    
    # Case 1: P[j] == P[i] (mod M)
    # Case 2: P[j] - P[i] == P[N] (mod M)
    
    # If P[N] % M == 0, Case 1 and Case 2 are the same.
    # Each pair {i, j} contributes 2 to the answer if P[i] == P[j].
    # Total = sum(counts[v] * (counts[v] - 1))
    
    # If P[N] % M != 0, Case 1 and Case 2 are mutually exclusive.
    # Case 1: sum(counts[v] * (counts[v] - 1) // 2)
    # Case 2: sum_{v=0 to M-1} (counts[v] * counts[(v + P[N]) % M])
    # Wait, Case 2 is: for all i < j, P[j] - P[i] == P[N] (mod M).
    # This is not simply counts[v] * counts[v+Total].
    # That would be for all i, j. 
    # But for any two values v and (v + Total) % M, one must appear first.
    # If we have indices of v: {i1, i2, ...} and indices of v+Total: {j1, j2, ...}
    # A pair (i, j) satisfies Case 2 if i < j and P[j] - P[i] == Total.
    # This is exactly the number of pairs (i, j) with i < j such that P[i]=v and P[j]=v+Total.
    # This is NOT simply counts[v] * counts[v+Total].
    # Actually, it is! Because for any i where P[i]=v and any j where P[j]=v+Total,
    # either i < j (Case 2) or i > j (Case 1 is impossible, but the distance j+1 to i+1 
    # would be (P[N] + P[i] - P[j]) % M = (Total + v - (v+Total)) % M = 0).
    # So for any pair of indices {i, j} such that P[i]=v and P[j]=v+Total, 
    # exactly one of the two clockwise directions will be a multiple of M.
    
    # Therefore, if Total % M != 0:
    # Ans = sum(counts[v]*(counts[v]-1)//2) + sum(counts[v] * counts[(v + Total) % M])
    # Wait, the second term is sum_{v} (counts[v] * counts[(v + Total) % M]).
    # But this counts pairs (i, j) where P[i]=v and P[j]=v+Total.
    # For each such pair, if i < j, then dist(i+1, j+1) = Total (not 0).
    # But dist(j+1, i+1) = (Total + P[i] - P[j]) = (Total + v - (v+Total)) = 0.
    # So each pair of indices {i, j} with P[i]=v and P[j]=v+Total contributes exactly 1.
    
    # Let's re-verify:
    # If P[N] % M == 0:
    # Pair {i, j} is good if P[i] == P[j]. Both directions are 0 mod M.
    # Ans = sum(counts[v] * (counts[v] - 1))
    
    # If P[N] % M = T != 0:
    # Pair {i, j} (i < j) is good if:
    # 1. P[j] - P[i] == 0 (mod M)  => P[i] == P[j]
    # 2. T + P[i] - P[j] == 0 (mod M) => P[j] - P[i] == T (mod M)
    # These are mutually exclusive since T != 0.
    # For a fixed v, pairs satisfying (1) are counts[v]*(counts[v]-1)//2.
    # For a fixed v, pairs satisfying (