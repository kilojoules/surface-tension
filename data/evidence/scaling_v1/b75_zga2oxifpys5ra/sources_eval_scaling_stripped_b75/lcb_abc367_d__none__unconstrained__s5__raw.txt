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
    
    # Let P_i be the prefix sum of A: P_i = A_1 + ... + A_i
    # The distance from s to t (s < t) is P_{t-1} - P_{s-1}
    # The distance from s to t (s > t) is (P_N - P_{s-1}) + P_{t-1}
    # We want distance % M == 0.
    
    # Calculate prefix sums modulo M
    # P[i] will be the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A_1 % M, ..., P[N] = (A_1 + ... + A_N) % M
    P = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    
    # Let X_i = P[i] % M. 
    # For s < t, distance is (X_{t-1} - X_{s-1}) % M == 0  => X_{t-1} == X_{s-1}
    # For s > t, distance is (X_N - X_{s-1} + X_{t-1}) % M == 0 => X_{s-1} - X_{t-1} == X_N % M
    
    # We are looking for pairs (s, t) with 1 <= s, t <= N and s != t.
    # Let i = s-1 and j = t-1. Then 0 <= i, j < N.
    # If i < j: X_j - X_i \equiv 0 (mod M)
    # If i > j: X_N - X_i + X_j \equiv 0 (mod M) => X_i - X_j \equiv X_N (mod M)
    
    # Let's count occurrences of each value in X_0, ..., X_{N-1}
    # Note: P has N+1 elements, we only need the first N (0 to N-1)
    X = P[:N]
    X_N = P[N]
    
    # Use a dictionary to count frequencies of each remainder
    counts = {}
    for val in X:
        counts[val] = counts.get(val, 0) + 1
        
    # For a fixed remainder r, there are counts[r] indices.
    # Pairs (i, j) with i < j and X_i = X_j:
    # This is sum(c * (c - 1) // 2) for all c in counts.values()
    
    # Pairs (i, j) with i > j and X_i - X_j \equiv X_N (mod M):
    # This is sum(counts[r] * counts[(r - X_N) % M]) 
    # But we must exclude the case where i = j (which is already handled by s != t)
    # and we must handle the case where X_N == 0 carefully.
    
    # Total = Sum_{r} [ (counts[r]*(counts[r]-1)) // 2 ]  <-- for i < j
    #       + Sum_{r} [ counts[r] * counts[(r - X_N) % M] ] <-- for i > j
    # However, if X_N == 0, the second term counts pairs where X_i = X_j.
    # Since we need i > j, if X_N == 0, the second term is also sum(c * (c-1) // 2).
    
    # Let's refine:
    # For every pair of indices {i, j} with i < j:
    # 1. Check if X_j - X_i \equiv 0 (mod M)
    # 2. Check if X_N - X_i + X_j \equiv 0 (mod M) => X_i - X_j \equiv X_N (mod M)
    
    # If X_N == 0:
    # Both conditions become X_i == X_j.
    # Each pair {i, j} contributes 2 to the answer (one for s<t, one for s>t).
    # Total = sum(c * (c - 1))
    
    # If X_N != 0:
    # Condition 1: X_i == X_j
    # Condition 2: X_i - X_j == X_N (mod M)
    # These are mutually exclusive because X_N != 0.
    # Total = sum(c * (c - 1) // 2) + sum(counts[r] * counts[(r - X_N) % M])
    
    # Wait, the second term sum(counts[r] * counts[(r - X_N) % M]) already accounts 
    # for the i > j requirement because for every pair of values (r, r-X_N), 
    # one must be the 'start' and one the 'end'.
    # Actually, for any two indices i, j, they form a pair (s, t).
    # If i < j, we need X_i == X_j.
    # If i > j, we need X_i - X_j == X_N (mod M).
    
    # Let's use the property:
    # Ans = \sum_{i < j} [X_i == X_j] + \sum_{i > j} [X_i - X_j \equiv X_N (mod M)]
    # Ans = \sum_{r} (counts[r] * (counts[r]-1) // 2) + \sum_{r} (counts[r] * counts[(r - X_N) % M])
    # But the second sum is over all i, j such that X_i - X_j = X_N.
    # This includes cases where i < j and i > j.
    # Let's use the logic:
    # For every pair of distinct indices {i, j}, they can be (s, t) or (t, s).
    # One is clockwise i -> j, the other is j -> i.
    # Dist(i, j) = (X_j - X_i) % M
    # Dist(j, i) = (X_N - X_j + X_i) % M
    # We want Dist == 0.
    # (X_j - X_i) % M == 0  <=> X_i == X_j
    # (X_N - X_j + X_i) % M == 0 <=> X_j - X_i == X_N (mod M)
    
    # If X_N == 0:
    # Both are true if X_i == X_j.
    # Each pair {i, j} with X_i == X_j gives 2 pairs (s, t).
    # Ans = sum(c * (c - 1))
    
    # If X_N != 0:
    # X_i == X_j and X_j - X_i == X_N (mod M) cannot both be true.
    # Ans = sum(c * (c - 1) // 2) + sum(counts[r] * counts[(r - X_N) % M])
    # Wait, the second term is sum_{i, j} [X_j - X_i == X_N].
    # This automatically handles the i > j vs i < j because we are looking for 
    # the specific distance.
    # Let's re-evaluate:
    # For any two indices i, j (i != j):
    # The clockwise distance from i+1 to j+1 is (X_j - X_i) % M.
    # We want (X_j - X_i) % M == 0.
    # This is simply X_j == X_i.
    # But the problem says "minimum number of steps to walk clockwise".
    # If s < t, dist is P_{t-1} - P_{s-1}.
    # If s > t, dist is (P_N - P_{s-1}) + P_{t-1}.
    
    # Let i = s-1, j = t-1.
    # If i < j: (X_j - X_i) % M == 0  => X_i == X_j
    # If i > j: (X_N - X_i + X_j) % M == 0 => X_i - X_j == X_N (mod M)
    
    # Let's use the counts:
    # Part 1 (i < j): For each remainder r, there are counts[r] indices.
    # Number of pairs (i, j) with i < j and X_i = X_j is counts[r] * (counts[r] - 1) // 2.
    # Part 2 (i > j): We need X_i - X_j \equiv X_N (mod M).
    # For a fixed i, we need X_j \equiv X_i - X_N (mod M).
    # The number of such j's is counts[(X_i - X_N) % M].
    # However, this counts all j, including j > i and j = i.
    # We only want j < i.
    # This is tricky because the counts don't know about indices.
    
    # Let's use the property:
    # Total = \sum_{i < j} [X_i == X_j] + \sum_{i > j} [X_i - X_j \equiv X_N (mod M)]
    # Let's rewrite the second term:
    # \sum_{i > j} [X_i - X_j \equiv X_N (mod M)] = \sum_{i, j} [X_i - X_j \equiv X_N (mod M)] 
    #                                              - \sum_{i < j} [X_i - X_j \equiv X_N (mod M)]
    #                                              - \sum_{i = j} [X_i - X_j \equiv X_N (mod M)]
    
    # \sum_{i, j} [X_i - X_j \equiv X_N (mod M)] = \sum_{r} counts[r] * counts[(r - X_N) % M]
    # \sum_{i = j} [X_i - X_j \equiv X_N (mod M)] = N if X_N == 0 else 0
    # \sum_{i < j} [X_i - X_j \equiv X_N (mod M)] is the number of pairs i < j such that X_j - X_i \equiv -X_N (mod M).
    
    # This is getting complex. Let's simplify.
    # We want pairs (i, j) such that:
    # 1. i < j and X_i == X_j
    # 2. i > j and X_i - X_j == X_N (mod M)
    
    # Let's use a different approach.
    # For every pair of indices {i, j} with i < j:
    # Pair (s, t) = (i+1, j+1) is valid if X_i == X_j.
    # Pair (s, t) = (j+1, i+1) is valid if X_j - X_i == X_N (mod M).
    
    # Total = \sum_{i < j} [X_i == X_j] + \sum_{i < j} [X_j - X_i \equiv X_N (mod M)]
    
    # If X_N == 0:
    # Total = \sum_{i < j} [X_i == X_j] + \sum_{i < j} [X_j == X_i] = 2 * \sum (c*(c-1)//2) = \sum c*(c-1)
    
    # If X_N != 0:
    # Total = \sum_{i < j} [X_i == X_j] + \sum_{i < j} [X_j - X_i \equiv X_N (mod M)]
    # The first term is \sum (c*(c-1)//2).
    # The second term: for a fixed j, we need X_i == (X_j - X_N) % M for some i < j.
    # We can calculate this by iterating through the array and keeping track of counts of X_i seen so far