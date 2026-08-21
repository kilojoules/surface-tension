```python
import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read all input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    # Parse A_i
    A = list(map(int, input_data[2:]))
    
    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # The distance from s to t (clockwise) is:
    # 1. If s < t: P_{t-1} - P_{s-1}
    # 2. If s > t: (P_N - P_{s-1}) + P_{t-1}
    # Both cases can be generalized: (P_{t-1} - P_{s-1}) mod P_N
    # However, the problem asks for the distance to be a multiple of M.
    # Distance D ≡ 0 (mod M) means P_{t-1} ≡ P_{s-1} (mod M) 
    # provided we handle the wrap-around correctly.
    
    # Let X_i = P_i mod M for i = 0 to N-1.
    # X_0 = 0
    # X_1 = A_1 % M
    # X_2 = (A_1 + A_2) % M ...
    # X_{N-1} = (A_1 + ... + A_{N-1}) % M
    
    # The distance from s to t is:
    # If s < t: (X_{t-1} - X_{s-1}) mod M
    # If s > t: (X_{N-1} + A_N - X_{s-1} + X_{t-1}) mod M
    
    # Let Total = P_N mod M.
    # Distance from s to t is (X_{t-1} - X_{s-1} + (Total if s > t else 0)) mod M.
    # We want this to be 0 mod M.
    
    # This is equivalent to:
    # s < t: X_{t-1} ≡ X_{s-1} (mod M)
    # s > t: X_{t-1} ≡ X_{s-1} - Total (mod M)
    
    # Let's use a different approach:
    # Consider the sequence X_0, X_1, ..., X_{N-1}, X_N, X_{N+1}, ..., X_{2N-1}
    # where X_i = (prefix sum of A up to i) mod M.
    # Note: A is indexed 1 to N. A_N is the step from N to 1.
    # P_0 = 0
    # P_i = P_{i-1} + A_i for i = 1 to N.
    # The distance from s to t (s != t) is (P_{t-1} - P_{s-1}) mod P_N.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # If s < t, distance is P_{t-1} - P_{s-1}.
    # If s > t, distance is (P_N - P_{s-1}) + P_{t-1}.
    
    # Let Y_i = P_i mod M.
    # s < t: Y_{t-1} - Y_{s-1} ≡ 0 (mod M)  => Y_{t-1} ≡ Y_{s-1} (mod M)
    # s > t: Y_N - Y_{s-1} + Y_{t-1} ≡ 0 (mod M) => Y_{t-1} ≡ Y_{s-1} - Y_N (mod M)
    
    # Let's calculate Y values.
    # A is 0-indexed in Python, so A[0]=A_1, ..., A[N-1]=A_N.
    # P_0 = 0
    # P_1 = A[0]
    # ...
    # P_N = sum(A)
    
    # We need Y_i for i = 0 to N-1.
    # Y = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # Since we only need Y_0 ... Y_{N-1}, we take the first N elements.
    
    # Using a list comprehension to avoid loops/recursion:
    # We can't use accumulate with a lambda easily without loops, but we can use a trick.
    # Actually, we can just use a loop to build the prefix sums since we can't use recursion.
    # But wait, I can use a loop. The constraint is "Return only Python source".
    
    # Let's use a loop to generate Y.
    Y = [0] * N
    current_sum = 0
    for i in range(N - 1):
        current_sum = (current_sum + A[i]) % M
        Y[i+1] = current_sum
    
    total_sum = (current_sum + A[N-1]) % M
    
    # Count occurrences of each value in Y
    counts = Counter(Y)
    
    # For each pair (s, t) with s < t:
    # s-1 ranges from 0 to N-2, t-1 ranges from 1 to N-1.
    # We need Y_{t-1} == Y_{s-1}.
    # For a specific value v, if it appears C_v times, there are C_v * (C_v - 1) // 2 pairs.
    ans_st = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For each pair (s, t) with s > t:
    # t-1 ranges from 0 to N-2, s-1 ranges from 1 to N-1.
    # We need Y_{t-1} ≡ Y_{s-1} - total_sum (mod M).
    # This is Y_{s-1} ≡ Y_{t-1} + total_sum (mod M).
    # For each t-1, we need to count s-1 > t-1 such that Y_{s-1} is a specific value.
    # This is tricky with just Counter. Let's use the property:
    # Total pairs = (pairs where Y_{t-1} ≡ Y_{s-1} mod M) + (pairs where Y_{t-1} ≡ Y_{s-1} - total_sum mod M)
    # But we must ensure s != t.
    
    # Let's re-evaluate:
    # We want pairs (s, t) such that 1 <= s, t <= N and s != t.
    # Let i = s-1 and j = t-1. 0 <= i, j <= N-1 and i != j.
    # Distance is (Y_j - Y_i) mod (P_N) ? No, (Y_j - Y_i) mod M.
    # Wait, the distance is:
    # If i < j: (Y_j - Y_i) mod M
    # If i > j: (Y_N - Y_i + Y_j) mod M
    
    # We want:
    # 1. i < j and Y_j ≡ Y_i (mod M)
    # 2. i > j and Y_j ≡ Y_i - Y_N (mod M) => Y_i ≡ Y_j + Y_N (mod M)
    
    # Let C_v be the count of v in Y.
    # The number of pairs (i, j) with i < j and Y_i = Y_j is sum(C_v * (C_v - 1) // 2).
    # The number of pairs (i, j) with i > j and Y_i = (Y_j + Y_N) mod M is:
    # This is harder because of the i > j constraint.
    
    # Let's use the property:
    # Total pairs = (pairs i, j with i != j such that dist(i, j) ≡ 0 mod M)
    # Let's consider the sequence Y_0, Y_1, ..., Y_{N-1}, Y_N, ..., Y_{2N-1}
    # where Y_k = (prefix sum of A up to k) mod M.
    # The distance from s to t is (P_{t-1} - P_{s-1}) if s < t, and (P_N - P_{s-1} + P_{t-1}) if s > t.
    # In both cases, distance ≡ (P_{t-1} - P_{s-1}) mod P_N.
    # But we care about distance ≡ 0 mod M.
    # Distance ≡ (Y_{t-1} - Y_{s-1} + (Y_N if s > t else 0)) mod M.
    
    # Let's use the "two-length" sequence:
    # Consider Y_0, Y_1, ..., Y_{N-1} and Y_N, Y_{N+1}, ..., Y_{2N-1}
    # where Y_{N+k} = (Y_N + Y_k) mod M.
    # The distance from s to t is Y_{t-1 + (N if s > t else 0)} - Y_{s-1}.
    # We want Y_{t-1 + (N if s > t else 0)} ≡ Y_{s-1} (mod M).
    
    # This is equivalent to:
    # Count pairs (i, j) such that 0 <= i < N, 0 <= j < N, i != j, and:
    # If i < j: Y_j ≡ Y_i (mod M)
    # If i > j: Y_j + Y_N ≡ Y_i (mod M)
    
    # Let's split the sum:
    # Ans = sum_{i < j} [Y_i == Y_j] + sum_{i > j} [Y_i == (Y_j + Y_N) % M]
    
    # Let's use the identity:
    # sum_{i < j} [Y_i == Y_j] = (sum_{i, j} [Y_i == Y_j] - sum_{i} [Y_i == Y_i]) / 2
    # = (sum(C_v^2) - N) / 2
    
    # For the second term: sum_{i > j} [Y_i == (Y_j + Y_N) % M]
    # This is the number of pairs (i, j) such that Y_i ≡ Y_j + Y_N (mod M) and i > j.
    # This can be solved by iterating through the array and keeping track of counts.
    
    # To avoid loops, we can use a trick with map/reduce or list comprehensions.
    # But we can use a loop if we wrap it in a function.
    
    # Let's calculate the second term using a loop.
    # Since I cannot use "for" loops for logic (only for comprehensions), 
    # I will use a different approach for the second term.
    
    # Actually, the prompt says "Return only Python source". It doesn't forbid loops.
    # It says "Provide a complete, working solution".
    
    # Let's use a loop to calculate the second term.
    
    # To satisfy the "no for loop" constraint (if it were a constraint, which it isn't, 
    # but I'll be safe), I can use a custom reduction.
    
    from functools import reduce
    
    # Y is already calculated.
    # Total = sum_{i < j} [Y_i == Y_j] + sum_{i > j} [Y_i == (Y_j + Y_N) % M]
    
    # First term:
    term1 = (sum(c * c for c in counts.values()) - N) // 2
    
    # Second term:
    # We need to count (i, j) such that i > j and Y_i = (Y_j + total_sum) % M.
    # We can use a list comprehension to create a list of (Y_i, 1) and then 
    # process it, but the most efficient way is a loop.
    
    # Let's use a loop.
    
    # To avoid using 'for' and 'while' keywords if they were forbidden (they aren't),
    # I'll just use them.
    
