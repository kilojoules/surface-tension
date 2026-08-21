```python
import sys
from functools import reduce

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))

    # Calculate prefix sums modulo M
    # P[i] is the distance from rest area 1 to rest area i+1
    # P[0] = 0, P[1] = A1 % M, P[2] = (A1 + A2) % M, ...
    # We use reduce to avoid loops and maintain a functional style
    # accumulator: (current_prefix_sum, list_of_prefix_sums)
    prefix_sums = reduce(
        lambda acc, x: ( (acc[0] + x) % M, acc[1] + [(acc[0] + x) % M] ),
        A,
        (0, [0])
    )[1]

    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # We want (P[t-1] - P[s-1]) % M == 0, which means P[t-1] == P[s-1]
    # For s > t, the distance is (Total_Sum - P[s-1] + P[t-1]) % M
    # We want (Total_Sum + P[t-1] - P[s-1]) % M == 0
    
    # Let L = Total_Sum % M
    L = prefix_sums[-1]
    # Note: prefix_sums has N+1 elements. P[0]...P[N].
    # But the problem defines N rest areas. 
    # The distance from i to i+1 is A_i.
    # Let's redefine: P[i] is distance from area 1 to area i+1.
    # P[0] = 0 (Area 1)
    # P[1] = A_1 (Area 2)
    # ...
    # P[N-1] = A_1 + ... + A_{N-1} (Area N)
    # Total = A_1 + ... + A_N (Area 1 again)
    
    # Correcting the prefix sum logic to match the N areas:
    # We only need P[0] to P[N-1].
    P = prefix_sums[:-1] 
    Total = prefix_sums[-1]
    
    # Count occurrences of each remainder
    # Using a list as a frequency array since M <= 10^6
    counts = [0] * M
    for x in P:
        counts[x] += 1
    
    # For a fixed s and t (s != t):
    # If s < t: distance is (P[t-1] - P[s-1]) % M. 
    # This is 0 if P[t-1] == P[s-1].
    # For each remainder r, there are counts[r] positions.
    # Number of pairs (s, t) with s < t is counts[r] * (counts[r] - 1) // 2.
    # However, the problem asks for pairs (s, t) where s != t.
    # If s < t and P[s-1] == P[t-1], that's one pair.
    # If s > t and (Total + P[t-1] - P[s-1]) % M == 0, that's another.
    
    # Let's use the property: 
    # Pair (s, t) is valid if:
    # 1. s < t and P[t-1] ≡ P[s-1] (mod M)
    # 2. s > t and P[s-1] ≡ P[t-1] + Total (mod M)
    
    # For a fixed remainder r:
    # Number of s < t such that P[s-1] = P[t-1] = r is counts[r] * (counts[r] - 1) // 2
    # But we need to count pairs (s, t).
    # For every pair of indices {i, j} with i < j:
    # - Pair (i+1, j+1) is valid if P[j] - P[i] ≡ 0 (mod M)
    # - Pair (j+1, i+1) is valid if Total + P[i] - P[j] ≡ 0 (mod M)
    
    # Total valid pairs = Sum_{r=0 to M-1} [
    #    (counts[r] * (counts[r]-1) // 2)  <-- for s < t
    #    + (counts[(r + Total) % M] * counts[r] if r != (r + Total) % M else 0) 
    #    Wait, the second part is simpler:
    #    For every pair i < j, (j+1, i+1) is valid if P[j] - P[i] ≡ Total (mod M).
    # ]
    
    # Let's refine:
    # A pair (s, t) with s < t is valid if P[t-1] ≡ P[s-1] (mod M)
    # A pair (s, t) with s > t is valid if P[s-1] ≡ P[t-1] + Total (mod M)
    
    # Part 1: s < t
    # For each r, there are counts[r] indices. Pairs: counts[r] * (counts[r] - 1) // 2
    # But we must multiply by 2? No, s < t is specific.
    # Actually, for any two indices i, j (i < j), 
    # (i+1, j+1) is valid if P[i] == P[j]
    # (j+1, i+1) is valid if P[j] - P[i] == Total (mod M)
    
    # Total = Sum_{r=0 to M-1} (counts[r] * (counts[r]-1) // 2) 
    #       + Sum_{r=0 to M-1} (counts[r] * counts[(r + Total) % M])
    # Note: if Total % M == 0, the second sum counts pairs where P[j] == P[i].
    # If Total % M == 0, then (j+1, i+1) is valid whenever (i+1, j+1) is.
    # So we get 2 * (counts[r] * (counts[r]-1) // 2) = counts[r] * (counts[r]-1).
    
    # If Total % M != 0:
    # s < t: P[t-1] == P[s-1] -> counts[r] * (counts[r]-1) // 2
    # s > t: P[s-1] == P[t-1] + Total -> counts[(r + Total)%M] * counts[r] 
    # Wait, the s > t case: for a fixed pair i < j, we check if P[j] - P[i] ≡ Total.
    # This is different from the s < t case.
    
    # Let's use the logic:
    # For all pairs i < j:
    # Count 1 if P[j] ≡ P[i] (mod M)
    # Count 1 if P[j] - P[i] ≡ Total (mod M)
    
    # Sum_{i < j} [P[j] == P[i]] = Sum_{r} (counts[r] * (counts[r]-1) // 2)
    # Sum_{i < j} [P[j] - P[i] == Total] = ?
    # This is harder because it depends on the order.
    # Let's use the property: 
    # Total pairs = (Number of s, t such that dist(s, t) ≡ 0 mod M)
    # dist(s, t) = (P[t-1] - P[s-1]) % M if s < t
    # dist(s, t) = (Total + P[t-1] - P[s-1]) % M if s > t
    
    # Let's use a different approach:
    # For each s, we want t != s such that dist(s, t) ≡ 0 (mod M).
    # If s < t, we need P[t-1] ≡ P[s-1] (mod M).
    # If s > t, we need P[t-1] ≡ P[s-1] - Total (mod M).
    
    # Let's use the counts of P[i] for i in 0...N-1.
    # For a fixed s (index i), we need:
    # t > s: P[t-1] ≡ P[i] (mod M). There are (counts[P[i]] - 1) such t's in total,
    # but some are < s.
    # t < s: P[t-1] ≡ P[i] - Total (mod M).
    
    # This is still tricky. Let's simplify:
    # A pair (s, t) is valid if:
    # (P[t-1] - P[s-1]) % M == 0  (for s < t)
    # (Total + P[t-1] - P[s-1]) % M == 0 (for s > t)
    
    # Let's use the fact that we can double the array: P_0, P_1, ..., P_{N-1}, P_N, ... P_{2N-1}
    # where P_{N+i} = P_i + Total.
    # The distance from s to t (clockwise) is P_{t-1} - P_{s-1} if we pick t > s.
    # For each s \in {1, ..., N}, we seek t \in {s+1, ..., s+N-1} such that
    # P_{t-1} \equiv P_{s-1} (mod M).
    
    # Let's create the sequence P_i % M for i = 0 to 2N-1.
    # P_i for i < N is already calculated.
    # For i \in {N, ..., 2N-1}, P_i = (P_{i-N} + Total) % M.
    
    # We want to count pairs (i, j) such that 0 <= i < N and i < j < i + N
    # and P_j \equiv P_i (mod M).
    
    # This is equivalent to:
    # For each i \in {0, ..., N-1}, count j \in {i+1, ..., i+N-1} such that P_j \equiv P_i (mod M).
    
    # Let's use the frequency of each remainder in the range [0, 2N-1].
    # But the range [i+1, i+N-1] shifts.
    # However, the total count of P_j \equiv P_i in the range [i+1, i+2N-1] is just
    # the number of times P_i appears in the doubled array, minus 1 (itself).
    # But we only care about the range of length N-1.
    
    # Let's use the property:
    # In the doubled array P_0, ..., P_{2N-1}, the value P_i appears at most twice.
    # P_j \equiv P_i (mod M) for j \in {i+1, ..., i+N-1} if:
    # 1. P_j \equiv P_i (mod M) for some j \in {i+1, ..., N-1}
    # 2. P_j \equiv P_i (mod M) for some j \in {N, ..., i+N-1}
    
    # Note that P_j \equiv P_i (mod M) for j \in {N, ..., 2N-1} 
    # is equivalent to P_{j-N} + Total \equiv P_i (mod M).
    
    # Let's use the counts of P_i % M for i \in {0, ..., N-1}.
    # For a fixed i, we want:
    # 1. j \in {i+1, ..., N-1} such that P_j \equiv P_i (mod M)
    # 2. j \in {N, ..., i+N-1} such that P_{j-N} + Total \equiv P_i (mod M)
    
    #