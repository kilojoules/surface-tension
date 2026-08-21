The constraint to avoid explicit loops and recursion forces a shift from imperative programming to a functional style. While idiomatic Python often uses `for` loops for readability, utilizing high-order functions and comprehensions can lead to more concise, declarative code that describes *what* to calculate rather than *how* to step through the data. For this problem, I will use `itertools.accumulate` to calculate prefix sums of distances, list comprehensions for data transformation, and `collections.Counter` to efficiently count occurrences of remainders, allowing the final answer to be computed via a `sum()` over the frequency distribution.

```python
import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator/iterator to extract N and M, then the list A
    # Since we cannot use loops, we use next() to get N and M
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Calculate prefix sums of distances: P_i is distance from area 1 to area i+1
    # P_0 = 0, P_1 = A_1, P_2 = A_1 + A_2, ..., P_N = sum(A_i)
    # We only need the prefix sums modulo M.
    # accumulate([0] + a) gives [0, A_1, A_1+A_2, ...]
    prefix_sums_mod = [x % m for x in accumulate([0] + a)]
    
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1}) mod M
    # We want (P_{t-1} - P_{s-1}) % M == 0, which means P_{t-1} % M == P_{s-1} % M
    # There are N rest areas, so we consider prefix sums P_0, P_1, ..., P_{N-1}
    # P_N is the total length, which is used for pairs where t < s (wrapping around)
    
    # Let S = P_N % M. 
    # For s < t: distance is (P_{t-1} - P_{s-1}) % M
    # For s > t: distance is (P_N + P_{t-1} - P_{s-1}) % M
    
    # Let x_i = P_i % M for i in 0...N-1
    # Pair (s, t) with s < t is valid if x_{t-1} == x_{s-1}
    # Pair (s, t) with s > t is valid if (S + x_{t-1} - x_{s-1}) % M == 0
    # which means x_{s-1} == (S + x_{t-1}) % M
    
    s_total = prefix_sums_mod[n]
    x = prefix_sums_mod[:n]
    counts = Counter(x)
    
    # For s < t:
    # For each remainder r, there are counts[r] positions.
    # Number of pairs is counts[r] * (counts[r] - 1) // 2
    # However, the problem asks for pairs (s, t). 
    # If s < t, we need x_{t-1} - x_{s-1} = 0 mod M.
    # If s > t, we need S + x_{t-1} - x_{s-1} = 0 mod M => x_{s-1} = (S + x_{t-1}) mod M.
    
    # Contribution from s < t:
    # For a fixed remainder r, any two indices i < j yield a valid pair.
    # Total = sum(c * (c - 1) // 2 for c in counts.values())
    # Wait, the logic above is for unordered pairs. The problem asks for (s, t).
    # If s < t, we need x_{t-1} == x_{s-1}.
    # If s > t, we need x_{s-1} == (S + x_{t-1}) % M.
    
    # Let's refine:
    # We are looking for pairs (s, t) with s != t.
    # Case 1: s < t. Distance is P_{t-1} - P_{s-1}. 
    # Valid if P_{t-1} % M == P_{s-1} % M.
    # Case 2: s > t. Distance is (P_N - P_{s-1}) + P_{t-1}.
    # Valid if (S - x_{s-1} + x_{t-1}) % M == 0  => x_{s-1} == (S + x_{t-1}) % M.
    
    # For a fixed t, we need s < t such that x_{s-1} == x_{t-1}
    # AND we need s > t such that x_{s-1} == (S + x_{t-1}) % M.
    
    # Total = sum_{t=1 to N} [ (count of s < t where x_{s-1} == x_{t-1}) 
    #                        + (count of s > t where x_{s-1} == (S + x_{t-1}) % M) ]
    
    # This can be simplified:
    # Total = sum_{r=0 to M-1} [ (counts[r] * (counts[r]-1) // 2) 
    #                            + (counts[(S + r) % M] * counts[r]) ]
    # But wait, the second term (s > t) needs to be careful if (S + r) % M == r.
    # If S % M == 0, then (S + r) % M == r. 
    # Then for a fixed r, we have counts[r] choices for s and counts[r] choices for t.
    # The condition s > t means we choose 2 distinct indices and the larger one is s.
    # That is counts[r] * (counts[r] - 1) // 2.
    # If S % M != 0, then r != (S + r) % M, so we just have counts[r] * counts[(S + r) % M].
    
    # Let's double check:
    # If S % M == 0:
    # s < t: x_{s-1} == x_{t-1}  => counts[r]*(counts[r]-1)//2
    # s > t: x_{s-1} == x_{t-1}  => counts[r]*(counts[r]-1)//2
    # Total = sum(counts[r] * (counts[r]-1))
    
    # If S % M != 0:
    # s < t: x_{s-1} == x_{t-1}  => counts[r]*(counts[r]-1)//2
    # s > t: x_{s-1} == (S + x_{t-1}) % M => counts[(S+r)%M] * counts[r]
    # Wait, the s > t case: for a fixed t, we need s in {t+1, ..., N}.
    # This is harder to do without loops. Let's use the property:
    # Total = sum_{s < t} [x_{s-1} == x_{t-1}] + sum_{s > t} [x_{s-1} == (S + x_{t-1}) % M]
    
    # Let's use the fact that sum_{s < t} [x_{s-1} == x_{t-1}] = sum(c*(c-1)//2 for c in counts.values())
    # For the second term: sum_{t < s} [x_{s-1} == (S + x_{t-1}) % M]
    # This is sum_{t=1}^{N-1} sum_{s=t+1}^{N} [x_{s-1} == (S + x_{t-1}) % M]
    
    # Let's use a different approach for the second term.
    # Let y_t = (S + x_{t-1}) % M. We want to count pairs (s, t) with t < s and x_{s-1} == y_t.
    # This is equivalent to: for each r, count how many t have y_t = r and how many s have x_{s-1} = r,
    # then subtract the cases where s <= t.
    # But since we can't use loops, the most reliable way is to use the property:
    # Total = sum_{r=0}^{M-1} (counts[r] * (counts[r]-1)//2)  <-- for s < t
    #       + sum_{t=1}^{N} (count of s > t such that x_{s-1} == (S + x_{t-1}) % M)
    
    # Let's use the symmetry. 
    # Let f(r) = counts[r].
    # The number of pairs (s, t) with s < t and x_{s-1} = x_{t-1} = r is f(r)*(f(r)-1)//2.
    # The number of pairs (s, t) with s > t and x_{s-1} = (S + x_{t-1}) % M is:
    # For a fixed r1 = x_{t-1} and r2 = x_{s-1} = (S + r1) % M:
    # If r1 != r2, then any t with x_{t-1}=r1 and any s with x_{s-1}=r2 
    # will have either t < s or s < t.
    # This is not quite right because we need specifically s > t.
    
    # Correct logic:
    # Let's use the property that the total distance is S.
    # Distance(s, t) = (P_{t-1} - P_{s-1}) mod S_total
    # We want (P_{t-1} - P_{s-1}) % M == 0.
    # This is equivalent to P_{t-1} % M == P_{s-1} % M.
    # Wait, the problem says "minimum number of steps to walk clockwise".
    # That is simply (P_{t-1} - P_{s-1}) if s < t, and (P_N - P_{s-1} + P_{t-1}) if s > t.
    # Both are equivalent to (P_{t-1} - P_{s-1}) mod P_N.
    # But we need the result to be a multiple of M.
    # (P_{t-1} - P_{s-1}) % M == 0 is the condition.
    # This is simply x_{t-1} == x_{s-1} where x_i = P_i % M.
    # For any two distinct indices i, j in {0, ..., N-1}, if x_i == x_j,
    # then the clockwise distance from the rest area corresponding to i to the one corresponding to j
    # is (P_j - P_i) % P_N.
    # Since P_j - P_i is a multiple of M (because x_j == x_i), 
    # and P_N is not necessarily a multiple of M, we must be careful.
    # Actually, the distance is:
    # If s < t: Dist = P_{t-1} - P_{s-1}
    # If s > t: Dist = P_N - P_{s-1} + P_{t-1}
    # Condition: Dist % M == 0.
    # If s < t: x_{t-1} - x_{s-1} \equiv 0 \pmod M  => x_{t-1} == x_{s-1}
    # If s > t: S - x_{s-1} + x_{t-1} \equiv 0 \pmod M => x_{s-1} == (S + x_{t-1}) \pmod M
    
    # Let's use the fact that we can't use loops, so we use sum() and comprehensions.
    # For s < t, we need x_{s-1} == x_{t-1}.
    # For s > t, we need x_{s-1} == (S + x_{t-1}) % M.
    
    # Let's compute the s < t part:
    # ans1 = sum(c * (c - 1) // 2 for c in counts.values())
    
    # For the s > t part, we can't easily use the Counter because the order matters.
    # However, we can use the fact that:
    # Total pairs (s, t) with s != t such that x_{s-1