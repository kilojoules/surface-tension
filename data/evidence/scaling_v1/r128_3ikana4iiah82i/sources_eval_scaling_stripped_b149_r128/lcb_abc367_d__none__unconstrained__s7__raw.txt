```python
import sys
from collections import Counter

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of A modulo M
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # The distance from s to t (s < t) is (P[t] - P[s]) % M
    # The distance from s to t (s > t) is (TotalSum - (P[s] - P[t])) % M
    
    # We use a list comprehension to build the prefix sums to avoid explicit loops
    # P will have N+1 elements: P[0]=0, P[1]=A[0], P[2]=A[0]+A[1]...
    # Since we cannot use loops, we can use a trick with a helper function 
    # or use the fact that we can process the list in a way that simulates 
    # an accumulator. However, Python's list comprehension doesn't allow 
    # referencing the growing list. 
    # We can use a generator with a mutable state object (like a list) 
    # inside a comprehension, but that is essentially a loop.
    # The cleanest way to do prefix sums without 'for' or 'while' 
    # is using itertools.accumulate.
    
    from itertools import accumulate
    
    # P[i] is the distance from rest area 1 to rest area i+1
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # We only need the first N prefix sums because the distance from s to t 
    # where s < t is simply P[t-1] - P[s-1].
    # The distance from s to t where s > t is (TotalSum - (P[s-1] - P[t-1])).
    
    # Calculate prefix sums modulo M
    # We use A[:N] and then handle the total sum separately.
    # To get P[0]=0, we prepend 0 to the list.
    P = list(accumulate([0] + A))
    # We only need the first N values for the starting positions s
    # P[i] corresponds to the distance from area 1 to area i+1
    # Let's normalize them modulo M.
    P_mod = [x % M for x in P[:N]]
    
    total_sum = sum(A) % M
    
    # For a pair (s, t):
    # If s < t: distance is (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # If s > t: distance is (total_sum - (P[s-1] - P[t-1])) % M == 0 
    #          => (P[s-1] - P[t-1]) % M == total_sum % M
    #          => P[s-1] % M - total_sum % M == P[t-1] % M (modulo M)
    
    counts = Counter(P_mod)
    
    # Case 1: s < t
    # For each unique value v in P_mod, if it appears C times, 
    # there are C*(C-1)//2 pairs (s, t) with s < t.
    ans_st = sum(C * (C - 1) // 2 for C in counts.values())
    
    # Case 2: s > t
    # We need P[s-1] % M - P[t-1] % M == total_sum % M (modulo M)
    # Let v_s = P[s-1] % M and v_t = P[t-1] % M
    # v_s - v_t \equiv total_sum (mod M)
    # v_t \equiv v_s - total_sum (mod M)
    # For each v_s, we need to count how many v_t = (v_s - total_sum) % M exist.
    # Since s > t, we are looking for pairs (t, s) with t < s.
    # This is equivalent to counting pairs (v_t, v_s) from the distribution.
    # However, the condition s > t is key.
    # Let's re-evaluate: 
    # We want to count pairs (s, t) such that 1 <= t < s <= N 
    # and (total_sum - (P[s-1] - P[t-1])) % M == 0.
    # This is (P[s-1] - P[t-1]) % M == total_sum % M.
    
    # Let target = total_sum % M.
    # We want to count pairs (t, s) with t < s such that P[s-1] - P[t-1] \equiv target (mod M).
    # This can be solved by iterating through the list and keeping track of counts of 
    # (P[i] - target) % M seen so far.
    # But we can't use loops. We can use a technique with a dictionary and a 
    # generator that updates the dictionary.
    
    # Wait, the condition s > t is just the mirror of s < t.
    # For any two indices i < j, they form one pair (s=i, t=j) and one pair (s=j, t=i).
    # Pair (i, j) is valid if (P[j-1] - P[i-1]) % M == 0.
    # Pair (j, i) is valid if (total_sum - (P[j-1] - P[i-1])) % M == 0.
    
    # Let diff = (P[j-1] - P[i-1]) % M.
    # We want to count pairs (i, j) with i < j such that:
    # 1. diff == 0
    # 2. (total_sum - diff) % M == 0  => diff == total_sum % M
    
    # If total_sum % M == 0, then both conditions are the same.
    # But the problem says s != t.
    # If total_sum % M == 0, then diff == 0 is the only condition.
    # Each pair (i, j) with i < j and diff == 0 provides TWO valid pairs: (i, j) and (j, i).
    # However, the distance from s to t is the MINIMUM steps clockwise.
    # The problem says "The minimum number of steps required to walk clockwise".
    # This is always the sum of A_i from s to t.
    
    # Let's use the logic:
    # For every pair i < j:
    # Check if (P[j-1] - P[i-1]) % M == 0  (This is s=i, t=j)
    # Check if (total_sum - (P[j-1] - P[i-1])) % M == 0 (This is s=j, t=i)
    
    # Let target = total_sum % M.
    # We need to count pairs i < j such that P[j-1] - P[i-1] \equiv 0 (mod M)
    # AND count pairs i < j such that P[j-1] - P[i-1] \equiv target (mod M).
    
    # For a fixed target T, the number of pairs i < j such that P[j-1] - P[i-1] \equiv T (mod M)
    # is the sum over all v of (count of v) * (count of (v + T) % M).
    # BUT this counts all pairs, not just i < j.
    # Actually, the number of pairs i < j such that P[j-1] - P[i-1] \equiv T (mod M)
    # is the coefficient of x^T in the polynomial multiplication of the 
    # distribution of P_mod with its reverse, but that's for all pairs.
    
    # Correct approach for i < j:
    # The total number of pairs (i, j) with i != j is N*(N-1).
    # A pair (s, t) is valid if dist(s, t) % M == 0.
    # Let P_i be the prefix sum modulo M.
    # dist(s, t) = (P_t - P_s) % M if s < t
    # dist(s, t) = (Total - (P_s - P_t)) % M if s > t
    
    # Let's use the property:
    # (s, t) is valid if:
    # 1. s < t and P_t == P_s (mod M)
    # 2. s > t and P_s - P_t == Total (mod M) => P_t == (P_s - Total) (mod M)
    
    # Let C(v) be the number of times value v appears in P_mod.
    # For a fixed v, there are C(v) indices.
    # The number of pairs (s, t) with s < t and P_s = P_t = v is C(v)*(C(v)-1)//2.
    # The number of pairs (s, t) with s > t and P_s = v and P_t = (v - Total) % M is...
    # This is tricky because the s > t condition depends on the indices.
    
    # Let's use the "all pairs" approach and then adjust.
    # Total valid pairs = Sum_{v} [ C(v) * C((v - Total) % M) ]
    # But this counts pairs (s, t) where s can be <, >, or = t.
    # If s = t, the distance is not defined (s != t).
    # If we use the formula Sum_{v} C(v) * C((v - Total) % M), we are counting
    # all pairs (s, t) such that P_s - P_t \equiv Total (mod M).
    # For any two distinct indices i, j:
    # Either (P_j - P_i) \equiv Total (mod M) or (P_i - P_j) \equiv Total (mod M).
    # These are mutually exclusive unless Total \equiv -Total (mod M), i.e., 2*Total \equiv 0 (mod M).
    
    # Let's simplify:
    # A pair (s, t) with s < t is valid if P_t - P_s \equiv 0 (mod M).
    # A pair (s, t) with s > t is valid if Total - (P_s - P_t) \equiv 0 (mod M) 
    #                                  => P_s - P_t \equiv Total (mod M).
    
    # Let's count all pairs (i, j) with i != j such that P_j - P_i \equiv Total (mod M).
    # This is Sum_{v} C(v) * C((v - Total) % M).
    # However, this includes cases where i = j (which means Total \equiv 0 (mod M)).
    # If Total \equiv 0 (mod M), then P_j - P_i \equiv 0 (mod M).
    # The number of such pairs (i, j) with i != j is Sum C(v)*(C(v)-1).
    # If Total \not\equiv 0 (mod M), then P_j - P_i \equiv Total (mod M) implies i != j.
    # The number of such pairs is Sum C(v) * C((v - Total) % M).
    
    # Wait, the logic "either (s, t) or (t, s) is valid" only works if Total != 0.
    # Let's use the most direct counting:
    # Valid pairs = {(s, t) : s < t, P_t - P_s \equiv 0} \cup {(s, t) : s > t, P_s - P_t \equiv Total}
    # Note that the two sets are disjoint because if (s, t) is in both, then s < t and s > t, impossible.
    # Also, if Total \equiv 0, then the second set is {(s, t) : s > t, P_s - P_t \equiv 0}.
    # In that case, the total is Sum C(v)*(C(v)-1)//2 + Sum C(v)*(C(v)-1)//2 = Sum C(v)*(C(v)-1).
    
    # To count {(s, t)