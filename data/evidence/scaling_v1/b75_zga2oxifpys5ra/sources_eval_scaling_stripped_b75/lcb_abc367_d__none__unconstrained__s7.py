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

    # Let P_i be the prefix sum of A.
    # P_0 = 0
    # P_i = A_1 + ... + A_i
    # The distance from s to t (s < t) is (P_{t-1} - P_{s-1})
    # The distance from s to t (s > t) is (P_{N} - P_{s-1}) + P_{t-1}
    # We want distance % M == 0.
    
    # Calculate prefix sums modulo M
    # P = [0, A_1, A_1+A_2, ..., A_1+...+A_{N-1}]
    # Note: we only need prefix sums up to N-1 to define positions of rest areas 1 to N.
    # Let's use a list comprehension with accumulate.
    # We use a slice A[:-1] because the distance from i to i+1 is A_i.
    # The position of rest area k (1-indexed) is sum(A_1 ... A_{k-1}).
    prefixes = list(accumulate(A, lambda x, y: (x + y) % M, initial=0))
    # prefixes[i] is the distance from rest area 1 to rest area i+1.
    # We only care about the first N prefixes (0 to N-1).
    P = prefixes[:N]
    total_sum = sum(A) % M
    
    # For a pair (s, t) with s < t:
    # (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1] (mod M)
    # For a pair (s, t) with s > t:
    # (total_sum - P[s-1] + P[t-1]) % M == 0 => P[s-1] - P[t-1] == total_sum (mod M)
    
    # Count occurrences of each remainder
    from collections import Counter
    counts = Counter(P)
    
    # Case 1: s < t
    # For each remainder r, if there are c copies, there are c*(c-1)//2 pairs.
    ans = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # We need P[s-1] - P[t-1] \equiv total_sum (mod M)
    # Let P[t-1] = r, then P[s-1] = (r + total_sum) % M
    # The number of pairs is sum(count(r) * count((r + total_sum) % M))
    # However, we must exclude cases where s = t (though the problem says s != t).
    # If total_sum % M == 0, then P[s-1] == P[t-1], which is already covered by s < t
    # logic if we just did c*c. But the condition is s > t.
    # If total_sum % M == 0, then for a fixed r, we have c*(c-1)//2 pairs where s > t.
    # If total_sum % M != 0, then for each r, we have count(r) * count((r + total_sum) % M) pairs.
    
    # To avoid loops, we use a generator expression.
    # We iterate over the unique remainders present in the Counter.
    ans += sum(counts[r] * counts[(r + total_sum) % M] 
               for r in counts 
               if total_sum % M != 0)
    
    # If total_sum % M == 0, the s > t case is identical to the s < t case.
    if total_sum % M == 0:
        ans += sum(c * (c - 1) // 2 for c in counts.values())

    print(ans)

if __name__ == "__main__":
    solve()