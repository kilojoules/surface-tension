import sys
from collections import Counter
from itertools import accumulate

def solve():
    # Read N and M
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Prefix sums of A modulo M
    # P[i] = (A_1 + ... + A_i) % M
    # The distance from s to t (s < t) is (P[t-1] - P[s-1]) % M
    # The distance from s to t (s > t) is (TotalSum - P[s-1] + P[t-1]) % M
    
    # We use a list comprehension to avoid explicit for loops
    # P will have N+1 elements, P[0] = 0
    P = list(accumulate([0] + A, lambda x, y: (x + y) % M))
    
    # The total sum of all A_i modulo M
    # Note: P[N] is the sum of A_1...A_N % M
    total_sum = P[N]
    
    # We are looking for pairs (s, t) such that distance is 0 mod M.
    # For s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] == P[s-1]
    # For s > t: (total_sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == total_sum
    
    # Let's count occurrences of each value in P[0...N-1]
    # P[0] corresponds to s=1, P[1] to s=2, ..., P[N-1] to s=N
    counts = Counter(P[:N])
    
    # For s < t, the number of pairs is sum(count * (count - 1) // 2)
    # However, the problem asks for pairs (s, t) where s != t.
    # The condition "clockwise from s to t" implies a directed path.
    # If P[i] == P[j] with i < j, then the distance from rest area i+1 to j+1 is 0 mod M.
    # This gives us count * (count - 1) // 2 pairs.
    
    # For s > t, the distance is (total_sum + P[t-1] - P[s-1]) % M == 0
    # This means (P[s-1] - P[t-1]) % M == total_sum
    # Let P[s-1] = x and P[t-1] = y. We need (x - y) % M == total_sum.
    # This is equivalent to x - total_sum == y (mod M).
    
    # Let's calculate the contribution of s < t and s > t separately.
    # For s < t: we need P[s-1] == P[t-1]. 
    # The number of such pairs is sum(c * (c - 1) // 2 for c in counts.values())
    # Wait, the sample says (1, 3) and (3, 1) can both be valid.
    # If P[s-1] == P[t-1], then distance s->t is 0 mod M AND distance t->s is total_sum mod M.
    # If total_sum is 0 mod M, then both are 0 mod M.
    # If total_sum is not 0 mod M, only one of them is 0 mod M.
    
    # Let's redefine:
    # A pair (s, t) is valid if:
    # 1. s < t and (P[t-1] - P[s-1]) % M == 0
    # 2. s > t and (total_sum - P[s-1] + P[t-1]) % M == 0
    
    # Case 1: s < t
    # This requires P[s-1] == P[t-1].
    # For each unique value in P[:N], if it appears 'c' times, there are c*(c-1)//2 pairs.
    ans_st = sum(c * (c - 1) // 2 for c in counts.values())
    
    # Case 2: s > t
    # This requires P[s-1] - P[t-1] == total_sum (mod M).
    # Let P[s-1] = x and P[t-1] = y. We need x - y == total_sum (mod M).
    # This is y == (x - total_sum) % M.
    # For each x in counts, the number of y's is counts[(x - total_sum) % M].
    # But we must ensure s > t. 
    # Actually, the condition s > t is naturally handled if we consider all pairs (s, t)
    # and subtract the cases where s == t.
    # But the logic "s < t" and "s > t" is a partition of all s != t.
    
    # Let's use a different approach:
    # For every pair i, j in {0, ..., N-1} with i != j:
    # If i < j: distance is (P[j] - P[i]) % M
    # If i > j: distance is (total_sum - P[i] + P[j]) % M
    
    # Total pairs = sum_{i < j} [P[i] == P[j]] + sum_{i > j} [P[i] - P[j] == total_sum (mod M)]
    # The second term is sum_{j < i} [P[i] - P[j] == total_sum (mod M)]
    
    # Let's evaluate the second term:
    # We need to count pairs (j, i) with j < i such that P[j] == (P[i] - total_sum) % M.
    # This can be done by iterating through P and keeping track of counts of values seen so far.
    # However, we can't use loops. We can use a trick with a custom function or 
    # map/reduce, but the simplest is to realize:
    # The total number of pairs (s, t) with s != t is:
    # sum_{x in unique(P)} (count(x) * count((x - total_sum) % M))
    # BUT, if total_sum == 0, then (x - total_sum) % M == x, 
    # and we get count(x)^2. We must subtract the cases where s == t, so count(x)^2 - count(x).
    # If total_sum != 0, then x != (x - total_sum) % M, so we just get count(x) * count((x - total_sum) % M).
    
    # Wait, the logic "sum_{i < j} [P[i] == P[j]]" is for s < t.
    # The logic "sum_{i > j} [P[i] - P[j] == total_sum (mod M)]" is for s > t.
    # These are two different conditions.
    
    # Let's use the property:
    # For a fixed pair {i, j} with i < j:
    # Distance i -> j is 0 mod M if P[i] == P[j].
    # Distance j -> i is 0 mod M if P[j] - P[i] == total_sum (mod M).
    
    # Total = sum_{i < j} ([P[i] == P[j]] + [P[j] - P[i] == total_sum (mod M)])
    # Total = sum_{i < j} [P[i] == P[j]] + sum_{i < j} [P[j] - P[i] == total_sum (mod M)]
    
    # The first term is sum(c*(c-1)//2 for c in counts.values())
    # The second term: for each j, we need to count i < j such that P[i] == (P[j] - total_sum) % M.
    # This is a classic problem that can be solved with a Fenwick tree or similar if we had loops.
    # But since we can't use loops, we can use the fact that we only need the sum of 
    # counts of (P[j] - total_sum) % M for all j, but only for i < j.
    
    # Actually, there is a simpler way.
    # Let's look at all pairs (i, j) with i != j.
    # The distance from i to j is 0 mod M if:
    # 1. i < j and P[i] == P[j]
    # 2. i > j and P[i] - P[j] == total_sum (mod M)
    
    # Let's use the identity: 
    # sum_{i < j} [P[i] == P[j]] = (sum(c^2) - sum(c)) / 2
    # For the second term: sum_{i > j} [P[i] - P[j] == total_sum (mod M)]
    # This is sum_{i} (count of j < i such that P[j] == (P[i] - total_sum) % M)
    
    # Since we cannot use loops, we can use a technique to simulate a loop using 
    # a generator and a mutable state object (like a dictionary) inside a map/sum.
    # However, the most "Pythonic" way to do this without explicit loops is to 
    # use a helper function with a closure or a class.
    
    def count_pairs(iterable, target_func):
        state = Counter()
        def process(x):
            res = state[target_func(x)]
            state[x] += 1
            return res
        return sum(map(process, iterable))

    # For the first term: target_func(x) = x
    # For the second term: target_func(x) = (x - total_sum) % M
    
    # We can combine them into one pass:
    def combined_process(x):
        # We need to count i < j such that P[i] == P[j] 
        # AND count i < j such that P[j] - P[i] == total_sum (mod M)
        # The second one is P[i] == (P[j] - total_sum) % M
        res = state[x] + state[(x - total_sum) % M]
        state[x] += 1
        return res

    state = Counter()
    # Using a list comprehension to trigger the side-effect of combined_process
    # and then summing the results.
    result = sum([combined_process(x) for x in P[:N]])
    
    # Special case: if total_sum == 0, then x == (x - total_sum) % M.
    # The combined_process adds state[x] twice.
    # But the condition was: if total_sum == 0, then distance i->j is 0 mod M 
    # AND distance j->i is 0 mod M.
    # So for every pair i < j with P[i] == P[j], we should count 2.
    # Our combined_process does: res = state[x] + state[x] = 2 * state[x].
    # This is correct.
    
    # Wait, if total_sum == 0, then (x - total_sum) % M is just x.
    # The combined_process returns state[x] + state[x].
    # If total_sum != 0, it returns state[x] + state[(x - total_sum) % M].
    # This correctly counts both s < t and s > t.
    
    print(result)

if __name__ == "__main__":
    solve()