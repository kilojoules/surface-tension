import sys
from itertools import accumulate
from collections import Counter

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use next() to extract N and M from the iterator
    # Since we cannot use loops, we use a iterator-based approach
    it = input_data
    n = next(it)
    m = next(it)
    a = list(it)
    
    # Let P[i] be the distance from rest area 1 to rest area i+1.
    # P[0] = 0
    # P[1] = A_1
    # P[2] = A_1 + A_2 ...
    # The distance from s to t (s < t) is (P[t-1] - P[s-1])
    # The distance from s to t (s > t) is (Total_Sum - P[s-1]) + P[t-1]
    
    # Calculate prefix sums modulo M
    # accumulate provides P[1], P[2]... P[N]
    # We prepend 0 to represent P[0]
    prefix_sums = [0] + list(accumulate(a))
    mods = [p % m for p in prefix_sums]
    
    # Total distance around the lake modulo M
    total_sum_mod = mods[-1] % m
    
    # We are looking for pairs (s, t) with s != t such that:
    # If s < t: (P[t-1] - P[s-1]) % M == 0  => P[t-1] % M == P[s-1] % M
    # If s > t: (Total_Sum - P[s-1] + P[t-1]) % M == 0 => (P[s-1] - P[t-1]) % M == Total_Sum % M
    
    # Count occurrences of each remainder modulo M
    counts = Counter(mods[:-1]) # We only care about P[0]...P[N-1]
    
    # For s < t:
    # For each remainder r, if there are c instances, there are c*(c-1)//2 pairs.
    # However, the problem asks for pairs (s, t). 
    # Let's evaluate the condition: Dist(s, t) % M == 0
    # Let X_i = P[i-1] % M for i = 1 to N.
    # For s < t: (X_t - X_s) % M == 0  => X_t == X_s
    # For s > t: (Total_Sum - X_s + X_t) % M == 0 => (X_s - X_t) % M == Total_Sum % M
    
    # Let's refine:
    # We have a list of values V = [P[0]%M, P[1]%M, ..., P[N-1]%M]
    # We want pairs (i, j) with 0 <= i, j < N and i != j such that:
    # If i < j: (V[j] - V[i]) % M == 0
    # If i > j: (Total_Sum - V[i] + V[j]) % M == 0
    
    # This is equivalent to:
    # 1. i < j and V[i] == V[j]
    # 2. i > j and (V[i] - V[j]) % M == Total_Sum % M
    
    # Let C be the Counter of V.
    # For a fixed remainder r, there are C[r] indices.
    # The number of pairs (i, j) with i < j and V[i] == V[j] is C[r]*(C[r]-1)//2.
    # The number of pairs (i, j) with i > j and (V[i] - V[j]) % M == Total_Sum % M:
    # This is equivalent to V[i] - Total_Sum % M == V[j] (mod M).
    # Let target_r = (r - total_sum_mod) % m.
    # We want to count pairs (i, j) such that i > j and V[i] == r and V[j] == target_r.
    
    # To avoid loops and complex indexing, we can use the property:
    # Total pairs = Sum_{r} (count(r) * count(r - Total_Sum % M))
    # But we must exclude cases where the condition i < j and i > j overlap 
    # or when i == j.
    
    # Let's use a different approach:
    # For each i in 0..N-1, we want to count j != i such that:
    # if i < j: V[j] == V[i]
    # if i > j: V[j] == (V[i] - Total_Sum) % M
    
    # Let's use the property that we can iterate over the unique remainders in the Counter.
    # For a fixed r1 and r2 such that (r1 - r2) % M == Total_Sum % M:
    # If r1 == r2 (which happens if Total_Sum % M == 0):
    #   Any pair (i, j) with V[i]==V[j]==r1 and i != j satisfies the condition.
    #   Number of pairs = C[r1] * (C[r1] - 1)
    # If r1 != r2:
    #   We need to count pairs (i, j) such that:
    #   (i < j and V[i]==r1 and V[j]==r1) OR (i > j and V[i]==r1 and V[j]==r2)
    #   Wait, the condition is:
    #   s < t: V[t-1] == V[s-1]
    #   s > t: V[s-1] - V[t-1] == Total_Sum % M
    
    # Let's use the indices. Let V = [P[0]%M, ..., P[N-1]%M].
    # We want pairs (i, j) 0 <= i, j < N, i != j such that:
    # (i < j and V[j] == V[i]) OR (i > j and (V[i] - V[j]) % M == total_sum_mod)
    
    # Let's simplify the second condition: i > j and V[j] == (V[i] - total_sum_mod) % M.
    # For a fixed i, we need:
    # 1. Count j > i such that V[j] == V[i]
    # 2. Count j < i such that V[j] == (V[i] - total_sum_mod) % M
    
    # Let's use a running count to solve this in one pass using a custom function and reduce or a list comprehension.
    # Since we can't use loops, we can use a trick with a mutable object and map.
    
    state = {'count': 0, 'seen': Counter()}
    
    def process(v):
        # For the current index i, V[i] is the value.
        # j < i: we need V[j] == (v - total_sum_mod) % m
        # j > i: we need V[j] == v
        # The total number of j > i such that V[j] == v is (Total_C[v] - state['seen'][v] - 1)
        # The total number of j < i such that V[j] == (v - total_sum_mod) % m is state['seen'][(v - total_sum_mod) % m]
        
        # We need Total_C, which is the Counter of all V.
        # This will be handled by closures.
        return 0

    # To implement this without loops, we use a helper function and map.
    # But we need Total_C first.
    total_c = Counter(mods[:-1])
    
    def calculate_pairs(v):
        # This function will be called for each v in mods[:-1]
        # It updates the running 'seen' counter and adds to the total
        res = state['seen'][(v - total_sum_mod) % m] + (total_c[v] - state['seen'][v] - 1)
        state['seen'][v] += 1
        return res

    # Use map to trigger the side effects and sum the results
    ans = sum(map(calculate_pairs, mods[:-1]))
    print(ans)

if __name__ == "__main__":
    solve()