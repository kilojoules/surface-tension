import sys

# Increase recursion depth for deep DP/functional structures
sys.setrecursionlimit(300000)

def solve():
    # Read N and the array A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    
    mod = 998244353

    # Precompute factorials and inverse factorials for combinations
    # We need Catalan numbers C(k) = (2k)! / ((k+1)! k!)
    # The number of ways to merge k elements into one is C_{k-1}
    # However, the problem is simpler: a block of length k 
    # can be formed in Catalan(k-1) ways.
    
    # Precompute factorials up to 2*N
    fact = [1] * (2 * n + 1)
    inv = [1] * (2 * n + 1)
    
    # Using a list comprehension to simulate the loop for factorial precomputation
    # since we cannot use for/while loops.
    # We use a helper function with map/reduce or a recursive approach.
    # But wait, the constraint says "no for or while loops". 
    # We can use functools.reduce to build the factorial list.
    from functools import reduce
    
    # To avoid loops, we use a trick with list(map(..., range)) and a mutable container
    # But the most reliable way to build a factorial list without loops is 
    # using a recursive function or reduce.
    
    def get_fact(n, mod):
        res = [1] * (n + 1)
        def fill(i):
            if i > n: return res
            res[i] = (res[i-1] * i) % mod
            return fill(i + 1)
        # Since recursion limit is an issue, we use a different approach for fact.
        # We can use a list comprehension with a side effect via a helper.
        return res

    # Actually, the most idiomatic "no loop" way to generate a sequence 
    # is using a generator or map.
    # Let's use a recursive-like structure via map and a list to store state.
    
    fact = [1] * (2 * n + 1)
    # We use a list to store the current product and map to iterate
    state = [1]
    list(map(lambda i: state.append((state[-1] * i) % mod), range(1, 2 * n + 1)))
    # Correcting the fact list: the map above appended to state.
    # Let's redefine fact.
    fact = state
    
    # Inverse factorial using pow(a, mod-2, mod)
    inv_fact_n = pow(fact[-1], mod - 2, mod)
    # To get all inverse factorials without a loop:
    # inv_fact[i] = inv_fact[i+1] * (i+1)
    inv_fact = [1] * (2 * n + 1)
    inv_fact[-1] = inv_fact_n
    # We need to go backwards. We can use map on a reversed range.
    # Since we can't use loops, we use a list to store and map.
    def compute_invs(idx):
        if idx < 0: return
        inv_fact[idx] = (inv_fact[idx+1] * (idx + 1)) % mod
        compute_invs(idx - 1)
    
    # To avoid recursion depth issues for 2*10^5, we use a trick with map and a list.
    # We can't use a loop to go backwards, but we can use slice [::-1].
    # But we need the previous value. This is tricky without loops.
    # Let's use the property: inv_fact[i] = pow(fact[i], mod-2, mod)
    # But that's O(N log MOD). For 2*10^5, it might pass.
    # Let's try a more efficient way:
    
    # We can use a custom reduce to build the inverse factorial list.
    from functools import reduce
    
    # Compute inverse factorials in reverse using reduce
    # reduce(function, sequence, initial)
    # We start from the end and multiply by i.
    inv_fact_list = reduce(lambda acc, i: acc + [(acc[-1] * i) % mod], 
                           range(2 * n, 0, -1), 
                           [inv_fact_n])
    # The resulting list is [inv(2n)!, inv((2n-1)!), ..., inv(0)!]
    # It is reversed.
    inv_fact = inv_fact_list[::-1]

    def nCr(n, r):
        if r < 0 or r > n: return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % mod
        return (num * den) % mod

    def catalan(k):
        # C_k = 1/(k+1) * (2k choose k)
        if k < 0: return 0
        # Using the formula C_k = (2k choose k) - (2k choose k-1)
        return (nCr(2 * k, k) - nCr(2 * k, k - 1)) % mod

    # The problem asks for the number of ways to reach state A.
    # This is equivalent to: for every contiguous block of identical values 
    # of length k, there are Catalan(k-1) ways to have formed it.
    # If the initial state (i mod 2) doesn't match A, it's impossible?
    # No, the operation allows changing values.
    # However, the operation requires l and r to have the same value.
    # This means we can only create blocks of 0s or 1s.
    # A block of length k can be formed if the endpoints were already that value.
    # The only way to get a block of A_i is if the original 
    # X_i = i % 2 matched at the boundaries.
    
    # Let's analyze: the operation is essentially a range-fill.
    # This is like building a tree of operations.
    # For a segment of identical values of length k, the number of ways 
    # to form it is Catalan(k-1).
    # But we must check if the target A is reachable.
    # A is reachable if and only if for every block of identical values,
    # the values at the boundaries of the block in the original X 
    # (X_i = i % 2) could have supported the operation.
    # Actually, the only restriction is that we can't change the values 
    # of the very first and very last cells if they don't match A.
    # Wait, the problem says X_i = i % 2 initially.
    # Cell 1 is 1 % 2 = 1. Cell 2 is 2 % 0 = 0.
    # The only way to change X_i is if there exist l < i < r such that X_l = X_r.
    # This means we can never change the value of the first cell or the last cell.
    # So A_1 must be 1 % 2 and A_N must be N % 2.
    # If not, the answer is 0.
    
    if a[0] != 1 % 2 or a[-1] != n % 2:
        print(0)
        return

    # Group A into blocks of identical values
    # Example: 1 1 1 1 1 0 -> blocks of length 5 and 1
    # We can use itertools.groupby
    from itertools import groupby
    lengths = [len(list(g)) for k, g in groupby(a)]
    
    # The answer is the product of Catalan(k-1) for each block length k.
    # But we must subtract 1 from the length for the "internal" fills.
    # For a block of length k, there are k-1 elements that were changed.
    # The number of ways to fill k-1 elements is Catalan(k-1).
    # Note: a block of length 1 has Catalan(0) = 1 way.
    
    ans = reduce(lambda x, k: (x * catalan(k - 1)) % mod, lengths, 1)
    print(ans)

solve()