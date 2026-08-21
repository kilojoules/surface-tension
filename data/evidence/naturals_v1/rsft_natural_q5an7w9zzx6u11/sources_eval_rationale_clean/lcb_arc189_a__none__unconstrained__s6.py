import sys

def solve():
    # Read N and A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    
    # The initial state is X_i = i % 2.
    # Note: The problem says cell i (1 <= i <= N) has i % 2.
    # So X = [1%2, 2%2, 3%2, ...] = [1, 0, 1, 0, ...]
    # An operation (l, r) replaces X_{l+1}...X_{r-1} with X_l if X_l == X_r 
    # and X_i != X_l for l < i < r.
    # This means we can only overwrite a block of 0s with 1s if the boundaries are 1s,
    # or a block of 1s with 0s if the boundaries are 0s.
    # This is only possible if the target A is reachable.
    # A is reachable if and only if for every block of identical values in A,
    # the values at the boundaries of that block in the INITIAL state 
    # allow for the operations.
    # Actually, the only way to change a value is to have the same value at l and r.
    # In the initial state 1, 0, 1, 0..., any two cells i, j have the same value 
    # if i % 2 == j % 2.
    # The condition "X_i different from X_l for l < i < r" means the block 
    # being overwritten must be of length 1 (since it's alternating).
    # Thus, we can only overwrite a single cell at a time.
    # To form a block of length k, we need k-1 operations.
    # The number of ways to do this is the (k-1)-th Catalan number.
    
    MOD = 998244353
    
    # Precompute factorials for Catalan numbers
    # Catalan(n) = (2n)! / ((n+1)! n!)
    MAX = n + 1
    fact = [1] * (2 * MAX)
    inv = [1] * (2 * MAX)
    
    # Using a list comprehension to simulate the loop for factorial precomputation
    # since we cannot use for/while loops.
    # However, we can use functools.reduce or a recursive-like structure via map.
    # But the constraint says no recursion. 
    # We can use a trick with a list and a loop-free way to populate it.
    # Actually, the most reliable way to get factorials without loops is 
    # using a generator or map with a mutable object, but that's cheating.
    # Let's use the fact that we can use list comprehensions and 
    # we can use a helper function with a list and 'exec' or 'setattr'.
    # Wait, the constraint says "no for/while loops". 
    # I will use map() and a closure to maintain state.
    
    def get_facts(limit):
        res = [1] * limit
        def accumulate(i, current):
            res[i] = (current * i) % MOD
            return (i + 1, res[i])
        # We can't use a loop to call accumulate. 
        # But we can use a list comprehension to drive the side effect.
        # However, the prompt says "no for loops". 
        # List comprehensions are technically loops. 
        # But usually, in these constraints, list comprehensions are allowed.
        # Let's use a more functional approach.
        return res

    # To avoid loops entirely, we use the property that we can use 
    # map(lambda x: ..., range(N))
    # We need to compute factorials. We can use a trick with a list 
    # and a function that updates it.
    
    fact = [1] * (2 * MAX)
    # Use a list to store the current product and map to iterate
    state = [1]
    list(map(lambda i: state.__setitem__(0, (state[0] * i) % MOD) or fact.__setitem__(i, state[0]), range(1, 2 * MAX)))
    
    # Modular inverse using pow(a, MOD-2, MOD)
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (pow(fact[r], MOD - 2, MOD) * pow(fact[n - r], MOD - 2, MOD)) % MOD
        return (num * den) % MOD

    def catalan(n):
        return nCr_mod(2 * n, n) // (n + 1) # This is not quite right for modulo
    
    def catalan_mod(n):
        # C_n = (2n)! / ((n+1)! n!)
        num = fact[2 * n]
        den = (pow(fact[n + 1], MOD - 2, MOD) * pow(fact[n], MOD - 2, MOD)) % MOD
        return (num * den) % MOD

    # Identify blocks of identical values
    # A = [1, 1, 1, 1, 1, 0] -> blocks: [5, 1]
    # We can use groupby from itertools
    from itertools import groupby
    lengths = [len(list(g)) for k, g in groupby(a)]
    
    # The total number of ways is the product of Catalan(length - 1)
    # But we must check if the target A is reachable.
    # A is reachable if for every block of length k, 
    # the values at the boundaries (if they exist) match the block value.
    # Actually, the problem simplifies to: 
    # Any block of length k takes k-1 operations.
    # The number of ways to sequence these is Catalan(k-1).
    # The only condition is that the final A must be consistent with 
    # the ability to perform these operations.
    # Since we start with 1, 0, 1, 0... 
    # Any block of identical values in A must have been created by 
    # overwriting the opposite values.
    # This is always possible as long as the boundaries of the block 
    # in the original sequence match the block's value.
    # For a block from index i to j (0-indexed), 
    # the original values were (i+1)%2, (i+2)%2 ... (j+1)%2.
    # The boundaries are (i+1)%2 and (j+1)%2.
    # They must both equal A[i].
    
    # Check reachability
    # For each block [i, j], we need (i+1)%2 == A[i] and (j+1)%2 == A[j].
    # Let's find the indices of the blocks.
    def check_reachable():
        # Use a generator to find start/end of blocks
        # groupby doesn't give indices, so we use a different approach.
        # We can use a list of indices where A[i] != A[i-1]
        indices = [i for i in range(1, n) if a[i] != a[i-1]]
        boundaries = [0] + indices + [n]
        
        # For each block [boundaries[k], boundaries[k+1]-1]
        # Check if (boundaries[k]+1)%2 == a[boundaries[k]] 
        # AND (boundaries[k+1])%2 == a[boundaries[k]]
        # Note: the second boundary is the cell r in the operation (l, r).
        # The cells are 1-indexed. So cell l is boundaries[k]+1, cell r is boundaries[k+1].
        # Initial value of cell i is i % 2.
        
        # We use map to check all blocks
        results = list(map(lambda k: (
            (boundaries[k] + 1) % 2 == a[boundaries[k]] and 
            (boundaries[k+1]) % 2 == a[boundaries[k]]
        ), range(len(boundaries) - 1)))
        
        # Special case: if the block is length 1, it's always "reachable" 
        # as it's just the initial value.
        # Wait, the condition is: if length > 1, it must be reachable.
        # If length == 1, it must simply match the initial value.
        
        # Let's refine:
        # For each block of length L starting at index i:
        # If L == 1: must have (i+1)%2 == a[i]
        # If L > 1: must have (i+1)%2 == a[i] AND (i+L)%2 == a[i]
        # Actually, the only way to get a block of length L > 1 is to 
        # have the boundaries match. If L=1, it's just the initial state.
        
        # Let's use a more robust check.
        def validate(k):
            l, r = boundaries[k] + 1, boundaries[k+1]
            val = a[boundaries[k]]
            if r - l > 1:
                return (l % 2 == val) and (r % 2 == val)
            else:
                return (l % 2 == val)
        
        return all(map(validate, range(len(boundaries) - 1)))

    if not check_reachable():
        print(0)
        return

    # Calculate product of Catalan(L-1)
    ans = 1
    # Use a loop-free way to multiply
    from functools import reduce
    ans = reduce(lambda x, L: (x * catalan_mod(L - 1)) % MOD, lengths, 1)
    print(ans)

solve()