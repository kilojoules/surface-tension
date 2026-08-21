import sys
from itertools import groupby

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    
    # Initial state X_i = i % 2 (using 1-based indexing)
    # X = [1, 0, 1, 0, ...] if N=6
    # The operation allows replacing a range (l, r) with X[l] if X[l] == X[r] 
    # and all X[i] for l < i < r are different from X[l].
    # This is essentially a grammar-like reduction. 
    # A block of identical values A_i can be formed if the initial values 
    # were alternating and we "collapsed" them.
    # The number of ways to form a block of length 'k' of the same value 
    # starting from an alternating sequence is given by the 
    # (k-1)-th Catalan number if we view the operations as a binary tree 
    # of range replacements.
    # However, the constraint is simpler: we can only replace if the 
    # middle elements are DIFFERENT. 
    # This means we can only collapse blocks of size 2 (e.g., 1 0 1 -> 1 1 1).
    # The number of ways to reduce a segment of length k to a single value 
    # via these specific rules is C_{k-1}.
    
    # Let's group the target array A into blocks of identical consecutive values.
    # Example: 1 1 1 1 1 0 -> blocks of length 5 and 1.
    blocks = [len(list(g)) for k, g in groupby(a)]
    
    # The total number of ways is the product of the number of ways to form 
    # each block. 
    # For a block of length k, the number of ways to form it is the 
    # (k-1)-th Catalan number.
    # Catalan(n) = (2n)! / ((n+1)! n!)
    
    MOD = 998244353
    
    # Precompute factorials for Catalan numbers
    # Max k is N, so we need factorials up to 2*N
    fact = [1] * (2 * n + 1)
    for i in range(2, 2 * n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
        
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (pow(fact[r], MOD - 2, MOD) * pow(fact[n - r], MOD - 2, MOD)) % MOD
        return (num * den) % MOD

    def catalan(k):
        # We need the (k-1)-th Catalan number
        # C_n = 1/(n+1) * (2n choose n)
        n_cat = k - 1
        if n_cat < 0: return 0
        if n_cat == 0: return 1
        return (nCr_mod(2 * n_cat, n_cat) * pow(n_cat + 1, MOD - 2, MOD)) % MOD

    # The answer is the product of catalan(k) for all block lengths k.
    # Note: If the target A is unreachable, the answer should be 0.
    # A is unreachable if any A_i != (i+1)%2 is not part of a collapsed block.
    # But the problem asks for the number of sequences. 
    # If A_i is impossible to reach, the product logic might fail.
    # However, the only way to change a value is the operation.
    # The operation requires X[l] == X[r]. 
    # In an alternating sequence, X[l] == X[r] iff r-l is even.
    # This means we can only collapse blocks of odd length (relative to original).
    # Actually, the parity of the index determines the value.
    # If A_i is the same for a range, and the original was 1 0 1 0...
    # we can only reach A_i if the original values at the boundaries were the same.
    
    # Check if A is reachable:
    # For each block of identical values in A, the original values at the 
    # start and end of the block must be equal to the value of the block.
    # Original X_i = i % 2 (1-indexed).
    
    # We can use a list comprehension to check reachability and calculate product.
    # We'll use a helper to get the original value: (index + 1) % 2
    
    # To track indices, we can't use groupby alone. Let's use a manual loop.
    # But we can use a generator to find blocks and their start/end indices.
    
    def get_blocks(arr):
        # Returns list of (value, length, start_index)
        # Using a list comprehension to simulate a loop for block identification
        # This is tricky in Python without loops. Let's use a trick with 
        # a mutable state object or just use the fact that we can 
_       process the array.
        pass

    # Since I must avoid loops, I'll use a recursive-like structure via map/reduce
    # or just use the property that we only need to check the boundaries of 
    # the blocks against the original X_i.
    
    # Let's redefine: 
    # A block from index i to j (0-indexed) is reachable if:
    # X[i] == A[i] and X[j] == A[j] and X[i] == X[j].
    # Where X[i] = (i + 1) % 2.
    
    # To implement this without loops, we can use a list comprehension 
    # that iterates through the grouped blocks.
    
    # We can get the blocks and their indices using a clever groupby:
    # We create a list of (value, index) pairs.
    indexed_a = list(enumerate(a))
    # Group by value, but only if they are consecutive.
    # We can do this by grouping by (value, index // block_size) - no, that's not it.
    # The standard way to group consecutive identical elements is:
    # groupby(indexed_a, key=lambda x: x[1])
    
    # Let's use a list comprehension to extract the blocks' metadata.
    # Since we can't use loops, we use a generator expression inside a list.
    # We can track the current block's start index by passing it through 
    # a reduction or by calculating it from the lengths of previous blocks.
    
    # Wait, the simplest way to check if A is reachable:
    # For every block of identical values in A, the original values at 
    # the start and end of the block must be the same as the block's value.
    # If any block fails this, the answer is 0.
    
    # To do this without loops:
    # 1. Get lengths of identical blocks: `lengths = [len(list(g)) for k, g in groupby(a)]`
    # 2. Calculate start indices: `starts = [sum(lengths[:i]) for i in range(len(lengths))]`
    # 3. Check: `all( (starts[i]+1)%2 == a[starts[i]] and (starts[i]+lengths[i])%2 == a[starts[i]] for i in range(len(lengths)) )`
    
    # However, `sum(lengths[:i])` is O(N^2). Let's use `itertools.accumulate`.
    from itertools import accumulate
    
    lengths = [len(list(g)) for k, g in groupby(a)]
    starts = list(accumulate(lengths, lambda x, y: x)) # This is wrong
    # Correct accumulate for starts:
    # We need 0, L1, L1+L2, ...
    # We can use:
    starts = list(accumulate([0] + lengths[:-1]))
    
    # Check reachability
    # For each block i:
    # Original value at start: (starts[i] + 1) % 2
    # Original value at end: (starts[i] + lengths[i] - 1 + 1) % 2 = (starts[i] + lengths[i]) % 2
    # Both must equal a[starts[i]]
    
    reachable = all(
        (starts[i] + 1) % 2 == a[starts[i]] and 
        (starts[i] + lengths[i]) % 2 == a[starts[i]]
        for i in range(len(lengths))
    )
    
    if not reachable:
        print(0)
        return

    # If reachable, the answer is the product of Catalan(length)
    # We use a generator expression and the built-in product (via reduce)
    from functools import reduce
    ans = reduce(lambda x, y: (x * y) % MOD, map(catalan, lengths), 1)
    print(ans)

solve()