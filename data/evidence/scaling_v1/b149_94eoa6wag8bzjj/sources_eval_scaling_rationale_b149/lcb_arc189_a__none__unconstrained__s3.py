import sys
from functools import reduce

def solve():
    # Increase recursion depth for deep structures, though not using recursion here
    sys.setrecursionlimit(300000)
    
    # Read N and the sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    a = list(map(int, input_data[1:]))
    
    MOD = 998244353

    # The problem asks for the number of sequences of operations.
    # An operation (l, r) is possible if A[l] == A[r] and all A[i] for l < i < r 
    # are different from A[l]. This means we are filling gaps of 0s with 1s or vice versa.
    # This structure is recursive. A block of k identical elements can be formed in
    # Catalan(k-1) ways if the underlying structure was alternating.
    # Specifically, if we have a block of length k, the number of ways to form it
    # is the (k-1)-th Catalan number.
    
    # Precompute factorials for Catalan numbers
    # Catalan(n) = (2n)! / ((n+1)! n!)
    MAX = n + 1
    fact = [1] * (2 * MAX)
    inv = [1] * (2 * MAX)
    
    # Using reduce to compute factorials without for-loops
    fact = reduce(lambda acc, i: acc + [acc[-1] * i % MOD], range(1, 2 * MAX), [1])
    
    # Compute inverse using pow(a, MOD-2, MOD)
    inv[2 * MAX - 1] = pow(fact[2 * MAX - 1], MOD - 2, MOD)
    inv = reduce(lambda acc, i: acc + [pow(fact[i], MOD - 2, MOD)], range(2 * MAX - 2, -1, -1), [])
    # The reduce above for inv is tricky because it goes backwards. 
    # Let's redefine inv properly.
    
    # Correct way to compute inverse factorials without loops:
    def get_inv_fact(n, mod, f):
        last_inv = pow(f[n], mod - 2, mod)
        # We need to go from n down to 0. 
        # Since we can't use loops, we use a list comprehension and reverse.
        # But we need the previous value. We can use a trick with a mutable list.
        res = [0] * (n + 1)
        res[n] = last_inv
        # We can't use a loop to fill res. Let's use a different approach for Catalan.
        return res

    # Since we cannot use loops, we use the formula C(n) = comb(2n, n) / (n+1)
    def nCr_mod(n, r, fact, mod):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (pow(fact[r], mod - 2, mod) * pow(fact[n - r], mod - 2, mod)) % mod
        return (num * den) % mod

    def catalan(k):
        # C(k) = comb(2k, k) / (k+1)
        return (nCr_mod(2 * k, k, fact, MOD) * pow(k + 1, MOD - 2, MOD)) % MOD

    # Group A into blocks of identical consecutive elements
    # Example: 1 1 1 1 1 0 -> [(1, 5), (0, 1)]
    def group_blocks(seq):
        if not seq: return []
        # Use a custom reduce to group identical elements
        grouped = reduce(lambda acc, x: acc + [acc[-1] + 1] if acc and x == seq[acc[-1][0]-1] 
                         else acc + [[x, 1]], seq, [])
        # The above logic is slightly flawed. Let's use a more robust grouping.
        return grouped

    # Correct grouping using a helper
    def get_groups(seq):
        if not seq: return []
        # We use a list to store (value, length)
        def step(acc, x):
            if not acc: return [(x, 1)]
            val, length = acc[-1]
            if x == val:
                # Replace last element with updated length
                return acc[:-1] + [(val, length + 1)]
            return acc + [(x, 1)]
        return reduce(step, seq, [])

    groups = get_groups(a)
    
    # The number of ways to form the final configuration is the product of 
    # Catalan(length - 1) for each block, BUT only if the block could 
    # have been formed by operations.
    # An operation is only possible if the elements being replaced were different.
    # The initial state is 1, 0, 1, 0... (i mod 2).
    # This means we can only merge blocks if they match the parity of their indices.
    
    # Let's check if the target A is reachable.
    # A[i] must be reachable from (i+1)%2.
    # The only way to change a value is via the operation.
    # The operation replaces A[l+1...r-1] with A[l], where A[l] == A[r].
    # This means we can only create blocks of the same value.
    # Crucially, the problem implies we start from X_i = i % 2.
    # For a block of length k to be formed, it must have been possible to 
    # "bridge" the gap.
    
    # The number of ways to form a block of length k is Catalan(k-1).
    # The total ways is the product of Catalan(len-1) for all blocks.
    # However, we must verify if the configuration is reachable.
    # A configuration is reachable if and only if for every block of identical 
    # values A[i...j], the values at the boundaries (if they exist) 
    # allow the operations.
    # Actually, the problem simplifies to: the answer is the product of 
    # Catalan(len-1) for all blocks, provided the final state is reachable.
    # A state is reachable if we don't have to change the values of the 
    # endpoints of the grid in a way that's impossible.
    # But the endpoints A[0] and A[N-1] can be anything? 
    # No, the operation requires l and r. The values at l and r are never changed.
    # So A[i] must be (i+1)%2 for all i that act as 'l' or 'r' in the last operations.
    # Wait, the simplest condition: the final state is reachable if and only if
    # for all i, A[i] == (i+1)%2 or it was covered by an operation.
    # This is always possible as long as we don't need to change A[0] or A[N-1]
    # to something other than their initial values, because they can never be 
    # the "inside" (l+1...r-1) of an operation that changes them.
    # Actually, A[0] and A[N-1] can be changed if they are the "inside" of 
    # an operation, but l >= 1 and r <= N. 
    # The indices are 1 to N. The operation is on l and r.
    # The cells being replaced are l+1 to r-1.
    # Cell 1 and Cell N can NEVER be replaced.
    # Therefore, A[0] must be 1 % 2 = 1 and A[N-1] must be N % 2.
    
    # Check reachability
    def is_reachable():
        if a[0] != 1 % 2: return False
        if a[-1] != n % 2: return False
        return True

    if not is_reachable():
        print(0)
        return

    # The answer is the product of Catalan(length - 1) for all blocks.
    # Note: blocks of length 1 have Catalan(0) = 1.
    ans = reduce(lambda acc, group: (acc * catalan(group[1] - 1)) % MOD, groups, 1)
    print(ans)

solve()