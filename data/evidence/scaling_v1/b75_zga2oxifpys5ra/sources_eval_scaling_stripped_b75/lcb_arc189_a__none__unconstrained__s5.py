import sys
from functools import reduce

def solve():
    # Read N and the target sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # Initial state X_i = i % 2 (1-indexed)
    # X = [1%2, 2%2, 3%2, ...] = [1, 0, 1, 0, ...]
    # The operation: if X[l] == X[r] and for all l < i < r, X[i] != X[l],
    # then set X[i] = X[l] for l < i < r.
    # This is essentially a range-fill operation that can only happen if the 
    # range is currently alternating.
    
    # Key insight: This problem is equivalent to counting ways to build the 
    # final configuration using a stack-based approach or parentheses matching.
    # A block of identical values A_i...A_j can be formed if they were 
    # "covered" by an operation.
    # The only way to change a value is if it's between two identical values.
    # This looks like a Dyck path / Catalan structure.
    # Specifically, we are looking for the number of ways to reduce the 
    # initial alternating sequence to the target sequence.
    
    # Let's group the target A into blocks of identical values.
    # Example: 1 1 1 1 1 0 -> blocks: (1, len 5), (0, len 1)
    # The initial sequence is 1 0 1 0 1 0...
    # To get a block of k identical values, we need (k-1) operations.
    # The number of ways to form a block of length k is the (k-1)-th Catalan number?
    # No, the operations are nested. The number of ways to collapse a segment 
    # of length k into a single value is C_{k-1}.
    
    # Let's refine: 
    # A sequence of operations is valid if it transforms X_init to A.
    # This is possible if and only if A_i is consistent with the parity 
    # of the operations.
    # Actually, the problem is simpler: we can only perform an operation (l, r)
    # if X[l] == X[r] and all X[i] for l < i < r are different from X[l].
    # This means the segment [l, r] must be X[l], NOT X[l], X[l].
    # This is exactly the structure of a binary tree or nested parentheses.
    # For a block of length k of the same character, there are C_{k-1} ways 
    # to form it.
    
    # However, we must check if the target A is even reachable.
    # The only way to change a value is to wrap it in two identical values.
    # The parity of the indices matters. 
    # Initial: X_i = i % 2.
    # An operation (l, r) requires X_l == X_r. 
    # Since X_i = i % 2, this means l % 2 == r % 2, so (r - l) must be even.
    # The number of elements changed is (r - l - 1), which is odd.
    
    # Wait, the sample 1: 1 1 1 1 1 0. 
    # Initial: 1 0 1 0 1 0.
    # Op 1: l=2, r=4. X[2]=0, X[4]=0. X becomes 1 0 0 0 1 0.
    # Op 2: l=1, r=5. X[1]=1, X[5]=1. X becomes 1 1 1 1 1 0.
    # This is valid.
    
    # The number of ways to form a block of length k is the (k-1)-th Catalan number.
    # The total number of ways is the product of C_{k-1} for each block of 
    # identical values in A, provided the configuration is reachable.
    # Is it always reachable? 
    # If A_i != (i % 2), it must have been changed by an operation.
    # An operation (l, r) makes all i in (l, r) have value X_l.
    # This is only possible if the block of identical values in A 
    # "covers" the parity changes.
    
    # Actually, the condition for reachability is simply that the 
    # first and last elements of the grid cannot be changed 
    # unless they are part of a larger range. But l and r are the 
    # boundaries. The values at the boundaries of the operation 
    # do not change.
    # Thus, A_i must be equal to (i % 2) for all i that were never 
    # inside any (l, r).
    # But the problem says we can perform any number of operations.
    # The only invariant is that we can't change X_i if we can't find 
    # l < i < r with X_l == X_r.
    # But we can always pick l=i-1 and r=i+1 if X_{i-1} == X_{i+1}.
    # Since X is 1 0 1 0..., X_{i-1} is always equal to X_{i+1}.
    # So we can always change any X_i (for 1 < i < N).
    # The only values that can NEVER change are X_1 and X_N.
    # So we must have A_1 == (1 % 2) and A_N == (N % 2).
    # If not, the answer is 0.
    
    if A[0] != (1 % 2) or A[-1] != (N % 2):
        print(0)
        return

    # Group A into blocks of identical values
    # Example: 1 1 1 1 1 0 -> blocks of lengths [5, 1]
    blocks = [reduce(lambda x, _: x + 1, group, 0) 
              for _, group in __import__('itertools').groupby(A)]
    
    # Catalan number C_n = (2n)! / ((n+1)! n!)
    # We need product of C_{k-1} for k in blocks.
    # Max k is 2*10^5, so we precompute factorials.
    
    def nCr_mod(n, r, fact, inv_fact):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv_fact[r] * inv_fact[n - r]) % MOD
        return (num * den) % MOD

    def catalan(n, fact, inv_fact):
        return nCr_mod(2 * n, n, fact, inv_fact) // (n + 1)
    
    # To avoid loops, use map/reduce for factorials
    fact = [1] * (2 * N + 1)
    for i in range(2, 2 * N + 1):
        fact[i] = (fact[i-1] * i) % MOD
        
    inv_fact = [1] * (2 * N + 1)
    inv_fact[2 * N] = pow(fact[2 * N], MOD - 2, MOD)
    for i in range(2 * N - 1, -1, -1):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD

    # C_n = (2n)! / ((n+1)! n!)
    def get_catalan(n):
        if n < 0: return 0
        return (fact[2 * n] * inv_fact[n + 1] * inv_fact[n]) % MOD

    # The answer is the product of C_{k-1} for each block length k
    # Note: a block of length 1 means k-1 = 0, C_0 = 1.
    ans = reduce(lambda x, k: (x * get_catalan(k - 1)) % MOD, blocks, 1)
    print(ans)

solve()