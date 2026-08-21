import sys
from itertools import groupby

def solve():
    # Read N and the target sequence A
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # The initial state is X_i = i % 2.
    # The operation allows us to fill a range (l, r) with the value of X_l if X_l == X_r
    # and all values in between are different.
    # This is essentially a grammar-like reduction. 
    # We can group consecutive identical elements in A.
    # Let the groups be G_1, G_2, ..., G_k.
    # If a group G_j has length L_j, it means L_j cells now have the same value.
    # One of these cells must have been the 'source' (the l in the operation).
    # For a block of length L, there are L-1 possible operations that could have 
    # created it if we view it as a binary tree of operations.
    # However, the constraint is that the middle elements must be DIFFERENT.
    # This means we can only merge blocks of the form: [val, opposite, val].
    # The number of ways to form a block of length L is the (L-1)-th Catalan number
    # if we could merge any, but here we can only merge if the middle is different.
    # Actually, the problem simplifies to: 
    # Each contiguous block of identical values in A of length L 
    # can be formed in C(L-1) ways, where C is the Catalan number.
    # But wait, the operation requires the middle to be DIFFERENT.
    # If we have a block of 1s, the only way to get it is to have had 0s in between.
    # The structure of the operations forms a rooted tree.
    # For a block of length L, the number of ways to form it is the 
    # number of binary trees with L leaves, which is the (L-1)-th Catalan number.
    # The total number of ways is the product of Catalan(L_i - 1) for all blocks i.
    
    # Let's refine: 
    # If a block has length 1, there's 1 way (it was already that value).
    # If a block has length L > 1, it must have been formed by an operation (l, r).
    # The cells l and r must have the same value, and everything in between must be different.
    # This implies the middle part was a single block of the opposite value.
    # So a block of length L is formed by:
    # [block of length i] [block of opposite value] [block of length L-i]
    # This is the recurrence for Catalan numbers: C_n = sum(C_i * C_{n-1-i})
    # The number of ways to form a block of length L is Catalan(L-1).
    
    # Group the target sequence A into contiguous identical blocks
    blocks = [list(g) for k, g in groupby(A)]
    lengths = [len(b) for b in blocks]
    
    # We need Catalan(L-1) for each length L in lengths.
    # Max L is N. We need to precompute Catalan numbers up to N.
    # C_n = (2n)! / ((n+1)! n!)
    # We can use the formula C_n = C_{n-1} * (4n-2) / (n+1)
    
    # Since we need the product of Catalan(L-1), and the total sum of L is N,
    # we can just compute the necessary ones.
    
    # To avoid loops, we can use a functional approach to generate Catalan numbers.
    # But the constraint allows any number of operations.
    # The total ways is Product(Catalan(L_i - 1))
    
    # Let's use a list comprehension to calculate the product.
    # We can use the formula C_n = comb(2n, n) // (n + 1)
    # Since we need it modulo 998244353, we use modular inverse.
    
    MOD = 998244353
    
    # Precompute factorials for combinations
    # Using a list comprehension to simulate a loop for factorial
    # However, N is 2*10^5, so we must be careful with recursion limits.
    # We can use a trick with map/reduce or a generator.
    
    # Instead of full precomputation, let's use the property that 
    # we only need C_{L-1} for the specific lengths in the input.
    # But we can't use loops. We can use a generator expression.
    
    # Let's use the formula C_n = (2n)! / ((n+1)! n!)
    # We can compute all factorials using a scan-like approach.
    # In Python, we can use a list comprehension with a helper function 
    # or use the fact that we can't use loops.
    
    # Actually, the most reliable way to do "loops" without 'for' or 'while'
    # is using map(), reduce(), or recursion (with increased limit).
    
    import functools
    
    # Compute factorials using reduce
    # fact[i] = i! % MOD
    # We need up to 2 * max(lengths)
    max_l = max(lengths) if lengths else 0
    fact = list(functools.reduce(lambda x, _: x + [x[-1] * (len(x) - 1) % MOD], 
                                range(2 * max_l + 1), [1]))
    
    # Correcting the reduce: the range should be 2*max_l, and the lambda 
    # needs to handle the index correctly.
    # Let's redefine:
    def get_facts(n):
        return functools.reduce(lambda acc, i: acc + [acc[-1] * i % MOD], 
                                range(1, n + 1), [1])

    facts = get_facts(2 * max_l)
    
    def inv(n):
        return pow(n, MOD - 2, MOD)

    def nCr_mod(n, r):
        if r < 0 or r > n: return 0
        num = facts[n]
        den = (facts[r] * facts[n - r]) % MOD
        return (num * inv(den)) % MOD

    def catalan(n):
        return (nCr_mod(2 * n, n) * inv(n + 1)) % MOD

    # The answer is the product of catalan(L-1) for all L in lengths
    ans = functools.reduce(lambda a, b: (a * b) % MOD, 
                           map(lambda L: catalan(L - 1), lengths), 1)
    
    print(ans)

if __name__ == "__main__":
    solve()