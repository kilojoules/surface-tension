import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    MOD = 998244353

    # The operation: choose l, r where X[l] == X[r] and X[i] != X[l] for l < i < r.
    # This replaces the middle segment with X[l].
    # This is equivalent to merging adjacent blocks of identical values.
    # Specifically, if we have a sequence of blocks (B1, B2, ..., Bk),
    # an operation merges Bi, Bi+1, Bi+2 into one block if Bi and Bi+2 have the same value.
    # The target state A consists of m blocks of identical values.
    # The initial state consists of N blocks of size 1 (alternating 0, 1).
    # To reach A from initial, we must perform (N - m) operations.
    # Each operation reduces the number of blocks by 2.
    # Wait, the constraint is: X[i] != X[l] for l < i < r.
    # This means we can only merge three consecutive blocks (color X, color Y, color X) into one (color X).
    # This is exactly the process of reducing a word in a free product of groups, 
    # or more simply, a stack-based reduction.
    
    # Let's represent A as a sequence of block lengths.
    # A = [1, 1, 1, 1, 1, 0] -> blocks: (1, 5), (0, 1)
    # Initial: (1, 1), (0, 1), (1, 1), (0, 1), (1, 1), (0, 1)
    
    # The number of ways to form a block of length L using this operation is 
    # the Catalan-like number. Specifically, if we have a sequence of alternating 
    # blocks of length 1, to merge them into one block of length L, 
    # we need (L-1)//2 operations.
    # The number of ways to do this is the (L-1)//2-th Catalan number? 
    # No, it's the number of binary trees.
    # For a block of length L, the number of ways to collapse the internal 
    # alternating blocks is C_{(L-1)//2}, where C_n is the n-th Catalan number.
    # However, we must also consider that we have multiple blocks in A.
    
    # Let the blocks of A be (val1, len1), (val2, len2), ..., (valm, lenm).
    # For each block i, we need to perform (leni - 1) // 2 operations.
    # Total operations = sum((leni - 1) // 2).
    # The total number of ways is (Total Ops)! / product( (leni-1)//2 ! ) * product(C_{(leni-1)//2})
    # But the operations can be interleaved.
    # Actually, the operations within one block are independent of others, 
    # EXCEPT that an operation requires l and r to be the boundaries.
    # The correct combinatorial result for this specific problem is:
    # Total ways = (Total Ops)! / product( (leni-1)//2 ! ) * product( (ways to form block i) )
    # Where ways to form block i is the number of ways to reduce a string of length leni 
    # of alternating characters to a single character using the given rule.
    # This is known to be the Catalan number C_k where k = (leni-1)//2.
    # C_k = (2k)! / (k! (k+1)!)
    
    # First, validate if A is reachable.
    # A is reachable if and only if A_i can be produced from the alternating sequence.
    # The alternating sequence is X_i = i % 2.
    # This means A_i must be consistent with the parity of the block boundaries.
    # Let's check:
    # Block 1: A[0...len1-1], Block 2: A[len1...len1+len2-1], etc.
    # The first element of block k must be the same as the (sum_{j=1}^{k-1} lenj)-th element of X.
    
    # Check validity:
    # X_i = i % 2 (1-indexed). So X_0 = 0, X_1 = 1, X_2 = 0... (0-indexed in Python)
    # Wait, the problem says cell i (1 <= i <= N) has i % 2.
    # So cell 1: 1, cell 2: 0, cell 3: 1, cell 4: 0...
    # A[0] must be 1 % 2 = 1. If A[0] == 0, it's impossible? 
    # No, we can change A[0] if we can find an l < 0, but l >= 1.
    # So A[0] must be 1. Similarly, A[N-1] must be N % 2.
    
    # Let's re-evaluate: the only way to change X_i is to be the middle of an operation.
    # The endpoints l and r are never changed.
    # Thus, A[0] must equal X[0] and A[N-1] must equal X[N-1].
    # Also, each block in A must have an odd length because we remove 2 elements at a time.
    # Exception: the last block can be shortened if the operations allow.
    # Actually, the rule is: we can merge (X, Y, X) -> (X). This reduces length by 2.
    # So len_i must have the same parity as the length of the segment of X it replaced.
    # Since X is alternating, any segment of length L has (L+1)//2 of one char and L//2 of other.
    # To get a block of length L of char V, we need to start with a segment of X that 
    # starts and ends with V and has length 2k + 1.
    
    # Correct logic:
    # A is reachable iff:
    # 1. A[0] == 1 % 2
    # 2. A[N-1] == N % 2
    # 3. Each block of identical characters in A has odd length.
    
    # Let's group A into blocks.
    from itertools import groupby
    blocks = [(val, len(list(group))) for val, group in groupby(A)]
    
    # Validation
    if A[0] != 1 or A[-1] != (N % 2):
        print(0)
        return
    
    for val, length in blocks:
        if length % 2 == 0:
            print(0)
            return

    # Combinatorics:
    # Let k_i = (length_i - 1) // 2.
    # Total ways = (sum k_i)! / product(k_i!) * product(C_{k_i})
    # = (sum k_i)! / product(k_i!) * product( (2k_i)! / (k_i! (k_i+1)!) )
    # = (sum k_i)! * product( (2k_i)! / (k_i!^2 * (k_i+1)) )
    
    K = [ (length - 1) // 2 for val, length in blocks ]
    sum_K = sum(K)
    
    # Precompute factorials
    MAX_VAL = 2 * N + 1
    fact = [1] * MAX_VAL
    inv = [1] * MAX_VAL
    for i in range(1, MAX_VAL):
        fact[i] = (fact[i-1] * i) % MOD
        
    inv[MAX_VAL-1] = pow(fact[MAX_VAL-1], MOD - 2, MOD)
    for i in range(MAX_VAL-2, -1, -1):
        inv[i] = (inv[i+1] * (i + 1)) % MOD
        
    def nCr_mod(n, r):
        if r < 0 or r > n:
            return 0
        num = fact[n]
        den = (inv[r] * inv[n-r]) % MOD
        return (num * den) % MOD

    def catalan(k):
        return nCr_mod(2*k, k) * pow(k + 1, MOD - 2, MOD) % MOD

    # Result = (sum_K)! * product( catalan(k_i) / k_i! )
    # Note: The formula for interleaving independent sequences of operations is 
    # (Total Steps)! / product(Steps_i!) * product(Ways_i)
    
    ans = fact[sum_K]
    for k in K:
        term = (catalan(k) * inv[k]) % MOD
        ans = (ans * term) % MOD
        
    print(ans % MOD)

solve()